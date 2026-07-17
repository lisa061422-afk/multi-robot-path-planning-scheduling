"""Find and save exact 3x3 scale-2, one-runner-per-intersection cases."""

from dataclasses import asdict
import json
from pathlib import Path
import random

from coarse_scheduler import (
    apply_entrance_headway,
    apply_relaxed_entrance_headway,
    build_vehicle_plan,
    search_dfs_bb,
    search_dynamic_codesign_dfs_bb,
    write_interactive_solution_html,
)
from main import make_relaxed_vehicle_plans, route_free_time
from traffic_map import TrafficMap
from trajectory_conflicts import set_trajectory_conflict_filter


def solve_case(tmap, requests, *, road_time, headway, lambda_path):
    relaxed = apply_relaxed_entrance_headway(
        make_relaxed_vehicle_plans(tmap, requests, Dt=road_time),
        headway=headway,
    )
    codesign = search_dynamic_codesign_dfs_bb(
        relaxed,
        lambda_path=lambda_path,
        branch_and_bound=True,
        verbose=False,
    )
    fixed = apply_entrance_headway(
        [
            build_vehicle_plan(
                tmap,
                vehicle_id=vehicle_id,
                entrance=entrance,
                exit=exit_,
                alpha0=alpha0,
                road_time=road_time,
            )
            for vehicle_id, entrance, exit_, alpha0 in requests
        ],
        headway=headway,
    )
    shortest = search_dfs_bb(fixed, branch_and_bound=True, verbose=False)
    return relaxed, fixed, codesign, shortest


def obvious_detour_selections(tmap, plans, result, *, road_time):
    changed = []
    for plan, candidates in zip(plans, result.best_node.route_candidates):
        selected = plan.route_options[candidates[0]]
        shortest = tmap.shortest_route_option(
            plan.entrance,
            plan.exit,
            road_time=road_time,
        )
        extra = route_free_time(selected, road_time=road_time) - route_free_time(
            shortest,
            road_time=road_time,
        )
        if len(selected.intersections) >= len(shortest.intersections) + 1:
            changed.append((plan.vehicle_id, shortest, selected, extra))
    return changed


def save_case(
    out_dir,
    *,
    case_number,
    trial,
    settings,
    requests,
    tmap,
    relaxed_plans,
    fixed_plans,
    codesign,
    shortest,
    changed,
):
    case_dir = out_dir / f"case_{case_number:02d}"
    case_dir.mkdir(parents=True, exist_ok=True)

    write_interactive_solution_html(
        codesign,
        case_dir / "codesign_interactive_solution.html",
        plans=relaxed_plans,
        tmap=tmap,
        max_terminal_paths=300,
        max_tree_nodes=8000,
        lambda_path=settings["lambda_path"],
    )
    write_interactive_solution_html(
        shortest,
        case_dir / "shortest_path_schedule_interactive_solution.html",
        plans=fixed_plans,
        tmap=tmap,
        max_terminal_paths=300,
        max_tree_nodes=8000,
        lambda_path=settings["lambda_path"],
    )

    selections = []
    for plan, candidates in zip(relaxed_plans, codesign.best_node.route_candidates):
        selected = plan.route_options[candidates[0]]
        shortest_route = tmap.shortest_route_option(
            plan.entrance,
            plan.exit,
            road_time=settings["road_time"],
        )
        selections.append(
            {
                "vehicle_id": plan.vehicle_id,
                "entrance": plan.entrance,
                "exit": plan.exit,
                "shortest_intersections": list(shortest_route.intersections),
                "selected_intersections": list(selected.intersections),
                "shortest_free_time": route_free_time(
                    shortest_route,
                    road_time=settings["road_time"],
                ),
                "selected_free_time": route_free_time(
                    selected,
                    road_time=settings["road_time"],
                ),
            }
        )

    data = {
        "case_number": case_number,
        "search_trial": trial,
        "settings": settings,
        "requests": [list(row) for row in requests],
        "codesign": {
            "objective": codesign.best_g,
            "delay": codesign.best_node.g_delay,
            "path_extra": codesign.best_node.g_path,
            "node_count": len(codesign.nodes),
            "leaf_count": len(codesign.leaves),
            "schedule": [asdict(segment) for segment in codesign.best_schedule],
        },
        "shortest_path_schedule": {
            "objective": shortest.best_g,
            "node_count": len(shortest.nodes),
            "leaf_count": len(shortest.leaves),
            "schedule": [asdict(segment) for segment in shortest.best_schedule],
        },
        "improvement": shortest.best_g - codesign.best_g,
        "route_selections": selections,
    }
    (case_dir / "case_data.json").write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )

    summary_lines = [
        f"Scale-2 single-intersection-resource counterexample {case_number}",
        f"search_trial={trial}",
        "trajectory_conflict_filter=False",
        "branch_and_bound=True",
        f"settings={settings}",
        f"requests={list(requests)}",
        f"codesign_J={codesign.best_g}",
        f"codesign_delay={codesign.best_node.g_delay}",
        f"codesign_path_extra={codesign.best_node.g_path}",
        f"shortest_path_J={shortest.best_g}",
        f"improvement={shortest.best_g - codesign.best_g}",
    ]
    for vehicle_id, shortest_route, selected, extra in changed:
        summary_lines.append(
            f"V{vehicle_id}: shortest={shortest_route.intersections}, "
            f"selected={selected.intersections}, path_extra={extra}"
        )
    (case_dir / "case_summary.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )
    return case_dir


def main():
    set_trajectory_conflict_filter(False)
    scale = 2.0
    road_time = 2.0
    headway = 2.0
    lambda_path = 1.0
    settings = {
        "map": "paper_3x3",
        "vehicle_count": 4,
        "minimum_detour_intersections": 1,
        "intersection_time_scale": scale,
        "road_time": road_time,
        "headway": headway,
        "lambda_path": lambda_path,
        "trajectory_conflict_filter": False,
        "branch_and_bound": True,
    }
    out_dir = Path("output/scale2_3x3_obvious_detour_counterexamples")
    out_dir.mkdir(parents=True, exist_ok=True)
    tmap = TrafficMap.paper_3x3(intersection_time_scale=scale)
    ods = [
        (entrance, exit_)
        for entrance in tmap.port_ids
        for exit_ in tmap.port_ids
        if entrance != exit_
        # These OD pairs have exactly eight complete simple paths. Keeping all
        # eight preserves exactness while avoiding the more expensive 10--12
        # option OD pairs during this deterministic counterexample search.
        and len(tmap.route_options(entrance, exit_)) == 8
    ]

    rng = random.Random(20260716)
    found = 0
    seen = set()
    for trial in range(1, 2001):
        requests = tuple(
            (vehicle_id, *rng.choice(ods), 0.0)
            for vehicle_id in range(1, 5)
        )
        signature = tuple((row[1], row[2]) for row in requests)
        if signature in seen:
            continue
        seen.add(signature)
        plans, fixed_plans, codesign, shortest = solve_case(
            tmap,
            requests,
            road_time=road_time,
            headway=headway,
            lambda_path=lambda_path,
        )
        changed = obvious_detour_selections(
            tmap,
            plans,
            codesign,
            road_time=road_time,
        )
        if not changed or codesign.best_g >= shortest.best_g - 1e-9:
            continue

        found += 1
        print(f"CASE {found} trial={trial}")
        print(f"requests={list(requests)}")
        print(
            f"codesign_J={codesign.best_g:.12f} "
            f"delay={codesign.best_node.g_delay:.12f} "
            f"path_extra={codesign.best_node.g_path:.12f}"
        )
        print(
            f"shortest_J={shortest.best_g:.12f} "
            f"improvement={shortest.best_g - codesign.best_g:.12f}"
        )
        for vehicle_id, shortest_route, selected, extra in changed:
            print(
                f"V{vehicle_id}: shortest={shortest_route.intersections} "
                f"selected={selected.intersections} "
                f"hop_extra={len(selected.intersections) - len(shortest_route.intersections)} "
                f"path_extra={extra:.12f}"
            )
        case_dir = save_case(
            out_dir,
            case_number=found,
            trial=trial,
            settings=settings,
            requests=requests,
            tmap=tmap,
            relaxed_plans=plans,
            fixed_plans=fixed_plans,
            codesign=codesign,
            shortest=shortest,
            changed=changed,
        )
        print(f"saved={case_dir}")
        print()
        if found >= 5:
            break

    print(
        f"done found={found} trials={trial} scale={scale} Dt={road_time} "
        f"headway={headway} lambda={lambda_path} conflict_filter=False"
    )


if __name__ == "__main__":
    main()
