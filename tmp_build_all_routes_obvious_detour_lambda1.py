from pathlib import Path

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


def main():
    intersection_time_scale = 2.0
    dt = 2.0
    headway = 2.0
    lambda_path = 1.0
    requests = [
        (1, 3, 9, 0.0),
        (2, 6, 12, 0.0),
        (3, 9, 3, 0.0),
        (4, 12, 6, 0.0),
        (5, 4, 10, 0.0),
        (6, 10, 4, 0.0),
    ]

    set_trajectory_conflict_filter(False)
    tmap = TrafficMap.paper_3x3(
        intersection_time_scale=intersection_time_scale,
    )
    relaxed_plans = apply_relaxed_entrance_headway(
        make_relaxed_vehicle_plans(tmap, requests, Dt=dt),
        headway=headway,
    )
    codesign = search_dynamic_codesign_dfs_bb(
        relaxed_plans,
        lambda_path=lambda_path,
        branch_and_bound=True,
        verbose=True,
    )

    shortest_plans = apply_entrance_headway(
        [
            build_vehicle_plan(
                tmap,
                vehicle_id=vehicle_id,
                entrance=entrance,
                exit=exit_,
                alpha0=alpha0,
                road_time=dt,
            )
            for vehicle_id, entrance, exit_, alpha0 in requests
        ],
        headway=headway,
    )
    shortest = search_dfs_bb(
        shortest_plans,
        branch_and_bound=True,
        verbose=False,
    )

    selections = []
    for plan, candidates in zip(relaxed_plans, codesign.best_node.route_candidates):
        selected = plan.route_options[candidates[0]]
        shortest_route = tmap.shortest_route_option(
            plan.entrance,
            plan.exit,
            road_time=dt,
        )
        selections.append(
            (
                plan.vehicle_id,
                shortest_route,
                selected,
                route_free_time(selected, road_time=dt)
                - route_free_time(shortest_route, road_time=dt),
            )
        )

    out_dir = Path("output/counterexample_3x3_all_routes_obvious_detour_lambda1")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_interactive_solution_html(
        codesign,
        out_dir / "codesign_interactive_solution.html",
        plans=relaxed_plans,
        tmap=tmap,
        max_terminal_paths=None,
        max_tree_nodes=2000,
        lambda_path=lambda_path,
    )
    write_interactive_solution_html(
        shortest,
        out_dir / "shortest_schedule_interactive_solution.html",
        plans=shortest_plans,
        tmap=tmap,
        max_terminal_paths=None,
        max_tree_nodes=2000,
        lambda_path=lambda_path,
    )

    with open(out_dir / "case_summary.txt", "w", encoding="utf-8") as summary:
        summary.write("3x3 all-routes obvious-detour bottleneck example\n")
        summary.write("trajectory_conflict_filter=False\n")
        summary.write("All six robots retain all route options.\n")
        summary.write("branch_and_bound=True (exact optimum; pruned search tree)\n")
        summary.write("HTML tree visualization is capped at 2000 nodes.\n")
        summary.write(f"requests={requests}\n")
        summary.write(f"lambda_path={lambda_path}\n")
        summary.write(f"intersection_time_scale={intersection_time_scale}\n")
        summary.write(f"Dt={dt}\n")
        summary.write(f"codesign_best_g={codesign.best_g}\n")
        summary.write(f"codesign_delay={codesign.best_node.g_delay}\n")
        summary.write(f"codesign_path_cost={codesign.best_node.g_path}\n")
        summary.write(f"shortest_schedule_best_g={shortest.best_g}\n")
        summary.write(f"improvement={shortest.best_g - codesign.best_g}\n")
        summary.write(f"codesign_node_count={len(codesign.nodes)}\n")
        summary.write(f"codesign_leaf_count={len(codesign.leaves)}\n")
        for vehicle_id, shortest_route, selected, extra in selections:
            summary.write(
                f"V{vehicle_id}: shortest={shortest_route.intersections} "
                f"({len(shortest_route.intersections)} intersections), "
                f"selected={selected.intersections} "
                f"({len(selected.intersections)} intersections), "
                f"path_extra={extra}\n"
            )

    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
