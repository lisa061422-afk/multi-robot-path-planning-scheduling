from dataclasses import replace
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
    tmap = TrafficMap.paper_3x3(
        intersection_time_scale=intersection_time_scale,
    )
    dt = 2.0
    headway = 2.0
    requests = [
        (1, 3, 9, 0.0),
        (2, 6, 12, 0.0),
        (3, 9, 3, 0.0),
        (4, 12, 6, 0.0),
        (5, 4, 10, 0.0),
        (6, 10, 4, 0.0),
    ]

    set_trajectory_conflict_filter(False)

    relaxed_plans = apply_relaxed_entrance_headway(
        make_relaxed_vehicle_plans(tmap, requests, Dt=dt),
        headway=headway,
    )

    # V1 is the path-planning vehicle. The other vehicles form fixed shortest-path
    # background traffic through the central bottleneck.
    relaxed_plans = [relaxed_plans[0]] + [
        replace(
            plan,
            route_options=(
                tmap.shortest_route_option(
                    plan.entrance,
                    plan.exit,
                    road_time=dt,
                ),
            ),
        )
        for plan in relaxed_plans[1:]
    ]

    codesign = search_dynamic_codesign_dfs_bb(
        relaxed_plans,
        branch_and_bound=False,
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
        branch_and_bound=False,
        verbose=True,
    )

    selected_index = codesign.best_node.route_candidates[0][0]
    selected_route = relaxed_plans[0].route_options[selected_index]
    shortest_route = tmap.shortest_route_option(3, 9, road_time=dt)
    path_extra = route_free_time(selected_route, road_time=dt) - route_free_time(
        shortest_route,
        road_time=dt,
    )
    improvement = shortest.best_g - codesign.best_g

    out_dir = Path("output/counterexample_3x3_obvious_detour_lambda1")
    out_dir.mkdir(parents=True, exist_ok=True)

    write_interactive_solution_html(
        codesign,
        out_dir / "codesign_interactive_solution.html",
        plans=relaxed_plans,
        tmap=tmap,
        max_terminal_paths=None,
        max_tree_nodes=None,
    )
    write_interactive_solution_html(
        shortest,
        out_dir / "shortest_schedule_interactive_solution.html",
        plans=shortest_plans,
        tmap=tmap,
        max_terminal_paths=None,
        max_tree_nodes=None,
    )

    with open(out_dir / "case_summary.txt", "w", encoding="utf-8") as summary:
        summary.write("3x3 obvious-detour bottleneck example\n")
        summary.write("trajectory_conflict_filter=False\n")
        summary.write("branch_and_bound=False (complete search tree)\n")
        summary.write(f"requests={requests}\n")
        summary.write("objective=delay+path_extra\n")
        summary.write(f"intersection_time_scale={intersection_time_scale}\n")
        summary.write(f"Dt={dt}\n")
        summary.write("V1 is path-selectable; V2-V6 are fixed shortest-path bottleneck traffic.\n")
        summary.write(
            f"V1 shortest_intersections={shortest_route.intersections}, "
            f"count={len(shortest_route.intersections)}\n"
        )
        summary.write(
            f"V1 selected_intersections={selected_route.intersections}, "
            f"count={len(selected_route.intersections)}\n"
        )
        summary.write(f"V1_path_extra={path_extra}\n")
        summary.write(f"codesign_best_g={codesign.best_g}\n")
        summary.write(f"codesign_delay={codesign.best_node.g_delay}\n")
        summary.write(f"codesign_path_cost={codesign.best_node.g_path}\n")
        summary.write(f"shortest_schedule_best_g={shortest.best_g}\n")
        summary.write(f"improvement={improvement}\n")
        summary.write(f"codesign_node_count={len(codesign.nodes)}\n")
        summary.write(f"codesign_leaf_count={len(codesign.leaves)}\n")
        summary.write(f"shortest_node_count={len(shortest.nodes)}\n")
        summary.write(f"shortest_leaf_count={len(shortest.leaves)}\n")

    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
