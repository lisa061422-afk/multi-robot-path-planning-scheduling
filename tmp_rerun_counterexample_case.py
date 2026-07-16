from pathlib import Path

from coarse_scheduler import (
    apply_relaxed_entrance_headway,
    search_dynamic_codesign_dfs_bb,
    write_interactive_solution_html,
)
from main import make_relaxed_vehicle_plans, shortest_path_delay_upper_bound
from traffic_map import TrafficMap
from trajectory_conflicts import set_trajectory_conflict_filter


def main():
    tmap = TrafficMap.paper_3x3()
    Dt = 3.0
    T_headway = 2.0
    lambda_path = 1.0
    requests = [
        (1, 8, 2, 0.0),
        (2, 4, 11, 0.0),
        (3, 1, 6, 0.0),
    ]

    set_trajectory_conflict_filter(False)
    relaxed_plans = make_relaxed_vehicle_plans(tmap, requests, Dt=Dt)
    relaxed_plans = apply_relaxed_entrance_headway(
        relaxed_plans,
        headway=T_headway,
    )

    result = search_dynamic_codesign_dfs_bb(
        relaxed_plans,
        lambda_path=lambda_path,
        branch_and_bound=False,
        verbose=True,
    )

    fixed_shortest_g = shortest_path_delay_upper_bound(
        tmap,
        requests,
        Dt=Dt,
        T_headway=T_headway,
        fixed_route_policy="shortest",
    )

    out_dir = Path(
        "output/counterexample_3x3_relaxed_vs_shortest_lambda1_bottleneck"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    html = out_dir / "relaxed_interactive_solution.html"

    interactive_path = write_interactive_solution_html(
        result,
        html,
        plans=relaxed_plans,
        tmap=tmap,
        max_terminal_paths=None,
        max_tree_nodes=None,
        lambda_path=lambda_path,
    )

    with open(out_dir / "case_summary.txt", "w", encoding="utf-8") as f:
        f.write("counterexample rerun (no omission, branch_and_bound=False)\n")
        f.write("trajectory_conflict_filter=False\n")
        f.write(f"requests={requests}\n")
        f.write(f"lambda_path={lambda_path}\n")
        f.write(f"best_g={result.best_g}\n")
        f.write(f"best_delay={result.best_node.g_delay}\n")
        f.write(f"best_path={result.best_node.g_path}\n")
        f.write(f"route_candidates={result.best_node.route_candidates}\n")
        f.write(f"node_count={len(result.nodes)}\n")
        f.write(f"leaf_count={len(result.leaves)}\n")
        f.write(f"fixed_shortest_best_g={fixed_shortest_g}\n")
        f.write(f"improvement={fixed_shortest_g - result.best_g}\n")

    print(f"rerun complete: {interactive_path}")
    print(f"best_g={result.best_g}, nodes={len(result.nodes)}, leaves={len(result.leaves)}")


if __name__ == "__main__":
    main()
