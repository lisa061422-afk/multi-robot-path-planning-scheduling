"""Check exact all-route subsets of the known 3x3 bottleneck construction."""

from itertools import combinations
from pathlib import Path

from traffic_map import TrafficMap
from trajectory_conflicts import set_trajectory_conflict_filter
from tmp_find_scale2_single_intersection_cases import (
    obvious_detour_selections,
    save_case,
    solve_case,
)


def main():
    set_trajectory_conflict_filter(False)
    tmap = TrafficMap.paper_3x3(intersection_time_scale=2.0)
    base = (
        (1, 3, 9, 0.0),
        (2, 6, 12, 0.0),
        (3, 9, 3, 0.0),
        (4, 12, 6, 0.0),
        (5, 4, 10, 0.0),
        (6, 10, 4, 0.0),
    )
    out_dir = Path("output/scale2_3x3_obvious_detour_counterexamples")
    out_dir.mkdir(parents=True, exist_ok=True)
    found = 0
    for vehicle_count in (3, 4, 5):
        for requests in combinations(base, vehicle_count):
            plans, fixed, codesign, shortest = solve_case(
                tmap,
                requests,
                road_time=2.0,
                headway=2.0,
            )
            changed = obvious_detour_selections(
                tmap,
                plans,
                codesign,
                road_time=2.0,
            )
            if changed and codesign.best_g < shortest.best_g - 1e-9:
                found += 1
                print(f"FOUND {found} requests={requests}", flush=True)
                print(
                    f"codesign={codesign.best_g} delay={codesign.best_node.g_delay} "
                    f"path={codesign.best_node.g_path} shortest={shortest.best_g}",
                    flush=True,
                )
                for vehicle_id, short, selected, extra in changed:
                    print(
                        f"V{vehicle_id}: {short.intersections} -> "
                        f"{selected.intersections}; "
                        f"hop_extra={len(selected.intersections)-len(short.intersections)}; "
                        f"path_extra={extra}",
                        flush=True,
                    )
                settings = {
                    "map": "paper_3x3",
                    "vehicle_count": vehicle_count,
                    "intersection_time_scale": 2.0,
                    "road_time": 2.0,
                    "headway": 2.0,
                    "objective": "delay+path_extra",
                    "trajectory_conflict_filter": False,
                    "branch_and_bound": True,
                    "all_route_options_retained": True,
                    "minimum_detour_intersections": 1,
                }
                case_dir = save_case(
                    out_dir,
                    case_number=found,
                    trial=found,
                    settings=settings,
                    requests=requests,
                    tmap=tmap,
                    relaxed_plans=plans,
                    fixed_plans=fixed,
                    codesign=codesign,
                    shortest=shortest,
                    changed=changed,
                )
                print(f"saved={case_dir}", flush=True)
                if found >= 5:
                    return
    print(f"done found={found}", flush=True)


if __name__ == "__main__":
    main()
