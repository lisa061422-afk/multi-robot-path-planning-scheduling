import os
import sys
import webbrowser
from dataclasses import replace
from pathlib import Path

from coarse_scheduler import (
    VehiclePlan,
    apply_entrance_headway,
    apply_relaxed_entrance_headway,
    build_vehicle_plan,
    build_relaxed_vehicle_plan,
    describe_intersection_demands,
    describe_plan,
    export_schedule_mat,
    search_dynamic_codesign_dfs_bb,
    search_dynamic_codesign_parallel_dfs_bb,
    search_parallel_dfs_bb,
    search_dfs_bb,
    write_decision_tree_svg,
    write_interactive_solution_html,
    write_resource_schedule_panel_html,
    write_resource_schedule_svgs,
)
from traffic_map import TrafficMap
from trajectory_conflicts import set_trajectory_conflict_filter


EPS = 1e-9


def make_vehicle_plans(
    tmap: TrafficMap,
    vehicle_requests: list[tuple],
    *,
    Dt: float,
    fixed_route_policy: str = "manual_or_shortest",
):
    """Build fixed-route plans from vehicle request rows.

    Supported row formats:
      (vehicle_id, entrance, exit, alpha0)
      (vehicle_id, entrance, exit, alpha0, fixed_intersections)

    fixed_route_policy:
      "manual_or_shortest": use fixed_intersections if supplied; otherwise shortest.
      "shortest": ignore fixed_intersections and always use current shortest rule.
    """

    plans = []
    for row in vehicle_requests:
        if len(row) not in (4, 5):
            raise ValueError(f"bad vehicle request row: {row}")

        vehicle_id, entrance, exit_, alpha0 = row[:4]
        if fixed_route_policy == "shortest" or len(row) == 4:
            plans.append(
                build_vehicle_plan(
                    tmap,
                    vehicle_id=vehicle_id,
                    entrance=entrance,
                    exit=exit_,
                    alpha0=alpha0,
                    road_time=Dt,
                )
            )
        elif fixed_route_policy == "manual_or_shortest":
            fixed_intersections = row[4]
            fixed_intersections = tuple(fixed_intersections)
            options = tmap.route_options(entrance, exit_)
            route = next(
                (opt for opt in options if opt.intersections == fixed_intersections),
                None,
            )
            if route is None:
                available = [opt.intersections for opt in options]
                raise ValueError(
                    f"vehicle {vehicle_id}: path {fixed_intersections} is not "
                    f"available for P{entrance}->P{exit_}; options={available}"
                )
            plans.append(
                VehiclePlan(
                    vehicle_id=vehicle_id,
                    entrance=entrance,
                    exit=exit_,
                    route=route,
                    alpha0=alpha0,
                    road_time=Dt,
                )
            )
        else:
            raise ValueError(f"unknown fixed_route_policy: {fixed_route_policy}")
    return plans


def make_relaxed_path_requests(
    tmap: TrafficMap,
    vehicle_requests: list[tuple],
):
    """Build OD requests with all path options for future co-design scheduling."""

    requests = []
    for row in vehicle_requests:
        if len(row) not in (4, 5):
            raise ValueError(f"bad vehicle request row: {row}")
        vehicle_id, entrance, exit_, alpha0 = row[:4]
        requests.append(
            {
                "vehicle_id": vehicle_id,
                "entrance": entrance,
                "exit": exit_,
                "alpha0": alpha0,
                "route_options": tmap.route_options(entrance, exit_),
            }
        )
    return tuple(requests)


def make_relaxed_vehicle_plans(
    tmap: TrafficMap,
    vehicle_requests: list[tuple],
    *,
    Dt: float,
):
    plans = []
    for row in vehicle_requests:
        if len(row) not in (4, 5):
            raise ValueError(f"bad vehicle request row: {row}")
        vehicle_id, entrance, exit_, alpha0 = row[:4]
        plans.append(
            build_relaxed_vehicle_plan(
                tmap,
                vehicle_id=vehicle_id,
                entrance=entrance,
                exit=exit_,
                alpha0=alpha0,
                road_time=Dt,
            )
        )
    return plans


def route_free_time(route, *, road_time: float) -> float:
    return sum(route.execution_times) + len(route.edges) * road_time


def shortest_path_delay_upper_bound(
    tmap: TrafficMap,
    vehicle_requests: list[tuple],
    *,
    Dt: float,
    T_headway: float,
    fixed_route_policy: str = "shortest",
) -> float:
    """Return a feasible upper bound from shortest-path fixed scheduling."""

    fixed_plans = make_vehicle_plans(
        tmap,
        vehicle_requests,
        Dt=Dt,
        fixed_route_policy=fixed_route_policy,
    )
    fixed_plans = apply_entrance_headway(fixed_plans, headway=T_headway)
    result = search_dfs_bb(
        fixed_plans,
        branch_and_bound=True,
        verbose=False,
    )
    return result.best_g


def filter_relaxed_plans_by_upper_bound(
    relaxed_plans,
    *,
    lambda_path: float,
    J_ub: float,
    keep_min_hop_options: bool = False,
):
    """Remove route options that cannot beat a known feasible upper bound.

    For each vehicle, the shortest/free-flow baseline path remains available.
    Any alternative with lambda_path * (T_path - T_shortest) > J_ub cannot
    appear in a globally better co-design solution because total delay savings
    are bounded above by the feasible incumbent objective J_ub.

    If keep_min_hop_options is true, every option with the minimum number of
    intersections is retained for path-selection visualization/training, even
    when turn costs make it slightly worse than the free-time shortest option.
    """

    filtered = []
    for plan in relaxed_plans:
        free_times = [
            route_free_time(option, road_time=plan.road_time)
            for option in plan.route_options
        ]
        base_time = min(free_times)
        min_hops = min(len(option.intersections) for option in plan.route_options)
        kept = tuple(
            option
            for option, free_time in zip(plan.route_options, free_times)
            if (
                lambda_path * max(0.0, free_time - base_time) <= J_ub + EPS
                or (
                    keep_min_hop_options
                    and len(option.intersections) == min_hops
                )
            )
        )
        if not kept:
            best_index = min(range(len(free_times)), key=free_times.__getitem__)
            kept = (plan.route_options[best_index],)
        filtered.append(replace(plan, route_options=kept))
    return filtered


def open_output_file(path: str | Path) -> bool:
    """Open a generated output file with the operating system default app."""

    resolved = Path(path).resolve()
    try:
        if sys.platform.startswith("win"):
            os.startfile(resolved)  # type: ignore[attr-defined]
        else:
            webbrowser.open(resolved.as_uri())
    except OSError as exc:
        print(f"  could not open {resolved}: {exc}")
        return False
    return True


def build_fixed_map(
    fixed_map: str,
    *,
    intersection_time_scale: float = 1.0,
) -> TrafficMap:
    if fixed_map == "paper_2x2":
        return TrafficMap.paper_2x2(
            intersection_time_scale=intersection_time_scale,
        )
    if fixed_map == "paper_3x3":
        return TrafficMap.paper_3x3(
            intersection_time_scale=intersection_time_scale,
        )
    raise ValueError(f"unknown fixed_map: {fixed_map}")


def default_vehicle_requests(fixed_map: str) -> list[tuple]:
    if fixed_map == "paper_2x2":
        return [
            # Reproducible case: lambda=1 co-design advantage with only 3 vehicles.
            #
            # Map: TrafficMap.paper_2x2()
            # Dt = 3.0, T_headway = 2.0, lambda_path = 1.0
            # fixed_route_policy = "shortest"
            #
            # fixed-shortest:
            #   N1 P2->P5: route (I1,I2,I3)
            #   N2 P2->P6: route (I1,I2,I3)
            #   N3 P6->P4: route (I3,I2)
            #   J = total delay = 3.854
            #
            # relaxed co-design:
            #   N1 P2->P5: route (I1,I4,I3), path_extra = +0.858
            #   N2 P2->P6: route (I1,I4,I3)
            #   N3 P6->P4: route (I3,I2)
            #   delay = 0.000, path_extra = 0.858, J = 0.858
            #
            # This demonstrates a non-shortest path choice reducing total cost
            # even when lambda_path = 1.0.
            # (1, 2, 5, 0.0), #0618 example.
            # (2, 2, 6, 0.0),
            # (3, 6, 4, 0.0),
            (1, 2, 5, 0.0),  # N1: P2 -> P5  #0618_2
            (2, 6, 3, 0.0),  # N2: P6 -> P3
            # Previous 3-vehicle example:
            # (1, 2, 7, 0.0, [1, 4]),
            # (3, 1, 4, 0.0, [1, 2]),
            # (4, 5, 2, 0.0, [3, 4, 1]),
            # (5, 5, 3, 0.0, [3, 2]),
            # (6, 6, 3, 0.0, [3, 2]),
            # (7, 6, 3, 0.0, [3, 2]),
            # (8, 6, 5, 0.0, [3]),
            # (9, 6, 8, 0.0, [3, 4]),
            # (10, 6, 3, 0.0, [3, 2]),
            # (11, 1, 4, 0.0, [1, 2]),
            # (12, 1, 2, 0.0, [1]),
            # (13, 1, 6, 0.0, [1, 2, 3]),
        ]

    if fixed_map == "paper_3x3":
        return [
            # 3x3 training case: P1->P7 has path choices after the start
            # intersection, not only at the entrance.
            # (1,3,12,0.0)
            (1, 3, 6, 0.0),
            (2, 4, 8, 0.0),   
            (3, 12, 5, 0.0),  # I4 -> I3, cross traffic through the middle
        ]

    raise ValueError(f"unknown fixed_map: {fixed_map}")


def demo_fixed_map() -> None:
    fixed_map = "paper_3x3"  # "paper_2x2" or "paper_3x3" HERE!!!

    # ======================================================================
    # ===================== EXPERIMENT PARAMETERS ==========================
    # All commonly adjusted parameters are centralized here. Check this section first when changing settings.
    # ======================================================================

    # ----- 1. Timing parameters -----
    intersection_time_scale = 2.0 # multiple two of org 2*[3/4pi, 2,pi/4] 
    Dt = 2.0 
    T_headway = 2.0 

    # ----- 2. Path-planning mode -----
    enable_path_selection = True
    planning_mode = (
        "relaxed_path_selection"
        if enable_path_selection
        else "fixed_path"
    )
    fixed_route_policy = "shortest"  # "manual_or_shortest" or "shortest"
    lambda_path = 1.0
    use_baseline_path_filter = False
    keep_min_hop_route_options = True

    # ----- 3. Decision-tree/search settings -----
    show_all_branches = True
    use_parallel_dynamic = False  # True is faster but harder to debug manually.
    # These limits affect only the HTML viewer; the search result stays complete.
    if show_all_branches:
        max_visualized_paths = 30
        max_visualized_nodes = 2000
    else:
        max_visualized_paths = 100
        max_visualized_nodes = 2000

    # ----- 4. Trajectory-contention model (IMPORTANT) -----
    # False: original one-runner-per-intersection contention model.
    # True: exclude whitelisted non-conflicting trajectory pairs.
    use_trajectory_conflict_filter = False
    set_trajectory_conflict_filter(use_trajectory_conflict_filter)

    # ----- 5. Parallel-search settings -----
    parallel_frontier_depth = 2
    parallel_max_workers = 4

    # ----- 6. Vehicle/sample settings -----
    full_tree_vehicle_count = 3
    vehicle_requests = default_vehicle_requests(fixed_map)

    # ======================================================================
    # =================== END EXPERIMENT PARAMETERS ========================
    # ======================================================================

    tmap = build_fixed_map(
        fixed_map,
        intersection_time_scale=intersection_time_scale,
    )
    svg_path = tmap.write_svg(f"output/{tmap.name}_map.svg")

    print(f"Map: {tmap.name}")
    print(f"Map drawing: {svg_path}")
    print(f"Intersection time scale: {intersection_time_scale:.3f}")
    print("Intersections:", tmap.intersection_ids)
    print("Roads/buffers:")
    for line in tmap.describe_roads():
        print("  " + line)
    print("Entrances/exits:")
    for line in tmap.describe_ports():
        print("  " + line)

    run_vehicle_requests = (
        vehicle_requests[:full_tree_vehicle_count]
        if show_all_branches
        else vehicle_requests
    )

    entrance = run_vehicle_requests[0][1]
    exit_ = run_vehicle_requests[0][2]
    vehicle_routes = tmap.vehicle_route_options(
        vehicle_id=1,
        entrance=entrance,
        exit=exit_,
    )

    print()
    print(
        "Trajectory contention filter: "
        f"{'enabled' if use_trajectory_conflict_filter else 'disabled'}"
    )
    print("Selected OD:")
    print("  entrance " + tmap.describe_port(entrance))
    print("  exit     " + tmap.describe_port(exit_))
    print(
        f"Vehicle {vehicle_routes.vehicle_id}: "
        f"port {entrance} -> port {exit_}"
    )
    print(f"Route options: {len(vehicle_routes.options)}")
    for option in vehicle_routes.options:
        print(
            f"  option {option.id}: resources {option.resource_sequence}, "
            f"execution_times {option.execution_times}"
        )
        for traversal in option.traversals:
            print(
                "    "
                f"I{traversal.intersection}: "
                f"{traversal.entry_dir}->{traversal.exit_dir}, "
                f"{traversal.turn}, "
                f"routeId {traversal.route_id}, "
                f"C={traversal.execution_time:.3f}"
            )
        for branch in option.branch_points:
            print(
                "    branch at "
                f"intersection {branch.intersection}: "
                f"next choices {branch.next_intersections}"
            )

    shortest = tmap.shortest_route_option(entrance, exit_, road_time=Dt)
    print()
    print("Predetermined shortest route for fixed-path scheduling:")
    print(
        f"  option {shortest.id}: intersections {shortest.intersections}, "
        f"T_free={sum(shortest.execution_times) + len(shortest.edges) * Dt:.3f}"
    )

    if planning_mode == "fixed_path":
        plans = make_vehicle_plans(
            tmap,
            run_vehicle_requests,
            Dt=Dt,
            fixed_route_policy=fixed_route_policy,
        )
    elif planning_mode == "relaxed_path_selection":
        relaxed_plans = make_relaxed_vehicle_plans(
            tmap,
            run_vehicle_requests,
            Dt=Dt,
        )
        if use_baseline_path_filter:
            J_ub = shortest_path_delay_upper_bound(
                tmap,
                run_vehicle_requests,
                Dt=Dt,
                T_headway=T_headway,
                fixed_route_policy="shortest",
            )
            before_counts = [len(plan.route_options) for plan in relaxed_plans]
            relaxed_plans = filter_relaxed_plans_by_upper_bound(
                relaxed_plans,
                lambda_path=lambda_path,
                J_ub=J_ub,
                keep_min_hop_options=keep_min_hop_route_options,
            )
            after_counts = [len(plan.route_options) for plan in relaxed_plans]
            print()
            print(
                "Baseline path filter: "
                f"J_ub={J_ub:.3f}, lambda_path={lambda_path:.3f}, "
                f"keep_min_hop={keep_min_hop_route_options}, "
                f"route options {before_counts} -> {after_counts}"
            )
        else:
            print()
            print("Baseline path filter: disabled; using all route options")
        relaxed_plans = apply_relaxed_entrance_headway(
            relaxed_plans,
            headway=T_headway,
        )
        print()
        print("Run mode: relaxed path-selection co-design schedule")
        print(f"lambda_path={lambda_path:.3f}, T_headway={T_headway:.1f}s")
        print("Relaxed route options:")
        for plan in relaxed_plans:
            options = [
                option.intersections
                for option in plan.route_options
            ]
            print(
                f"  N{plan.vehicle_id}: "
                f"P{plan.entrance}->P{plan.exit}, alpha0={plan.alpha0:.3f}, "
                f"options={options}"
            )
        if use_parallel_dynamic:
            result = search_dynamic_codesign_parallel_dfs_bb(
                relaxed_plans,
                lambda_path=lambda_path,
                frontier_depth=parallel_frontier_depth,
                max_workers=parallel_max_workers,
                branch_and_bound=not show_all_branches,
                verbose=True,
            )
        else:
            result = search_dynamic_codesign_dfs_bb(
                relaxed_plans,
                lambda_path=lambda_path,
                branch_and_bound=not show_all_branches,
                verbose=True,
            )
        print(f"  best_J = delay + lambda*path_extra = {result.best_g:.3f}")
        print(
            f"  best delay={result.best_node.g_delay:.3f}, "
            f"path_extra={result.best_node.g_path:.3f}"
        )
        print(f"  nodes={len(result.nodes)}, leaves={len(result.leaves)}")
        interactive_path = write_interactive_solution_html(
            result,
            "output/relaxed_interactive_solution.html",
            plans=relaxed_plans,
            tmap=tmap,
            max_terminal_paths=max_visualized_paths,
            max_tree_nodes=max_visualized_nodes,
            lambda_path=lambda_path,
        )
        print(f"  interactive relaxed solution viewer: {interactive_path}")
        open_output_file(interactive_path)
        print("  best relaxed schedule:")
        for seg in result.best_schedule:
            print(
                "  "
                f"vehicle {seg.vehicle_id}, task {seg.task_index}, "
                f"I{seg.resource}: request {seg.requested_time:.3f}, "
                f"start {seg.start_time:.3f}, end {seg.end_time:.3f}, "
                f"delay {seg.delay:.3f}"
            )
        return
    else:
        raise ValueError(f"unknown planning_mode: {planning_mode}")

    plans = apply_entrance_headway(plans, headway=T_headway)
    print()
    if show_all_branches:
        print(
            "Run mode: fixed-path full decision tree preview "
            f"using first {len(run_vehicle_requests)} vehicles "
            f"(fixed_route_policy={fixed_route_policy})"
        )
    else:
        print(
            "Run mode: fixed-path parallel compact search using "
            f"{len(run_vehicle_requests)} vehicles "
            f"(fixed_route_policy={fixed_route_policy})"
        )
    print()
    print(f"Fixed vehicle plans with entrance headway T_headway={T_headway:.1f}s:")
    for plan in plans:
        print("  " + describe_plan(plan))

    print()
    print("Intersection demand from fixed routes:")
    for line in describe_intersection_demands(plans):
        print("  " + line)

    print()
    print("Centralized DFS schedule, each intersection is one resource:")
    if show_all_branches:
        result = search_dfs_bb(
            plans,
            branch_and_bound=False,
            verbose=True,
        )
    else:
        result = search_parallel_dfs_bb(
            plans,
            frontier_depth=1,
            max_workers=2,
            verbose=True,
        )
    tree_path = write_decision_tree_svg(
        result,
        "output/coarse_decision_tree.svg",
        plans=plans,
    )
    schedule_paths = write_resource_schedule_svgs(
        result,
        "output",
        plans=plans,
    )
    mat_path = export_schedule_mat(
        result,
        "output/coarse_schedule_data.mat",
        plans=plans,
        tmap=tmap,
    )
    interactive_path = write_interactive_solution_html(
        result,
        "output/coarse_interactive_solution.html",
        plans=plans,
        tmap=tmap,
        max_terminal_paths=max_visualized_paths,
        max_tree_nodes=max_visualized_nodes,
        lambda_path=lambda_path,
    )
    panel_path = write_resource_schedule_panel_html(
        schedule_paths,
        "output/coarse_schedule_panel.html",
        tmap=tmap,
    )
    print(f"  best_g = total intersection delay = {result.best_g:.3f}")
    print(f"  nodes={len(result.nodes)}, leaves={len(result.leaves)}")
    print(f"  decision tree drawing: {tree_path}")
    print("  local schedule drawings:")
    for schedule_path in schedule_paths:
        print(f"    {schedule_path}")
    print(f"  MATLAB schedule data: {mat_path}")
    print(f"  local schedule panel: {panel_path}")
    print(f"  interactive solution viewer: {interactive_path}")
    open_output_file(interactive_path)
    for seg in result.best_schedule:
        print(
            "  "
            f"vehicle {seg.vehicle_id}, task {seg.task_index}, "
            f"I{seg.resource}: request {seg.requested_time:.3f}, "
            f"start {seg.start_time:.3f}, end {seg.end_time:.3f}, "
            f"delay {seg.delay:.3f}"
        )


if __name__ == "__main__":
    demo_fixed_map()
