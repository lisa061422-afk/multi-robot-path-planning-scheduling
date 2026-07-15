"""Coarse centralized decision-tree scheduler.

This is the first fixed-path baseline for the RL/co-design work:

* each intersection is one resource;
* every vehicle follows one predetermined shortest route;
* contention at the same intersection creates DFS branches;
* route choice is not opened inside the tree yet.

The state names intentionally mirror the old CR-MPC code:
`d` is time until the next task can be generated, `r` is the remaining
intersection execution time, `o` is accumulated active/waiting time,
`tw` is the current significant moment, and `ni` is the current task index.
"""

from __future__ import annotations

import itertools
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from scheduler_models import (
    BIG_M,
    EPS,
    AttemptSegment,
    CoarseNode,
    RelaxedNode,
    RelaxedSearchResult,
    RelaxedVehiclePlan,
    ScheduleSegment,
    SearchResult,
    VehiclePlan,
)
from traffic_map import PortId, RouteOption, TrafficMap
from trajectory_conflicts import (
    route_ids_conflict,
    simultaneous_prefix,
    trajectory_conflict_filter_enabled,
)


from coarse_expansion import expand_array, expand_node


def build_vehicle_plan(
    tmap: TrafficMap,
    *,
    vehicle_id: int,
    entrance: PortId,
    exit: PortId,
    alpha0: float = 0.0,
    road_time: float = 3.0,
    max_hops: Optional[int] = None,
    max_paths: Optional[int] = None,
) -> VehiclePlan:
    """Create a fixed-path vehicle plan using the current shortest-route rule."""

    route = tmap.shortest_route_option(
        entrance,
        exit,
        road_time=road_time,
        max_hops=max_hops,
        max_paths=max_paths,
    )
    return VehiclePlan(
        vehicle_id=vehicle_id,
        entrance=entrance,
        exit=exit,
        route=route,
        alpha0=float(alpha0),
        road_time=float(road_time),
    )


def make_root(plans: Sequence[VehiclePlan]) -> CoarseNode:
    """Build the root node for DFS."""

    alpha = tuple(tuple(math.nan for _ in plan.resources) for plan in plans)
    gamma = tuple(tuple(math.nan for _ in plan.resources) for plan in plans)
    return CoarseNode(
        idx=0,
        parent=-1,
        tw=0.0,
        d=tuple(plan.alpha0 for plan in plans),
        r=tuple(0.0 for _ in plans),
        o=tuple(0.0 for _ in plans),
        ni=tuple(0 for _ in plans),
        U_c=tuple(None for _ in plans),
        U_temp=tuple(None for _ in plans),
        g=0.0,
        alpha=alpha,
        gamma=gamma,
        segments=(),
        attempts=(),
        priority_queues=(),
    )


def search_dfs_bb(
    plans: Sequence[VehiclePlan],
    *,
    deadline: Optional[float] = None,
    max_nodes: Optional[int] = None,
    branch_and_bound: bool = True,
    verbose: bool = True,
) -> SearchResult:
    """Run DFS with branch-and-bound over contention choices."""

    root = make_root(plans)
    nodes: List[CoarseNode] = [root]
    leaves: List[int] = []
    stack: List[int] = [0]
    best_g = math.inf
    best_idx = -1
    log: List[str] = []
    pruned = 0
    step = 0
    t0 = time.perf_counter() #real time in pc

    while stack:
        if deadline is not None and time.perf_counter() - t0 > deadline:
            log.append("[DFS_BB] deadline hit; returning best-so-far")
            break
        if max_nodes is not None and len(nodes) >= max_nodes:
            log.append("[DFS_BB] max_nodes hit; returning best-so-far")
            break

        c_idx = stack.pop()
        step += 1
        if branch_and_bound and nodes[c_idx].g >= best_g - EPS:
            pruned += 1
            continue

        children, is_leaf = expand_node(nodes, c_idx, plans)
        if is_leaf:
            leaves.append(c_idx)
            if nodes[c_idx].g < best_g:
                best_g = nodes[c_idx].g
                best_idx = c_idx
                msg = (
                    f"[DFS_BB] step {step}: new best_g={best_g:.6f} "
                    f"at node {c_idx}"
                )
                log.append(msg)
                if verbose:
                    print(msg)
            continue

        children = sorted(children, key=lambda node: node.g, reverse=True)
        for child in children:
            if branch_and_bound and child.g >= best_g - EPS:
                pruned += 1
                continue
            child.idx = len(nodes)
            nodes.append(child)
            stack.append(child.idx)

    summary = (
        f"[DFS_BB] done. expanded={step}, pruned={pruned}, "
        f"nodes={len(nodes)}, leaves={len(leaves)}, best_g={best_g:.6f}"
    )
    log.append(summary)
    if verbose:
        print(summary)
    return SearchResult(
        nodes=tuple(nodes),
        leaves=tuple(leaves),
        best_idx=best_idx,
        best_g=best_g,
        log=tuple(log),
    )


def extract_path(nodes: Sequence[CoarseNode], leaf_idx: int) -> Tuple[int, ...]:
    """Trace node indices from root to a leaf."""

    out: List[int] = []
    cur = leaf_idx
    while cur >= 0:
        out.append(cur)
        cur = nodes[cur].parent
    return tuple(reversed(out))


def describe_plan(plan: VehiclePlan) -> str:
    """One-line route description for printing/debugging."""

    return (
        f"vehicle {plan.vehicle_id}: P{plan.entrance}->P{plan.exit}, "
        f"route {plan.route.intersections}, "
        f"C={tuple(round(x, 3) for x in plan.durations)}, "
        f"T_free={plan.free_flow_time:.3f}"
    )


def apply_entrance_headway(
    plans: Sequence[VehiclePlan],
    *,
    headway: float,
) -> List[VehiclePlan]:
    """Serialize vehicles that enter from the same port."""

    last_by_entrance: Dict[int, float] = {}
    out: List[VehiclePlan] = []
    for plan in plans:
        last = last_by_entrance.get(plan.entrance, -math.inf)
        alpha0 = max(plan.alpha0, last + headway)
        last_by_entrance[plan.entrance] = alpha0
        out.append(replace(plan, alpha0=alpha0))
    return out


def count_intersection_demands(
    plans: Sequence[VehiclePlan],
) -> Dict[int, Tuple[Tuple[int, int], ...]]:
    demand: Dict[int, List[Tuple[int, int]]] = {}
    for plan in plans:
        for task_index, resource in enumerate(plan.resources, start=1):
            demand.setdefault(resource, []).append((plan.vehicle_id, task_index))
    return {resource: tuple(items) for resource, items in sorted(demand.items())}


def describe_intersection_demands(plans: Sequence[VehiclePlan]) -> List[str]:
    lines: List[str] = []
    for resource, items in count_intersection_demands(plans).items():
        vehicles = {vehicle_id for vehicle_id, _task in items}
        task_text = ", ".join(f"N{vehicle_id}(K{task})" for vehicle_id, task in items)
        lines.append(
            f"I{resource}: {len(vehicles)} vehicles, {len(items)} visits -> {task_text}"
        )
    return lines


def search_parallel_dfs_bb(
    plans: Sequence[VehiclePlan],
    *,
    frontier_depth: int = 1,
    max_workers: int = 2,
    verbose: bool = True,
) -> SearchResult:
    """Compatibility wrapper. Uses serial DFS to avoid stressing the machine."""

    return search_dfs_bb(plans, branch_and_bound=True, verbose=verbose)


def build_relaxed_vehicle_plan(
    tmap: TrafficMap,
    *,
    vehicle_id: int,
    entrance: PortId,
    exit: PortId,
    alpha0: float = 0.0,
    road_time: float = 3.0,
    max_hops: Optional[int] = None,
    max_paths: Optional[int] = None,
) -> RelaxedVehiclePlan:
    return RelaxedVehiclePlan(
        vehicle_id=vehicle_id,
        entrance=entrance,
        exit=exit,
        route_options=tmap.route_options(
            entrance,
            exit,
            max_hops=max_hops,
            max_paths=max_paths,
        ),
        alpha0=float(alpha0),
        road_time=float(road_time),
    )


def apply_relaxed_entrance_headway(
    plans: Sequence[RelaxedVehiclePlan],
    *,
    headway: float,
) -> List[RelaxedVehiclePlan]:
    last_by_entrance: Dict[int, float] = {}
    out: List[RelaxedVehiclePlan] = []
    for plan in plans:
        last = last_by_entrance.get(plan.entrance, -math.inf)
        alpha0 = max(plan.alpha0, last + headway)
        last_by_entrance[plan.entrance] = alpha0
        out.append(replace(plan, alpha0=alpha0))
    return out


def _fixed_plan_from_relaxed(plan: RelaxedVehiclePlan, route: RouteOption) -> VehiclePlan:
    return VehiclePlan(
        vehicle_id=plan.vehicle_id,
        entrance=plan.entrance,
        exit=plan.exit,
        route=route,
        alpha0=plan.alpha0,
        road_time=plan.road_time,
    )


def _free_time(route: RouteOption, road_time: float) -> float:
    return sum(route.execution_times) + len(route.edges) * road_time


def _route_option_movements(
    entrance: int,
    exit_port: int,
    option: RouteOption,
) -> List[Dict[str, object]]:
    """Movement-level DAG metadata for one route option.

    Occupation time is attached to the local movement through an intersection,
    not only to the intersection node.  The key information is
    prev -> I_i -> next, because the same I_i can have different execution time
    under different incoming/outgoing directions.
    """

    intersections = list(option.intersections)
    out: List[Dict[str, object]] = []
    for k, traversal in enumerate(option.traversals):
        prev_label = f"B{entrance}" if k == 0 else f"I{intersections[k - 1]}"
        node_label = f"I{traversal.intersection}"
        next_label = (
            f"B{exit_port}"
            if k == len(intersections) - 1
            else f"I{intersections[k + 1]}"
        )
        out.append(
            {
                "prev": prev_label,
                "node": node_label,
                "next": next_label,
                "edge": f"{node_label}->{next_label}",
                "movement": f"{prev_label}->{node_label}->{next_label}",
                "entry_dir": traversal.entry_dir,
                "exit_dir": traversal.exit_dir,
                "turn": traversal.turn,
                "route_id": traversal.route_id,
                "execution_time": traversal.execution_time,
                "path_index": traversal.path_index,
            }
        )
    return out


def search_relaxed_dfs_bb(
    plans: Sequence[RelaxedVehiclePlan],
    *,
    lambda_path: float = 1.0,
    deadline: Optional[float] = None,
    max_nodes: Optional[int] = None,
    branch_and_bound: bool = True,
    verbose: bool = True,
) -> RelaxedSearchResult:
    """Enumerate route choices and attach the fixed-path scheduling DFS under each."""

    nodes: List[RelaxedNode] = [
        RelaxedNode(
            idx=0,
            parent=-1,
            tw=0.0,
            g=0.0,
            g_delay=0.0,
            g_path=0.0,
            segments=(),
            route_choices=(),
            route_candidates=tuple(tuple(range(len(p.route_options))) for p in plans),
        )
    ]
    leaves: List[int] = []
    best_g = math.inf
    best_idx = -1
    log: List[str] = []

    option_ranges = [range(len(plan.route_options)) for plan in plans]
    for choices in itertools.product(*option_ranges):
        fixed_plans = [
            _fixed_plan_from_relaxed(plan, plan.route_options[choice])
            for plan, choice in zip(plans, choices)
        ]
        fixed_result = search_dfs_bb(
            fixed_plans,
            deadline=deadline,
            max_nodes=max_nodes,
            branch_and_bound=branch_and_bound,
            verbose=False,
        )
        base_times = [
            min(_free_time(option, plan.road_time) for option in plan.route_options)
            for plan in plans
        ]
        chosen_times = [
            _free_time(plan.route_options[choice], plan.road_time)
            for plan, choice in zip(plans, choices)
        ]
        g_path = sum(max(0.0, chosen - base) for chosen, base in zip(chosen_times, base_times))
        combo_idx = len(nodes)
        nodes.append(
            RelaxedNode(
                idx=combo_idx,
                parent=0,
                tw=0.0,
                g=lambda_path * g_path,
                g_delay=0.0,
                g_path=g_path,
                segments=(),
                route_choices=tuple(choices),
                route_candidates=tuple((choice,) for choice in choices),
                U_temp=tuple(None for _ in plans),
                ni=tuple(0 for _ in plans),
            )
        )

        index_map = {0: combo_idx}
        for fixed_node in fixed_result.nodes[1:]:
            idx = len(nodes)
            index_map[fixed_node.idx] = idx
            g_delay = fixed_node.g
            g = g_delay + lambda_path * g_path
            node = RelaxedNode(
                idx=idx,
                parent=index_map.get(fixed_node.parent, combo_idx),
                tw=fixed_node.tw,
                g=g,
                g_delay=g_delay,
                g_path=g_path,
                segments=fixed_node.segments,
                route_choices=tuple(choices),
                attempts=fixed_node.attempts,
                route_candidates=tuple((choice,) for choice in choices),
                U_temp=fixed_node.U_temp,
                ni=fixed_node.ni,
            )
            nodes.append(node)

        for leaf in fixed_result.leaves:
            mapped = index_map.get(leaf)
            if mapped is None:
                continue
            leaves.append(mapped)
            node = nodes[mapped]
            if node.g < best_g:
                best_g = node.g
                best_idx = mapped
                msg = f"[Relaxed_DFS_BB] new best_g={best_g:.6f} at node {mapped}"
                log.append(msg)
                if verbose:
                    print(msg)

        if not fixed_result.leaves and fixed_result.best_idx >= 0:
            mapped = index_map.get(fixed_result.best_idx)
            if mapped is not None:
                leaves.append(mapped)
                node = nodes[mapped]
                if node.g < best_g:
                    best_g = node.g
                    best_idx = mapped
                    msg = f"[Relaxed_DFS_BB] new best_g={best_g:.6f} at node {mapped}"
                    log.append(msg)
                    if verbose:
                        print(msg)

    summary = (
        f"[Relaxed_DFS_BB] done. nodes={len(nodes)}, "
        f"leaves={len(leaves)}, best_g={best_g:.6f}"
    )
    log.append(summary)
    if verbose:
        print(summary)
    return RelaxedSearchResult(
        nodes=tuple(nodes),
        leaves=tuple(leaves),
        best_idx=best_idx,
        best_g=best_g,
        log=tuple(log),
    )


def _solve_relaxed_choice_worker(args):
    (
        order,
        choices,
        plans,
        lambda_path,
        deadline,
        max_nodes,
        branch_and_bound,
    ) = args
    fixed_plans = [
        _fixed_plan_from_relaxed(plan, plan.route_options[choice])
        for plan, choice in zip(plans, choices)
    ]
    fixed_result = search_dfs_bb(
        fixed_plans,
        deadline=deadline,
        max_nodes=max_nodes,
        branch_and_bound=branch_and_bound,
        verbose=False,
    )
    base_times = [
        min(_free_time(option, plan.road_time) for option in plan.route_options)
        for plan in plans
    ]
    chosen_times = [
        _free_time(plan.route_options[choice], plan.road_time)
        for plan, choice in zip(plans, choices)
    ]
    g_path = sum(max(0.0, chosen - base) for chosen, base in zip(chosen_times, base_times))
    return order, tuple(choices), fixed_result, g_path, lambda_path * g_path


def search_relaxed_parallel_dfs_bb(
    plans: Sequence[RelaxedVehiclePlan],
    *,
    lambda_path: float = 1.0,
    deadline: Optional[float] = None,
    max_nodes: Optional[int] = None,
    branch_and_bound: bool = True,
    max_workers: int = 2,
    verbose: bool = True,
) -> RelaxedSearchResult:
    """Parallel route-choice enumeration for relaxed path-selection scheduling.

    Each route-choice combination is an independent fixed-path DFS subproblem.
    Parallelizing across combinations preserves global optimality because every
    combination is solved exactly (modulo the same admissible branch-and-bound
    pruning used by the serial fixed-path solver), then the best leaf over all
    combinations is selected.
    """

    option_ranges = [range(len(plan.route_options)) for plan in plans]
    choice_list = [tuple(choice) for choice in itertools.product(*option_ranges)]
    if max_workers <= 1 or len(choice_list) <= 1:
        return search_relaxed_dfs_bb(
            plans,
            lambda_path=lambda_path,
            deadline=deadline,
            max_nodes=max_nodes,
            branch_and_bound=branch_and_bound,
            verbose=verbose,
        )

    workers = max(1, min(int(max_workers), len(choice_list)))
    tasks = [
        (
            order,
            choice,
            tuple(plans),
            lambda_path,
            deadline,
            max_nodes,
            branch_and_bound,
        )
        for order, choice in enumerate(choice_list)
    ]

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        solved = list(pool.map(_solve_relaxed_choice_worker, tasks))
    solved.sort(key=lambda item: item[0])

    nodes: List[RelaxedNode] = [
        RelaxedNode(
            idx=0,
            parent=-1,
            tw=0.0,
            g=0.0,
            g_delay=0.0,
            g_path=0.0,
            segments=(),
            route_choices=(),
            route_candidates=tuple(tuple(range(len(p.route_options))) for p in plans),
        )
    ]
    leaves: List[int] = []
    best_g = math.inf
    best_idx = -1
    log: List[str] = []

    for _order, choices, fixed_result, g_path, initial_g in solved:
        combo_idx = len(nodes)
        nodes.append(
            RelaxedNode(
                idx=combo_idx,
                parent=0,
                tw=0.0,
                g=initial_g,
                g_delay=0.0,
                g_path=g_path,
                segments=(),
                route_choices=tuple(choices),
                route_candidates=tuple((choice,) for choice in choices),
                U_temp=tuple(None for _ in plans),
                ni=tuple(0 for _ in plans),
            )
        )

        index_map = {0: combo_idx}
        for fixed_node in fixed_result.nodes[1:]:
            idx = len(nodes)
            index_map[fixed_node.idx] = idx
            g_delay = fixed_node.g
            node = RelaxedNode(
                idx=idx,
                parent=index_map.get(fixed_node.parent, combo_idx),
                tw=fixed_node.tw,
                g=g_delay + lambda_path * g_path,
                g_delay=g_delay,
                g_path=g_path,
                segments=fixed_node.segments,
                route_choices=tuple(choices),
                attempts=fixed_node.attempts,
                route_candidates=tuple((choice,) for choice in choices),
                U_temp=fixed_node.U_temp,
                ni=fixed_node.ni,
            )
            nodes.append(node)

        mapped_leaves = [index_map[leaf] for leaf in fixed_result.leaves if leaf in index_map]
        if not mapped_leaves and fixed_result.best_idx >= 0 and fixed_result.best_idx in index_map:
            mapped_leaves = [index_map[fixed_result.best_idx]]
        for mapped in mapped_leaves:
            leaves.append(mapped)
            node = nodes[mapped]
            if node.g < best_g:
                best_g = node.g
                best_idx = mapped
                msg = f"[Relaxed_Parallel_DFS_BB] new best_g={best_g:.6f} at node {mapped}"
                log.append(msg)
                if verbose:
                    print(msg)

    summary = (
        f"[Relaxed_Parallel_DFS_BB] done. workers={workers}, "
        f"combos={len(choice_list)}, elapsed={time.perf_counter() - t0:.3f}s, "
        f"nodes={len(nodes)}, leaves={len(leaves)}, best_g={best_g:.6f}"
    )
    log.append(summary)
    if verbose:
        print(summary)
    return RelaxedSearchResult(
        nodes=tuple(nodes),
        leaves=tuple(leaves),
        best_idx=best_idx,
        best_g=best_g,
        log=tuple(log),
    )


def _dynamic_root(plans: Sequence[RelaxedVehiclePlan]) -> RelaxedNode:
    max_tasks = [
        max((len(option.intersections) for option in plan.route_options), default=0)
        for plan in plans
    ]
    alpha = tuple(tuple(math.nan for _ in range(count)) for count in max_tasks)
    gamma = tuple(tuple(math.nan for _ in range(count)) for count in max_tasks)
    return RelaxedNode(
        idx=0,
        parent=-1,
        tw=0.0,
        g=0.0,
        g_delay=0.0,
        g_path=0.0,
        segments=(),
        route_choices=(),
        attempts=(),
        route_candidates=tuple(tuple(range(len(p.route_options))) for p in plans),
        U_temp=tuple(None for _ in plans),
        ni=tuple(0 for _ in plans),
        d=tuple(plan.alpha0 for plan in plans),
        r=tuple(0.0 for _ in plans),
        o=tuple(0.0 for _ in plans),
        alpha=alpha,
        gamma=gamma,
        priority_queues=(),
        path_decisions=(),
    )


def _dynamic_task_count(plan: RelaxedVehiclePlan, candidates: Sequence[int]) -> int:
    if not candidates:
        return 0
    return max(len(plan.route_options[idx].intersections) for idx in candidates)


def _dynamic_traversal_signature(
    plan: RelaxedVehiclePlan,
    option_index: int,
    task_index0: int,
) -> Tuple[int, int, str, str, int, float, str]:
    option = plan.route_options[option_index]
    traversal = option.traversals[task_index0]
    if task_index0 + 1 < len(option.intersections):
        next_label = f"I{option.intersections[task_index0 + 1]}"
    else:
        next_label = f"B{plan.exit}"
    return (
        traversal.intersection,
        traversal.route_id,
        traversal.entry_dir,
        traversal.exit_dir,
        traversal.path_index,
        round(traversal.execution_time, 12),
        next_label,
    )


def _dynamic_task_info(
    plan: RelaxedVehiclePlan,
    candidates: Sequence[int],
    task_index0: int,
) -> Tuple[int, float]:
    option = plan.route_options[candidates[0]]
    traversal = option.traversals[task_index0]
    return traversal.intersection, traversal.execution_time


def _dynamic_groups_for_task(
    plan: RelaxedVehiclePlan,
    candidates: Sequence[int],
    task_index0: int,
) -> Dict[Tuple[int, int, str, str, int, float, str], Tuple[int, ...]]:
    groups: Dict[Tuple[int, int, str, str, int, float, str], List[int]] = {}
    for option_index in candidates:
        option = plan.route_options[option_index]
        if task_index0 >= len(option.intersections):
            continue
        sig = _dynamic_traversal_signature(plan, option_index, task_index0)
        groups.setdefault(sig, []).append(option_index)
    return {sig: tuple(items) for sig, items in groups.items()}


def _dynamic_path_extra(
    plan: RelaxedVehiclePlan,
    before: Sequence[int],
    after: Sequence[int],
) -> float:
    if not before or not after:
        return 0.0
    best_before = min(_free_time(plan.route_options[idx], plan.road_time) for idx in before)
    best_after = min(_free_time(plan.route_options[idx], plan.road_time) for idx in after)
    return max(0.0, best_after - best_before)


def _dynamic_path_decision(
    vehicle_id: int,
    option_display: int,
    signature: Tuple[int, int, str, str, int, float, str],
    tw: float,
    extra: float,
) -> Optional[Tuple[int, int, int, int, float, float]]:
    from_i = signature[0]
    next_label = signature[6]
    if not next_label.startswith("I"):
        return None
    return (vehicle_id, option_display, from_i, int(next_label[1:]), tw, extra)

# path selection
def _dynamic_path_choice_children(
    c: RelaxedNode,
    plans: Sequence[RelaxedVehiclePlan],
    *,
    lambda_path: float,
) -> List[RelaxedNode]:
    simultaneous_choices: List[
        Tuple[
            int,
            RelaxedVehiclePlan,
            Tuple[int, ...],
            List[Tuple[Tuple[int, int, str, str, int, float, str], Tuple[int, ...], float]],
        ]
    ] = []

    for n, plan in enumerate(plans):
        if c.r[n] > EPS:
            continue
        allow_pre_entry_choice = c.ni[n] == 0
        if c.d[n] > EPS and not allow_pre_entry_choice:
            continue
        candidates = c.route_candidates[n]
        task_index0 = c.ni[n]
        if task_index0 >= _dynamic_task_count(plan, candidates):
            continue
        groups = _dynamic_groups_for_task(plan, candidates, task_index0)
        if len(groups) <= 1:
            continue

        options = [
            (
                sig,
                selected_candidates,
                _dynamic_path_extra(plan, candidates, selected_candidates),
            )
            for sig, selected_candidates in sorted(
                groups.items(),
                key=lambda item: (_dynamic_path_extra(plan, candidates, item[1]), item[0]),
                reverse=True,
            )
        ]
        simultaneous_choices.append((n, plan, candidates, options))

    if not simultaneous_choices:
        return []

    children: List[RelaxedNode] = []
    option_products = itertools.product(*(item[3] for item in simultaneous_choices))
    for product in option_products:
        route_candidates = [tuple(items) for items in c.route_candidates]
        path_decisions = list(c.path_decisions)
        total_extra = 0.0

        for (n, plan, _candidates, _options), (sig, selected_candidates, extra) in zip(
            simultaneous_choices,
            product,
        ):
            total_extra += extra
            route_candidates[n] = selected_candidates
            decision = _dynamic_path_decision(
                plan.vehicle_id,
                min(selected_candidates) + 1,
                sig,
                c.tw,
                extra,
            )
            if decision is not None:
                path_decisions.append(decision)

        children.append(
            replace(
                c,
                idx=-1,
                parent=c.idx,
                g=c.g + lambda_path * total_extra,
                g_path=c.g_path + total_extra,
                route_candidates=tuple(route_candidates),
                path_decisions=tuple(path_decisions),
            )
        )

    children.sort(key=lambda node: node.g, reverse=True)
    return children


def _dynamic_current_requests(
    plans: Sequence[RelaxedVehiclePlan],
    candidates_by_vehicle: Sequence[Sequence[int]],
    ra: Sequence[float],
    ni2: Sequence[int],
) -> Tuple[Optional[int], ...]:
    requests: List[Optional[int]] = []
    for n, plan in enumerate(plans):
        if ra[n] <= EPS or ni2[n] < 1:
            requests.append(None)
        else:
            resource, _duration = _dynamic_task_info(
                plan,
                candidates_by_vehicle[n],
                ni2[n] - 1,
            )
            requests.append(resource)
    return tuple(requests)


def _dynamic_valid_running_choices(
    c: RelaxedNode,
    U_c: Tuple[Optional[int], ...],
    ra: Sequence[float],
    ni2: Sequence[int],
    plans: Sequence[RelaxedVehiclePlan],
) -> Tuple[Tuple[Optional[int], ...], ...]:
    by_resource: Dict[int, List[int]] = {}
    for n, resource in enumerate(U_c):
        if resource is not None and ra[n] > EPS:
            by_resource.setdefault(resource, []).append(n)

    queue_by_resource = {resource: list(queue) for resource, queue in c.priority_queues}
    per_resource_options: List[Tuple[int, List[Tuple[int, ...]]]] = []

    for resource, vehicles in sorted(by_resource.items()):
        old_queue = [n for n in queue_by_resource.get(resource, []) if n in vehicles]
        new_vehicles = [n for n in vehicles if n not in old_queue]
        if old_queue:
            if new_vehicles:
                options = []
                for winner in [old_queue[0], *new_vehicles]:
                    if winner == old_queue[0]:
                        queue = [old_queue[0], *new_vehicles, *old_queue[1:]]
                    else:
                        others = [n for n in new_vehicles if n != winner]
                        queue = [winner, *old_queue, *others]
                    options.append(tuple(queue))
            else:
                options = [tuple(old_queue)]
        elif len(vehicles) == 1:
            options = [tuple(vehicles)]
        else:
            options = list(itertools.permutations(vehicles))
        per_resource_options.append((resource, options))

    choices: List[Tuple[Optional[int], ...]] = []
    route_ids = []
    for n, plan in enumerate(plans):
        if ni2[n] < 1:
            route_ids.append(-1)
            continue
        option = plan.route_options[c.route_candidates[n][0]]
        route_ids.append(option.traversals[ni2[n] - 1].route_id)
    products = itertools.product(*(item[1] for item in per_resource_options))
    for queues in products:
        U_temp: List[Optional[int]] = [None for _ in plans]
        for (resource, _options), queue in zip(per_resource_options, queues):
            for n in simultaneous_prefix(queue, route_ids):
                U_temp[n] = resource
        choices.append(tuple(U_temp))
    return tuple(choices) if choices else (tuple(None for _ in plans),)


def _dynamic_reset_interrupted_repeat_tasks(
    c: RelaxedNode,
    plans: Sequence[RelaxedVehiclePlan],
    U_c: Tuple[Optional[int], ...],
    U_temp: Tuple[Optional[int], ...],
    ra: Sequence[float],
    ni2: Sequence[int],
) -> Tuple[float, ...]:
    out = list(ra)
    for n, resource in enumerate(U_c):
        if (
            resource is not None
            and c.r[n] > EPS
            and c.U_temp[n] == resource
            and U_temp[n] != resource
            and ni2[n] >= 1
        ):
            _resource, duration = _dynamic_task_info(
                plans[n],
                c.route_candidates[n],
                ni2[n] - 1,
            )
            out[n] = duration
    return tuple(out)


def NextSigM(
    tw: float,
    da: Sequence[float],
    ra: Sequence[float],
    oa: Sequence[float],
    U_temp: Tuple[Optional[int], ...],
) -> Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...], float]:
    running = [n for n, resource in enumerate(U_temp) if resource is not None]
    if running:
        pos_d = [x for x in da if x > EPS and x < BIG_M]
        pos_r = [ra[n] for n in running if ra[n] > EPS and math.isfinite(ra[n])]
        candidates = pos_d + pos_r
    else:
        candidates = [x for x in da if x > EPS and math.isfinite(x)]
    if not candidates:
        return tuple(da), tuple(ra), tuple(oa), tw

    step = min(candidates)
    tw1 = tw + step
    d2: List[float] = []
    r2: List[float] = []
    o2: List[float] = []
    for n in range(len(da)):
        d_next = da[n] - step if math.isfinite(da[n]) else math.inf
        if abs(d_next) <= EPS:
            d_next = 0.0
        d2.append(d_next)
        if U_temp[n] is not None:
            r_next = max(0.0, ra[n] - step)
            o_next = oa[n] + step
        else:
            r_next = ra[n]
            o_next = oa[n] + (step if ra[n] > EPS else 0.0)
        if abs(r_next) <= EPS:
            r_next = 0.0
        if abs(o_next) <= EPS:
            o_next = 0.0
        r2.append(r_next)
        o2.append(o_next)
    return tuple(d2), tuple(r2), tuple(o2), tw1


def NewNode(
    c: RelaxedNode,
    plans: Sequence[RelaxedVehiclePlan],
    d2: Tuple[float, ...],
    r2: Tuple[float, ...],
    o2: Tuple[float, ...],
    tw1: float,
    ni2: Sequence[int],
    U_c: Tuple[Optional[int], ...],
    U_temp: Tuple[Optional[int], ...],
    ra: Sequence[float],
    alpha_in: Sequence[Sequence[float]],
) -> RelaxedNode:
    alpha = [list(row) for row in alpha_in]
    gamma = [list(row) for row in c.gamma]
    d_new = list(d2)
    g_delay = c.g_delay
    segments = list(c.segments)
    attempts = list(c.attempts)
    queue_by_resource = {resource: list(queue) for resource, queue in c.priority_queues}

    for n, resource in enumerate(U_c):
        if (
            resource is not None
            and c.r[n] > EPS
            and c.U_temp[n] == resource
            and U_temp[n] != resource
            and ni2[n] >= 1
        ):
            _task_resource, duration = _dynamic_task_info(
                plans[n],
                c.route_candidates[n],
                ni2[n] - 1,
            )
            progress = max(0.0, duration - c.r[n])
            if progress > EPS:
                attempts.append(
                    AttemptSegment(
                        vehicle_id=plans[n].vehicle_id,
                        task_index=ni2[n],
                        resource=resource,
                        start_time=c.tw - progress,
                        end_time=c.tw,
                    )
                )

    for resource in sorted({item for item in U_c if item is not None}):
        active = [n for n, req in enumerate(U_c) if req == resource and ra[n] > EPS]
        if not active:
            continue
        old_queue = [n for n in queue_by_resource.get(resource, []) if n in active]
        new_items = [n for n in active if n not in old_queue]
        winner = next((n for n, req in enumerate(U_temp) if req == resource), None)
        if winner is None:
            continue
        rest = [n for n in [*old_queue, *new_items] if n != winner]
        queue_by_resource[resource] = [winner, *rest]

    for n, plan in enumerate(plans):
        if ra[n] > EPS and r2[n] <= EPS and ni2[n] >= 1:
            ti = ni2[n] - 1
            resource, duration = _dynamic_task_info(plan, c.route_candidates[n], ti)
            gamma[n][ti] = tw1
            start_time = tw1 - duration
            requested_time = alpha[n][ti]
            delay = max(0.0, start_time - requested_time)
            g_delay += delay
            segments.append(
                ScheduleSegment(
                    vehicle_id=plan.vehicle_id,
                    task_index=ni2[n],
                    resource=resource,
                    requested_time=requested_time,
                    start_time=start_time,
                    end_time=tw1,
                    delay=delay,
                )
            )
            if ni2[n] < _dynamic_task_count(plan, c.route_candidates[n]):
                d_new[n] = plan.road_time
            else:
                d_new[n] = math.inf

    return RelaxedNode(
        idx=-1,
        parent=c.idx,
        tw=tw1,
        g=c.g + (g_delay - c.g_delay),
        g_delay=g_delay,
        g_path=c.g_path,
        segments=tuple(segments),
        route_choices=(),
        attempts=tuple(attempts),
        route_candidates=c.route_candidates,
        U_temp=U_temp,
        ni=tuple(ni2),
        d=tuple(d_new),
        r=r2,
        o=o2,
        alpha=tuple(tuple(row) for row in alpha),
        gamma=tuple(tuple(row) for row in gamma),
        priority_queues=tuple(
            (resource, tuple(queue))
            for resource, queue in sorted(queue_by_resource.items())
            if queue
        ),
        path_decisions=c.path_decisions,
    )


def expand_array_IN(
    nodes: Sequence[RelaxedNode],
    c_idx: int,
    plans: Sequence[RelaxedVehiclePlan],
    *,
    lambda_path: float,
) -> Tuple[List[RelaxedNode], bool]:
    c = nodes[c_idx]
    path_children = _dynamic_path_choice_children(c, plans, lambda_path=lambda_path)
    if path_children:
        merged_children: List[RelaxedNode] = []
        for path_child in path_children:
            temp_node = replace(path_child, idx=c.idx, parent=c.parent)
            next_children, is_leaf = expand_array_IN(
                (temp_node,),
                0,
                plans,
                lambda_path=lambda_path,
            )
            if is_leaf:
                merged_children.append(replace(temp_node, idx=-1, parent=c.idx))
            else:
                merged_children.extend(next_children)
        return merged_children, False

    n_vehicle = len(plans)
    da = [math.inf for _ in plans]
    ra = [0.0 for _ in plans]
    oa = [0.0 for _ in plans]
    ni2 = list(c.ni)
    alpha = [list(row) for row in c.alpha]

    for n, plan in enumerate(plans):
        task_count = _dynamic_task_count(plan, c.route_candidates[n])
        if ni2[n] >= task_count and c.r[n] <= EPS:
            ni2[n] = task_count
            da[n] = math.inf
            ra[n] = 0.0
            oa[n] = 0.0
            continue

        if c.d[n] <= EPS and c.ni[n] < task_count:
            next_task = c.ni[n] + 1
            ti = next_task - 1
            groups = _dynamic_groups_for_task(plan, c.route_candidates[n], ti)
            if len(groups) != 1:
                raise RuntimeError(
                    "dynamic co-design invariant failed: path choice should "
                    "be resolved before task generation"
                )
            _resource, duration = _dynamic_task_info(
                plan,
                c.route_candidates[n],
                ti,
            )
            ni2[n] = next_task
            da[n] = BIG_M
            ra[n] = duration
            oa[n] = 0.0
            if math.isnan(alpha[n][ti]):
                alpha[n][ti] = c.tw
        else:
            da[n] = c.d[n]
            ra[n] = c.r[n]
            oa[n] = c.o[n]

    if all(
        ni2[n] >= _dynamic_task_count(plans[n], c.route_candidates[n]) and ra[n] <= EPS
        for n in range(n_vehicle)
    ):
        return [], True

    U_c = _dynamic_current_requests(plans, c.route_candidates, ra, ni2)
    active = [n for n in range(n_vehicle) if U_c[n] is not None]

    if not active:
        U_temp = tuple(None for _ in plans)
        d2, r2, o2, tw1 = NextSigM(c.tw, da, ra, oa, U_temp)
        child = NewNode(
            c,
            plans,
            d2,
            r2,
            o2,
            tw1,
            ni2,
            U_c,
            U_temp,
            ra,
            alpha,
        )
        return [child], False

    children: List[RelaxedNode] = []
    for U_temp in _dynamic_valid_running_choices(c, U_c, ra, ni2, plans):
        ra_temp = _dynamic_reset_interrupted_repeat_tasks(
            c,
            plans,
            U_c,
            U_temp,
            ra,
            ni2,
        )
        d2, r2, o2, tw1 = NextSigM(c.tw, da, ra_temp, oa, U_temp)
        child = NewNode(
            c,
            plans,
            d2,
            r2,
            o2,
            tw1,
            ni2,
            U_c,
            U_temp,
            ra_temp,
            alpha,
        )
        children.append(child)

    unique: Dict[Tuple, RelaxedNode] = {}
    for child in children:
        sig = (
            round(child.tw, 9),
            child.ni,
            tuple(round(x, 9) for x in child.d),
            tuple(round(x, 9) for x in child.r),
            child.U_temp,
            child.route_candidates,
            round(child.g, 9),
        )
        unique.setdefault(sig, child)
    return list(unique.values()), False


# Backward-compatible aliases for callers using the earlier Python names.
_dynamic_next_sig_m = NextSigM
_dynamic_make_child = NewNode
_expand_dynamic_codesign_node = expand_array_IN


def search_dynamic_codesign_dfs_bb(
    plans: Sequence[RelaxedVehiclePlan],
    *,
    lambda_path: float = 1.0,
    deadline: Optional[float] = None,
    max_nodes: Optional[int] = None,
    branch_and_bound: bool = True,
    verbose: bool = True,
) -> RelaxedSearchResult:
    """Run the true online path-selection/scheduling co-design DFS.

    Unlike ``search_relaxed_dfs_bb``, this does not enumerate full path
    combinations at the root.  A path-selection branch is generated only when a
    vehicle reaches the next task-generation moment and the current traversal is
    not unique among its still-feasible OD route options.
    """

    root = _dynamic_root(plans)
    nodes: List[RelaxedNode] = [root]
    leaves: List[int] = []
    stack: List[int] = [0]
    best_g = math.inf
    best_idx = -1
    log: List[str] = []
    pruned = 0
    step = 0
    t0 = time.perf_counter()

    while stack:
        if deadline is not None and time.perf_counter() - t0 > deadline:
            log.append("[Dynamic_CoDesign_DFS_BB] deadline hit; returning best-so-far")
            break
        if max_nodes is not None and len(nodes) >= max_nodes:
            log.append("[Dynamic_CoDesign_DFS_BB] max_nodes hit; returning best-so-far")
            break

        c_idx = stack.pop()
        step += 1
        if branch_and_bound and nodes[c_idx].g >= best_g - EPS:
            pruned += 1
            continue

        children, is_leaf = expand_array_IN(
            nodes,
            c_idx,
            plans,
            lambda_path=lambda_path,
        )
        if is_leaf:
            leaves.append(c_idx)
            if nodes[c_idx].g < best_g:
                best_g = nodes[c_idx].g
                best_idx = c_idx
                msg = (
                    f"[Dynamic_CoDesign_DFS_BB] step {step}: "
                    f"new best_g={best_g:.6f} at node {c_idx}"
                )
                log.append(msg)
                if verbose:
                    print(msg)
            continue

        children = sorted(children, key=lambda node: node.g, reverse=True)
        for child in children:
            if branch_and_bound and child.g >= best_g - EPS:
                pruned += 1
                continue
            child = replace(child, idx=len(nodes))
            nodes.append(child)
            stack.append(child.idx)

    summary = (
        f"[Dynamic_CoDesign_DFS_BB] done. expanded={step}, "
        f"pruned={pruned}, nodes={len(nodes)}, leaves={len(leaves)}, "
        f"best_g={best_g:.6f}"
    )
    log.append(summary)
    if verbose:
        print(summary)
    return RelaxedSearchResult(
        nodes=tuple(nodes),
        leaves=tuple(leaves),
        best_idx=best_idx,
        best_g=best_g,
        log=tuple(log),
    )


def _collect_dynamic_frontier(
    plans: Sequence[RelaxedVehiclePlan],
    *,
    lambda_path: float,
    frontier_depth: int,
) -> Tuple[List[RelaxedNode], List[int], List[int]]:
    root = _dynamic_root(plans)
    nodes: List[RelaxedNode] = [root]
    frontier: List[int] = [0]
    leaves: List[int] = []

    for _level in range(max(0, frontier_depth)):
        next_frontier: List[int] = []
        for c_idx in frontier:
            children, is_leaf = expand_array_IN(
                nodes,
                c_idx,
                plans,
                lambda_path=lambda_path,
            )
            if is_leaf:
                leaves.append(c_idx)
                continue
            for child in children:
                child = replace(child, idx=len(nodes))
                nodes.append(child)
                next_frontier.append(child.idx)
        if not next_frontier:
            return nodes, [], leaves
        frontier = next_frontier

    return nodes, frontier, leaves


def _search_dynamic_subtree_worker(args):
    (
        order,
        frontier_node,
        plans,
        lambda_path,
        branch_and_bound,
        deadline,
        max_nodes,
    ) = args

    root = replace(frontier_node, idx=0, parent=-1)
    nodes: List[RelaxedNode] = [root]
    leaves: List[int] = []
    stack: List[int] = [0]
    best_g = math.inf
    best_idx = -1
    pruned = 0
    expanded = 0
    t0 = time.perf_counter()

    while stack:
        if deadline is not None and time.perf_counter() - t0 > deadline:
            break
        if max_nodes is not None and len(nodes) >= max_nodes:
            break

        c_idx = stack.pop()
        expanded += 1
        if branch_and_bound and nodes[c_idx].g >= best_g - EPS:
            pruned += 1
            continue

        children, is_leaf = expand_array_IN(
            nodes,
            c_idx,
            plans,
            lambda_path=lambda_path,
        )
        if is_leaf:
            leaves.append(c_idx)
            if nodes[c_idx].g < best_g:
                best_g = nodes[c_idx].g
                best_idx = c_idx
            continue

        children = sorted(children, key=lambda node: node.g, reverse=True)
        for child in children:
            if branch_and_bound and child.g >= best_g - EPS:
                pruned += 1
                continue
            child = replace(child, idx=len(nodes))
            nodes.append(child)
            stack.append(child.idx)

    return order, tuple(nodes), tuple(leaves), best_idx, best_g, expanded, pruned


def search_dynamic_codesign_parallel_dfs_bb(
    plans: Sequence[RelaxedVehiclePlan],
    *,
    lambda_path: float = 1.0,
    frontier_depth: int = 2,
    max_workers: int = 4,
    deadline: Optional[float] = None,
    max_nodes: Optional[int] = None,
    branch_and_bound: bool = True,
    verbose: bool = True,
) -> RelaxedSearchResult:
    """Parallel dynamic co-design DFS using frontier subtrees.

    The root is expanded for ``frontier_depth`` levels in the main process.
    Each frontier node is then solved by a worker with the same dynamic
    path-selection/scheduling expansion logic.  If ``branch_and_bound`` is
    false, all branches in all subtrees are kept for visualization.
    """

    t0 = time.perf_counter()
    prefix_nodes, frontier, prefix_leaves = _collect_dynamic_frontier(
        plans,
        lambda_path=lambda_path,
        frontier_depth=frontier_depth,
    )

    if not frontier:
        best_idx = min(prefix_leaves, key=lambda idx: prefix_nodes[idx].g, default=-1)
        best_g = prefix_nodes[best_idx].g if best_idx >= 0 else math.inf
        return RelaxedSearchResult(
            nodes=tuple(prefix_nodes),
            leaves=tuple(prefix_leaves),
            best_idx=best_idx,
            best_g=best_g,
            log=(
                "[Dynamic_Parallel_DFS_BB] frontier exhausted during prefix expansion",
            ),
        )

    workers = max(1, min(int(max_workers), len(frontier)))
    if workers <= 1:
        return search_dynamic_codesign_dfs_bb(
            plans,
            lambda_path=lambda_path,
            deadline=deadline,
            max_nodes=max_nodes,
            branch_and_bound=branch_and_bound,
            verbose=verbose,
        )

    tasks = [
        (
            order,
            prefix_nodes[frontier_idx],
            tuple(plans),
            lambda_path,
            branch_and_bound,
            deadline,
            max_nodes,
        )
        for order, frontier_idx in enumerate(frontier)
    ]

    with ProcessPoolExecutor(max_workers=workers) as pool:
        solved = list(pool.map(_search_dynamic_subtree_worker, tasks))
    solved.sort(key=lambda item: item[0])

    nodes: List[RelaxedNode] = list(prefix_nodes)
    leaves: List[int] = list(prefix_leaves)
    best_g = math.inf
    best_idx = -1
    expanded = 0
    pruned = 0

    for result_item, frontier_idx in zip(solved, frontier):
        _order, local_nodes, local_leaves, local_best_idx, _local_best_g, local_expanded, local_pruned = result_item
        expanded += local_expanded
        pruned += local_pruned
        index_map = {0: frontier_idx}

        for local_node in local_nodes[1:]:
            mapped_parent = index_map[local_node.parent]
            mapped_idx = len(nodes)
            index_map[local_node.idx] = mapped_idx
            nodes.append(
                replace(
                    local_node,
                    idx=mapped_idx,
                    parent=mapped_parent,
                )
            )

        for local_leaf in local_leaves:
            mapped_leaf = index_map[local_leaf]
            leaves.append(mapped_leaf)
            if nodes[mapped_leaf].g < best_g:
                best_g = nodes[mapped_leaf].g
                best_idx = mapped_leaf

        if local_best_idx >= 0:
            mapped_best = index_map[local_best_idx]
            if nodes[mapped_best].g < best_g:
                best_g = nodes[mapped_best].g
                best_idx = mapped_best

    summary = (
        f"[Dynamic_Parallel_DFS_BB] done. workers={workers}, "
        f"frontier_depth={frontier_depth}, frontier={len(frontier)}, "
        f"expanded={expanded}, pruned={pruned}, nodes={len(nodes)}, "
        f"leaves={len(leaves)}, elapsed={time.perf_counter() - t0:.3f}s, "
        f"best_g={best_g:.6f}"
    )
    if verbose:
        print(summary)
    return RelaxedSearchResult(
        nodes=tuple(nodes),
        leaves=tuple(leaves),
        best_idx=best_idx,
        best_g=best_g,
        log=(summary,),
    )


def extract_path(nodes: Sequence, leaf_idx: int) -> Tuple[int, ...]:  # type: ignore[no-redef]
    out: List[int] = []
    cur = leaf_idx
    while cur >= 0:
        out.append(cur)
        cur = nodes[cur].parent
    return tuple(reversed(out))


def _ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def write_decision_tree_svg(
    result,
    path: str | Path,
    *,
    plans: Sequence = (),
) -> Path:
    p = _ensure_parent(path)
    width = max(360, 70 * max(1, len(result.nodes)))
    height = 160
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for i, node in enumerate(result.nodes):
        x = 30 + i * 65
        y = 80
        if getattr(node, "parent", -1) >= 0:
            px = 30 + node.parent * 65
            parts.append(f'<line x1="{px}" y1="{y}" x2="{x}" y2="{y}" stroke="#94a3b8"/>')
        color = "#16a34a" if i == result.best_idx else "#f8fafc"
        parts.append(f'<circle cx="{x}" cy="{y}" r="14" fill="{color}" stroke="#64748b"/>')
        parts.append(f'<text x="{x}" y="{y+4}" text-anchor="middle" font-family="Arial" font-size="10">{i}</text>')
    parts.append("</svg>")
    p.write_text("\n".join(parts), encoding="utf-8")
    return p


def write_resource_schedule_svgs(
    result,
    out_dir: str | Path,
    *,
    plans: Sequence,
) -> List[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    resources = sorted({seg.resource for seg in result.best_schedule})
    paths: List[Path] = []
    for resource in resources:
        p = out / f"coarse_schedule_I{resource}.svg"
        segs = [seg for seg in result.best_schedule if seg.resource == resource]
        width, height = 520, 90 + 30 * len(segs)
        max_t = max((seg.end_time for seg in segs), default=1.0)
        scale = (width - 80) / max(max_t, 1.0)
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="20" y="24" font-family="Arial" font-size="16" font-weight="700">I{resource}</text>',
        ]
        for row, seg in enumerate(segs):
            y = 45 + row * 28
            x = 60 + seg.start_time * scale
            w = max(2, (seg.end_time - seg.start_time) * scale)
            parts.append(f'<text x="20" y="{y+13}" font-family="Arial" font-size="11">N{seg.vehicle_id}</text>')
            parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="18" fill="#4ade80"/>')
            parts.append(f'<text x="{x+w/2}" y="{y+13}" text-anchor="middle" font-family="Arial" font-size="10">K{seg.task_index}</text>')
        parts.append("</svg>")
        p.write_text("\n".join(parts), encoding="utf-8")
        paths.append(p)
    return paths


def write_resource_schedule_panel_html(
    schedule_paths: Iterable[str | Path],
    path: str | Path,
    *,
    tmap: Optional[TrafficMap] = None,
) -> Path:
    p = _ensure_parent(path)
    imgs = "\n".join(f'<img src="{Path(item).name}" style="max-width:100%;border:1px solid #ddd">' for item in schedule_paths)
    p.write_text(f"<!doctype html><html><body>{imgs}</body></html>", encoding="utf-8")
    return p


def write_interactive_solution_html(
    result,
    path: str | Path,
    *,
    plans: Sequence,
    tmap: Optional[TrafficMap] = None,
    max_terminal_paths: Optional[int] = None,
    max_tree_nodes: Optional[int] = None,
    lambda_path: float = 1.0,
) -> Path:
    p = _ensure_parent(path)
    all_terminals = list(result.leaves) if result.leaves else [result.best_idx]
    all_terminals = [idx for idx in all_terminals if idx >= 0]
    node_by_idx = {node.idx: node for node in result.nodes}

    def terminal_path(terminal: int) -> Tuple[int, ...]:
        chain = []
        cur = terminal
        while cur >= 0 and cur in node_by_idx:
            chain.append(cur)
            cur = node_by_idx[cur].parent
        return tuple(reversed(chain))

    terminal_paths = {terminal: terminal_path(terminal) for terminal in all_terminals}
    selected_terminals = list(all_terminals)
    included = set(node_by_idx)
    if max_terminal_paths is not None or max_tree_nodes is not None:
        path_limit = (
            len(all_terminals)
            if max_terminal_paths is None
            else max(1, int(max_terminal_paths))
        )
        node_limit = (
            len(node_by_idx)
            if max_tree_nodes is None
            else max(1, int(max_tree_nodes))
        )
        best = result.best_idx
        score_key = lambda idx: (getattr(node_by_idx[idx], "g", math.inf), idx)

        # Preserve branch diversity: best path first, then the best terminal
        # under each root child, then the remaining terminals by objective.
        representatives: Dict[int, int] = {}
        for terminal, chain in terminal_paths.items():
            branch = chain[1] if len(chain) > 1 else chain[0]
            current = representatives.get(branch)
            if current is None or score_key(terminal) < score_key(current):
                representatives[branch] = terminal
        ranked = []
        for terminal in [
            best,
            *sorted(representatives.values(), key=score_key),
            *sorted(all_terminals, key=score_key),
        ]:
            if terminal in terminal_paths and terminal not in ranked:
                ranked.append(terminal)

        selected_terminals = []
        included = set()
        for terminal in ranked:
            if len(selected_terminals) >= path_limit:
                break
            candidate_nodes = set(terminal_paths[terminal])
            expanded = included | candidate_nodes
            if selected_terminals and len(expanded) > node_limit:
                continue
            selected_terminals.append(terminal)
            included = expanded
        if not selected_terminals and ranked:
            selected_terminals = [ranked[0]]
            included = set(terminal_paths[ranked[0]])

    children_by_parent: Dict[int, List[int]] = {}
    for node in result.nodes:
        children_by_parent.setdefault(node.parent, []).append(node.idx)
    terminal_set = set(all_terminals)
    subtree_nodes: Dict[int, int] = {}
    subtree_terminals: Dict[int, int] = {}
    for node in reversed(result.nodes):
        children = children_by_parent.get(node.idx, [])
        subtree_nodes[node.idx] = 1 + sum(subtree_nodes.get(child, 0) for child in children)
        subtree_terminals[node.idx] = (
            (1 if node.idx in terminal_set else 0)
            + sum(subtree_terminals.get(child, 0) for child in children)
        )

    omitted_branches = []
    for parent in sorted(included):
        hidden_children = [
            child
            for child in children_by_parent.get(parent, [])
            if child not in included
        ]
        if not hidden_children:
            continue
        omitted_branches.append(
            {
                "parent": parent,
                "path_count": sum(subtree_terminals.get(child, 0) for child in hidden_children),
                "node_count": sum(subtree_nodes.get(child, 0) for child in hidden_children),
            }
        )

    nodes = []
    for node in result.nodes:
        if node.idx not in included:
            continue
        nodes.append(
            {
                "idx": node.idx,
                "parent": node.parent,
                "tw": node.tw,
                "g": node.g,
                "g_delay": getattr(node, "g_delay", node.g),
                "g_path": getattr(node, "g_path", 0.0),
                "U_temp": list(getattr(node, "U_temp", ())),
                "segments": [
                    {
                        "vehicle_id": seg.vehicle_id,
                        "task_index": seg.task_index,
                        "resource": seg.resource,
                        "requested_time": seg.requested_time,
                        "start_time": seg.start_time,
                        "end_time": seg.end_time,
                        "delay": seg.delay,
                    }
                    for seg in node.segments
                ],
                "attempts": [
                    {
                        "vehicle_id": attempt.vehicle_id,
                        "task_index": attempt.task_index,
                        "resource": attempt.resource,
                        "start_time": attempt.start_time,
                        "end_time": attempt.end_time,
                    }
                    for attempt in getattr(node, "attempts", ())
                ],
                "route_choices": list(getattr(node, "route_choices", ())),
                "route_candidates": [
                    list(items) for items in getattr(node, "route_candidates", ())
                ],
                "path_decisions": [
                    {
                        "vehicle": item[0],
                        "option": item[1],
                        "from": f"I{item[2]}",
                        "to": f"I{item[3]}",
                        "tw": item[4],
                        "extra": item[5] if len(item) > 5 else None,
                    }
                    for item in getattr(node, "path_decisions", ())
                ],
            }
        )

    plan_data = []
    path_trees = []
    for i, plan in enumerate(plans):
        if hasattr(plan, "route_options"):
            options = []
            base = min(_free_time(opt, plan.road_time) for opt in plan.route_options)
            shortest_index = min(
                range(len(plan.route_options)),
                key=lambda j: (
                    len(plan.route_options[j].intersections),
                    _free_time(plan.route_options[j], plan.road_time),
                    plan.route_options[j].id,
                ),
            )
            for j, option in enumerate(plan.route_options):
                options.append(
                    {
                        "candidate_index": j,
                        "intersections": list(option.intersections),
                        "execution_times": list(option.execution_times),
                        "movements": _route_option_movements(
                            plan.entrance, plan.exit, option
                        ),
                        "free_time": _free_time(option, plan.road_time),
                        "extra_time": max(0.0, _free_time(option, plan.road_time) - base),
                    }
                )
            path_trees.append(
                {
                    "plan_index": i,
                    "vehicle_id": plan.vehicle_id,
                    "entrance": plan.entrance,
                    "exit": plan.exit,
                    "shortest_index": shortest_index,
                    "options": options,
                }
            )
            plan_data.append(
                {
                    "vehicle_id": plan.vehicle_id,
                    "entrance": plan.entrance,
                    "exit": plan.exit,
                    "turns_by_option": [
                        [trav.turn for trav in option.traversals]
                        for option in plan.route_options
                    ],
                    "route_ids_by_option": [
                        [trav.route_id for trav in option.traversals]
                        for option in plan.route_options
                    ],
                    "resources_by_option": [
                        list(option.intersections) for option in plan.route_options
                    ],
                }
            )
        else:
            if tmap is not None:
                route_options = tmap.route_options(plan.entrance, plan.exit)
                base = min(_free_time(opt, plan.road_time) for opt in route_options)
                selected_index = 0
                shortest_index = min(
                    range(len(route_options)),
                    key=lambda j: (
                        len(route_options[j].intersections),
                        _free_time(route_options[j], plan.road_time),
                        route_options[j].id,
                    ),
                )
                options = []
                for j, option in enumerate(route_options):
                    if option.intersections == plan.route.intersections:
                        selected_index = j
                    options.append(
                        {
                            "candidate_index": j,
                            "intersections": list(option.intersections),
                            "execution_times": list(option.execution_times),
                            "movements": _route_option_movements(
                                plan.entrance, plan.exit, option
                            ),
                            "free_time": _free_time(option, plan.road_time),
                            "extra_time": max(0.0, _free_time(option, plan.road_time) - base),
                        }
                    )
                path_trees.append(
                    {
                        "plan_index": i,
                        "vehicle_id": plan.vehicle_id,
                        "entrance": plan.entrance,
                        "exit": plan.exit,
                        "selected_index": selected_index,
                        "shortest_index": shortest_index,
                        "options": options,
                    }
                )
            plan_data.append(
                {
                    "vehicle_id": plan.vehicle_id,
                    "entrance": plan.entrance,
                    "exit": plan.exit,
                    "turns": [trav.turn for trav in plan.route.traversals],
                    "route_ids": [trav.route_id for trav in plan.route.traversals],
                    "resources": list(plan.resources),
                }
            )

    coords = {str(k): list(v) for k, v in (tmap.coords.items() if tmap else [])}
    ports = (
        [
            {
                "id": port.id,
                "intersection": port.intersection,
                "direction": port.direction,
            }
            for port in sorted(tmap.ports.values(), key=lambda item: item.id)
        ]
        if tmap
        else []
    )
    roads = (
        [
            {
                "id": road.id,
                "a": road.endpoints[0],
                "b": road.endpoints[1],
            }
            for road in sorted(tmap.roads.values(), key=lambda item: item.id)
        ]
        if tmap
        else []
    )
    resources = sorted(
        {
            seg.resource
            for node in result.nodes
            for seg in node.segments
        }
        or set(coords.keys())
    )
    data = {
        "nodes": nodes,
        "terminals": selected_terminals,
        "terminal_count_total": len(all_terminals),
        "terminal_count_shown": len(selected_terminals),
        "node_count_total": len(result.nodes),
        "node_count_shown": len(nodes),
        "omitted_branches": omitted_branches,
        "max_terminal_paths": max_terminal_paths,
        "max_tree_nodes": max_tree_nodes,
        "best_idx": result.best_idx,
        "plans": plan_data,
        "path_trees": path_trees,
        "coords": coords,
        "ports": ports,
        "roads": roads,
        "resources": resources,
        "lambda_path": lambda_path,
        "trajectory_conflict_filter": trajectory_conflict_filter_enabled(),
        "conflicting_route_pairs": [
            [left, right]
            for left in range(1, 13)
            for right in range(left, 13)
            if route_ids_conflict(left, right)
        ],
    }

    data_json = json.dumps(data, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Interactive Coarse Solution</title>
  <style>
    body {{ margin:0; font-family:Arial,sans-serif; background:#f8fafc; color:#111827; }}
    header {{ display:flex; justify-content:space-between; gap:12px; align-items:center; padding:10px 14px; background:white; border-bottom:1px solid #d1d5db; }}
    h1 {{ margin:0; font-size:18px; }}
    h2 {{ margin:10px 12px; font-size:15px; }}
    h3 {{ margin:0 0 6px; }}
    #meta {{ font-size:14px; font-weight:700; white-space:nowrap; }}
    .hint {{ color:#64748b; font-size:12px; margin-left:14px; }}
    .notation {{ color:#475569; font-size:12px; margin:-4px 12px 8px; }}
    .layout {{ --left-pane:75%; display:grid; grid-template-columns:minmax(360px,var(--left-pane)) 8px minmax(300px,1fr); gap:8px; padding:12px; height:calc(100vh - 48px); box-sizing:border-box; }}
    .pane {{ background:white; border:1px solid #d1d5db; border-radius:6px; overflow:auto; }}
    .splitter {{ cursor:col-resize; border-radius:6px; background:linear-gradient(90deg, transparent 0 2px, #cbd5e1 2px 6px, transparent 6px 8px); }}
    .splitter:hover, .splitter.dragging {{ background:linear-gradient(90deg, transparent 0 2px, #64748b 2px 6px, transparent 6px 8px); }}
    .tree-header {{ display:flex; align-items:center; justify-content:space-between; gap:8px; margin:10px 12px 4px; }}
    .tree-header h2 {{ margin:0; }}
    .tree-toolbar {{ display:flex; align-items:center; flex-wrap:wrap; gap:4px; }}
    .tree-toolbar button {{ border:1px solid #cbd5e1; border-radius:4px; background:#fff; color:#334155; padding:3px 7px; font-size:11px; font-weight:700; cursor:pointer; }}
    .tree-toolbar button:hover, .tree-toolbar button.active {{ border-color:#2563eb; background:#eff6ff; color:#1d4ed8; }}
    .tree-summary {{ margin:0 12px 7px; padding:5px 8px; border:1px solid #fde68a; border-radius:5px; background:#fffbeb; color:#92400e; font-size:11px; font-weight:700; }}
    .tree-stage {{ position:relative; height:60vh; min-height:420px; max-height:720px; margin:0 10px 12px; border:1px solid #dbe3ef; border-radius:6px; overflow:hidden; background:#fff; }}
    #treeSvg {{ width:100%; height:100%; display:block; cursor:default; user-select:none; touch-action:none; }}
    #treeSvg.zoomed {{ cursor:grab; }}
    #treeSvg.dragging {{ cursor:grabbing; }}
    .tree-magnifier {{ position:absolute; right:10px; top:10px; width:280px; height:190px; border:2px solid #2563eb; border-radius:7px; background:rgba(255,255,255,0.97); box-shadow:0 4px 18px rgba(15,23,42,0.18); overflow:hidden; pointer-events:none; display:none; }}
    .tree-magnifier.visible {{ display:block; }}
    .tree-magnifier-label {{ position:absolute; left:5px; top:4px; z-index:2; max-width:265px; padding:2px 5px; border-radius:3px; background:rgba(255,255,255,0.9); color:#1e3a8a; font-size:10px; font-weight:700; }}
    #treeLensSvg {{ width:100%; height:100%; display:block; }}
    #basicMapPanel {{ padding:0 10px 12px; display:grid; justify-items:start; }}
    #basicMapPanel svg {{ display:block; max-width:100%; height:auto; }}
    #pathSummaryPanel {{ padding:0 10px 10px; }}
    .path-table {{ border-collapse:collapse; font-size:11px; min-width:720px; max-width:100%; margin-bottom:8px; }}
    .path-table th, .path-table td {{ border:1px solid #d1d5db; padding:4px 6px; text-align:left; vertical-align:top; }}
    .path-table th {{ background:#f1f5f9; font-weight:700; white-space:nowrap; }}
    .path-table td:first-child, .path-table td:nth-child(2), .path-table td:nth-child(3) {{ white-space:nowrap; font-weight:700; }}
    .path-list {{ display:flex; flex-wrap:wrap; gap:3px; }}
    .path-pill {{ border:1px solid #cbd5e1; border-radius:4px; padding:2px 5px; background:#f8fafc; white-space:nowrap; display:inline-flex; gap:5px; align-items:baseline; }}
    .path-pill.selected {{ border-color:#16a34a; background:#dcfce7; color:#166534; font-weight:700; }}
    .path-cost {{ color:#475569; font-family:Consolas, monospace; font-size:10px; font-weight:600; }}
    .path-pill.selected .path-cost {{ color:#166534; }}
    .path-note {{ color:#475569; font-size:11px; margin:0 0 6px; }}
    #pathTreePanel {{ padding:0 10px 12px; display:grid; grid-template-columns:1fr; gap:8px; justify-items:start; }}
    .path-card {{ border:1px solid #d1d5db; border-radius:6px; padding:6px 8px; background:white; width:max-content; max-width:calc(100% - 18px); min-width:0; overflow:auto; }}
    .path-card h3 {{ font-size:10px; line-height:1.2; margin:0 0 4px; white-space:nowrap; }}
    .path-card svg {{ display:block; margin:0; max-width:100%; height:auto; }}
    .path-card-visuals {{ display:grid; grid-template-columns:auto auto; gap:12px; align-items:start; }}
    .path-visual-title {{ margin:0 0 3px; font-size:9px; font-weight:700; color:#475569; }}
    #schedulePanel {{ padding:10px; display:grid; grid-template-columns:1fr; gap:10px; }}
    .icard {{ border:1px solid #d1d5db; border-radius:6px; padding:7px 9px; background:white; }}
    .icard h3 {{ font-size:14px; }}
    .empty {{ min-height:90px; display:grid; place-items:center; color:#64748b; border:1px dashed #cbd5e1; }}
    @media (max-width:1000px) {{ .layout {{ grid-template-columns:1fr; height:auto; }} .splitter {{ display:none; }} #schedulePanel {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header>
    <div><h1>Interactive Coarse Solution</h1><div class="hint">Use arrow keys to switch displayed paths. Use the tree toolbar or Ctrl+wheel to zoom; drag only while zoomed.</div></div>
    <div id="meta"></div>
  </header>
  <main class="layout" id="mainLayout">
    <section class="pane">
      <div class="tree-header">
        <h2>Decision Tree</h2>
        <div class="tree-toolbar">
          <button id="treeZoomOut" type="button" title="Zoom out around the viewport center">−</button>
          <button id="treeZoomIn" type="button" title="Zoom in around the viewport center">+</button>
          <button id="treeFitAll" type="button">Fit tree</button>
          <button id="treeFitPath" type="button">Fit selected path</button>
          <button id="treeMagnifierToggle" type="button" class="active">Magnifier</button>
        </div>
      </div>
      <div class="notation">Notation: z<sub>nq</sub>(i,j)(tw) means vehicle Nn selects path option q from Ii to Ij at time tw; q is displayed from 1, while decision-tree node IDs start from 0. v<sub>in</sub>(tw) means vehicle Nn occupies Ii at time tw.</div>
      <div class="tree-summary" id="treeSummary"></div>
      <div class="tree-stage" id="treeStage">
        <svg id="treeSvg"></svg>
        <div class="tree-magnifier" id="treeMagnifier">
          <div class="tree-magnifier-label" id="treeMagnifierLabel"></div>
          <svg id="treeLensSvg"><use href="#treeScene"></use></svg>
        </div>
      </div>
      <h2>Vehicle Path Branch Trees</h2>
      <div id="pathTreePanel"></div>
      <h2>Vehicle Path Options</h2>
      <div id="pathSummaryPanel"></div>
      <h2>Path Selection Branches</h2>
      <div id="pathBranchPanel"></div>
      <h2>Basic Map</h2>
      <div id="basicMapPanel"></div>
    </section>
    <div class="splitter" id="splitter" title="Drag to resize panels; double-click to reset"></div>
    <section class="pane">
      <h2 id="scheduleTitle">Schedule</h2>
      <div id="schedulePanel"></div>
    </section>
  </main>
  <script>
    const DATA = {data_json};
    const NS = "http://www.w3.org/2000/svg";
    const nodeById = new Map(DATA.nodes.map(n => [n.idx, n]));
    const terminals = DATA.terminals.length ? DATA.terminals : [DATA.best_idx];
    let cursor = Math.max(0, terminals.indexOf(DATA.best_idx));
    let treeBaseViewBox = null;
    let treeViewBox = null;
    let treeSelectedBounds = null;
    let treeDrag = null;
    let splitterDrag = null;
    let treeMagnifierEnabled = true;

    function el(name, attrs={{}}, text=null) {{
      const e = document.createElementNS(NS, name);
      for (const [k,v] of Object.entries(attrs)) e.setAttribute(k, v);
      if (text !== null) e.textContent = text;
      return e;
    }}

    function pathTo(idx) {{
      const out = [];
      let cur = idx;
      while (cur >= 0 && nodeById.has(cur)) {{
        out.push(cur);
        cur = nodeById.get(cur).parent;
      }}
      return out.reverse();
    }}

    function sameTreeBox(a, b) {{
      return a && b && a.length === b.length && a.every((item, index) => Math.abs(item - b[index]) < 1e-9);
    }}

    function applyTreeViewBox() {{
      const svg = document.getElementById("treeSvg");
      if (treeViewBox) svg.setAttribute("viewBox", treeViewBox.join(" "));
      const zoomed = treeBaseViewBox && treeViewBox && treeViewBox[2] < treeBaseViewBox[2] * 0.995;
      svg.classList.toggle("zoomed", Boolean(zoomed));
    }}

    function resetTreeZoom() {{
      if (!treeBaseViewBox) return;
      treeViewBox = treeBaseViewBox.slice();
      applyTreeViewBox();
    }}

    function zoomTreeAtCenter(factor) {{
      if (!treeViewBox || !treeBaseViewBox) return;
      const minW = treeBaseViewBox[2] * 0.08;
      const maxW = treeBaseViewBox[2];
      const nextW = Math.min(maxW, Math.max(minW, treeViewBox[2] * factor));
      const ratio = nextW / treeViewBox[2];
      const nextH = treeViewBox[3] * ratio;
      const cx = treeViewBox[0] + treeViewBox[2] / 2;
      const cy = treeViewBox[1] + treeViewBox[3] / 2;
      treeViewBox = [cx - nextW / 2, cy - nextH / 2, nextW, nextH];
      applyTreeViewBox();
    }}

    function fitSelectedTreePath() {{
      if (!treeSelectedBounds || !treeBaseViewBox) return;
      const padX = Math.max(35, (treeSelectedBounds.x2 - treeSelectedBounds.x1) * 0.12);
      const padY = Math.max(35, (treeSelectedBounds.y2 - treeSelectedBounds.y1) * 0.25);
      const width = Math.max(90, treeSelectedBounds.x2 - treeSelectedBounds.x1 + 2 * padX);
      const height = Math.max(90, treeSelectedBounds.y2 - treeSelectedBounds.y1 + 2 * padY);
      const stage = document.getElementById("treeStage").getBoundingClientRect();
      const aspect = Math.max(0.2, stage.width / Math.max(1, stage.height));
      let viewW = width, viewH = height;
      if (viewW / viewH < aspect) viewW = viewH * aspect;
      else viewH = viewW / aspect;
      const cx = (treeSelectedBounds.x1 + treeSelectedBounds.x2) / 2;
      const cy = (treeSelectedBounds.y1 + treeSelectedBounds.y2) / 2;
      treeViewBox = [cx - viewW / 2, cy - viewH / 2, viewW, viewH];
      applyTreeViewBox();
    }}

    function treeSvgPoint(event) {{
      const svg = document.getElementById("treeSvg");
      const point = svg.createSVGPoint();
      point.x = event.clientX;
      point.y = event.clientY;
      return point.matrixTransform(svg.getScreenCTM().inverse());
    }}

    function fmtTime(t) {{
      return Number(t || 0).toFixed(2);
    }}

    function segmentKey(s) {{
      return `${{s.vehicle_id}}|${{s.task_index}}|${{s.resource}}|${{s.start_time}}|${{s.end_time}}`;
    }}

    function newSegments(parent, child) {{
      const oldKeys = new Set((parent.segments || []).map(segmentKey));
      return (child.segments || []).filter(s => !oldKeys.has(segmentKey(s)));
    }}

    function optionLabelsForDecision(pathTree, option) {{
      return (option ? option.intersections : []).map(m => `I${{m}}`);
    }}

    function firstBranchEdge(pathTree, option) {{
      if (!pathTree || !option) return null;
      const selectedLabels = optionLabelsForDecision(pathTree, option);
      for (let k = 0; k < selectedLabels.length - 1; k++) {{
        const prefix = selectedLabels.slice(0, k + 1);
        const nextChoices = new Set();
        for (const candidate of pathTree.options || []) {{
          const labels = optionLabelsForDecision(pathTree, candidate);
          const ok = prefix.every((label, index) => labels[index] === label);
          if (ok && labels.length > k + 1) nextChoices.add(labels[k + 1]);
        }}
        if (nextChoices.size > 1) {{
          return {{from: selectedLabels[k], to: selectedLabels[k + 1]}};
        }}
      }}
      if (selectedLabels.length >= 2) {{
        return {{from: selectedLabels[0], to: selectedLabels[1]}};
      }}
      return null;
    }}

    function edgeDecisionLabels(parent, child) {{
      const labels = [];
      const parentZ = new Set((parent.path_decisions || []).map(item => `${{item.vehicle}}|${{item.option}}|${{item.from}}|${{item.to}}|${{Number(item.tw).toFixed(9)}}`));
      for (const decision of (child.path_decisions || [])) {{
        const key = `${{decision.vehicle}}|${{decision.option}}|${{decision.from}}|${{decision.to}}|${{Number(decision.tw).toFixed(9)}}`;
        if (parentZ.has(key)) continue;
        labels.push({{
          type: "z",
          vehicle: decision.vehicle,
          option: decision.option,
          from: decision.from,
          to: decision.to,
          tw: decision.tw,
          extra: decision.extra
        }});
      }}
      if (
        labels.length === 0 &&
        child.route_choices && child.route_choices.length &&
        (!parent.route_choices || !parent.route_choices.length)
      ) {{
        child.route_choices.forEach((choice, planIndex) => {{
          const pathTree = DATA.path_trees.find(item => item.plan_index === planIndex);
          if (!pathTree || !pathTree.options || pathTree.options.length <= 1) return;
          const option = pathTree.options[choice] || pathTree.options[0];
          const edge = firstBranchEdge(pathTree, option);
          labels.push({{
            type: "z",
            vehicle: pathTree.vehicle_id,
            option: Number(choice) + 1,
            from: edge ? edge.from : "",
            to: edge ? edge.to : "",
            tw: parent.tw
          }});
        }});
      }}
      for (const seg of newSegments(parent, child)) {{
        labels.push({{
          type: "v",
          resource: seg.resource,
          vehicle: seg.vehicle_id,
          tw: parent.tw
        }});
      }}
      return labels;
    }}

    function compactNode(value) {{
      return String(value || "").replace(/^I/, "");
    }}

    function compactLabelText(label) {{
      if (label.type === "more") return `+${{label.count}} more`;
      if (label.type === "v") return `v${{label.resource}}${{label.vehicle}}(${{fmtTime(label.tw)}})`;
      if (label.type === "z") {{
        const extra = Number.isFinite(Number(label.extra)) ? `,+${{Number(label.extra).toFixed(2)}}` : "";
        return `z${{label.vehicle}}${{label.option}}(${{compactNode(label.from)}},${{compactNode(label.to)}})(${{fmtTime(label.tw)}}${{extra}})`;
      }}
      return String(label);
    }}

    function estimatedEdgeLabelWidth(labels) {{
      if (!labels.length) return 0;
      const shown = labels.slice(0, 3);
      if (labels.length > shown.length) shown.push({{type:"more", count:labels.length - shown.length}});
      return Math.max(...shown.map(label => compactLabelText(label).length)) * 4.4;
    }}

    function appendMathPieces(text, label) {{
      const baseSize = 6.5;
      const scriptSize = 5;
      if (label.type === "v") {{
        text.appendChild(el("tspan", {{}}, `v${{label.resource}}${{label.vehicle}}(${{fmtTime(label.tw)}})`));
        return;
      }} else if (label.type === "z") {{
        text.appendChild(el("tspan", {{}}, compactLabelText(label)));
        return;
      }} else {{
        text.appendChild(el("tspan", {{}}, String(label)));
        return;
      }}
      text.appendChild(el("tspan", {{"baseline-shift":"baseline", "font-size":baseSize}}, " (t"));
      text.appendChild(el("tspan", {{"baseline-shift":"sub", "font-size":scriptSize}}, "w"));
      text.appendChild(el("tspan", {{"baseline-shift":"baseline", "font-size":baseSize}}, `=${{fmtTime(label.tw)}})=1`));
    }}

    function boxesOverlap(a, b) {{
      return !(a.x2 < b.x1 || b.x2 < a.x1 || a.y2 < b.y1 || b.y2 < a.y1);
    }}

    function labelBox(cx, cy, shown) {{
      const width = Math.max(24, estimatedEdgeLabelWidth(shown)) + 8;
      const height = shown.length * 9 + 8;
      return {{
        x1: cx - width / 2,
        x2: cx + width / 2,
        y1: cy - height / 2,
        y2: cy + height / 2
      }};
    }}

    function drawEdgeLabel(svg, labels, x1, y1, x2, y2, strong, occupiedLabels) {{
      if (!labels.length) return;
      const dx = x2 - x1;
      const dy = y2 - y1;
      const len = Math.max(1, Math.hypot(dx, dy));
      const nx = -dy / len;
      const ny = dx / len;
      const side = dy >= 0 ? -1 : 1;
      const shown = labels.slice(0, 3);
      if (labels.length > shown.length) shown.push({{type:"more", count:labels.length - shown.length}});
      const baseOffset = Math.abs(dy) < 1e-6 ? -10 : 15 * side;
      const offsetOptions = [baseOffset, baseOffset + 16 * side, baseOffset - 16 * side, baseOffset + 32 * side, baseOffset - 32 * side];
      const alongOptions = [0, 14, -14, 28, -28];
      let lx = (x1 + x2) / 2 + nx * baseOffset;
      let ly = (y1 + y2) / 2 + ny * baseOffset;
      let chosenBox = labelBox(lx, ly, shown);
      let found = false;
      for (const off of offsetOptions) {{
        for (const along of alongOptions) {{
          const cx = (x1 + x2) / 2 + nx * off + (dx / len) * along;
          const cy = (y1 + y2) / 2 + ny * off + (dy / len) * along;
          const box = labelBox(cx, cy, shown);
          if (!occupiedLabels.some(existing => boxesOverlap(existing, box))) {{
            lx = cx;
            ly = cy;
            chosenBox = box;
            found = true;
            break;
          }}
        }}
        if (found) break;
      }}
      occupiedLabels.push(chosenBox);
      shown.forEach((label, index) => {{
        const text = el("text", {{
          x:lx,
          y:ly - (shown.length - 1) * 4 + index * 9,
          "text-anchor":"middle",
          "font-family":"Arial",
          "font-size":6.5,
          "font-weight":strong ? 700 : 500,
          fill:strong ? "#111827" : "#475569",
          stroke:"#ffffff",
          "stroke-width":2.5,
          "paint-order":"stroke fill",
          "stroke-linejoin":"round"
        }});
        if (label.type === "more") {{
          text.appendChild(el("tspan", {{}}, `+${{label.count}} more`));
        }} else {{
          appendMathPieces(text, label);
        }}
        svg.appendChild(text);
      }});
    }}

    function updateTreeMagnifier(x, y, label) {{
      if (!treeMagnifierEnabled) return;
      const lens = document.getElementById("treeMagnifier");
      const lensSvg = document.getElementById("treeLensSvg");
      lensSvg.setAttribute("viewBox", `${{x-95}} ${{y-65}} 190 130`);
      document.getElementById("treeMagnifierLabel").textContent = label;
      lens.classList.add("visible");
    }}

    function renderTree() {{
      const selected = terminals[cursor];
      const selectedPath = new Set(pathTo(selected));
      const bestPath = new Set(pathTo(DATA.best_idx));
      const summaryNodes = (DATA.omitted_branches || []).map((item, index) => ({{
        idx:`omitted-${{index}}`, parent:item.parent, omitted:true,
        path_count:item.path_count, node_count:item.node_count
      }}));
      const displayNodes = [...DATA.nodes, ...summaryNodes];
      const levels = new Map();
      const depths = new Map();
      for (const n of displayNodes) {{
        const d = n.parent < 0 ? 0 : (depths.get(n.parent) || 0) + 1;
        depths.set(n.idx, d);
        if (!levels.has(d)) levels.set(d, []);
        levels.get(d).push(n.idx);
      }}
      const maxDepth = Math.max(...Array.from(levels.keys()), 0);
      const maxRows = Math.max(...Array.from(levels.values()).map(v => v.length), 1);
      const minXGap = 70, maxXGap = 150, yGap = 56, margin = 46;
      const xGaps = Array.from({{length:Math.max(1, maxDepth)}}, () => minXGap);
      for (const n of DATA.nodes) {{
        if (n.parent < 0) continue;
        const parent = nodeById.get(n.parent);
        const parentDepth = depths.get(n.parent) || 0;
        const labelWidth = estimatedEdgeLabelWidth(edgeDecisionLabels(parent, n));
        const needed = Math.min(maxXGap, Math.max(minXGap, labelWidth + 44));
        xGaps[parentDepth] = Math.max(xGaps[parentDepth], needed);
      }}
      const xByDepth = [0];
      for (let d = 0; d < maxDepth; d++) {{
        xByDepth[d + 1] = xByDepth[d] + (xGaps[d] || minXGap);
      }}
      const width = margin * 2 + Math.max(1, xByDepth[maxDepth] || 0);
      const height = margin * 2 + Math.max(1, maxRows - 1) * yGap;
      const pos = new Map();
      for (const [d, ids] of levels.entries()) {{
        const startY = height / 2 - (ids.length - 1) * yGap / 2;
        ids.forEach((id, r) => pos.set(id, [margin + (xByDepth[d] || 0), startY + r * yGap]));
      }}
      const selectedPoints = [...selectedPath].map(id => pos.get(id)).filter(Boolean);
      treeSelectedBounds = selectedPoints.length ? {{
        x1:Math.min(...selectedPoints.map(point => point[0])),
        x2:Math.max(...selectedPoints.map(point => point[0])),
        y1:Math.min(...selectedPoints.map(point => point[1])),
        y2:Math.max(...selectedPoints.map(point => point[1]))
      }} : null;

      const svg = document.getElementById("treeSvg");
      svg.innerHTML = "";
      const scene = el("g", {{id:"treeScene"}});
      svg.appendChild(scene);
      const nextBaseViewBox = [0, 0, width, height];
      if (!sameTreeBox(treeBaseViewBox, nextBaseViewBox)) {{
        treeBaseViewBox = nextBaseViewBox;
        treeViewBox = nextBaseViewBox.slice();
      }}
      applyTreeViewBox();
      const occupiedLabels = [];
      for (const n of displayNodes) {{
        const [x,y] = pos.get(n.idx);
        occupiedLabels.push({{x1:x-38, x2:x+38, y1:y-34, y2:y+22}});
      }}
      for (const n of displayNodes) {{
        if (n.parent < 0) continue;
        const [x1,y1] = pos.get(n.parent);
        const [x2,y2] = pos.get(n.idx);
        if (n.omitted) {{
          scene.appendChild(el("line", {{x1,y1,x2,y2, stroke:"#94a3b8", "stroke-width":1.2, "stroke-dasharray":"4 3"}}));
          continue;
        }}
        const parent = nodeById.get(n.parent);
        const onSelected = selectedPath.has(n.idx) && selectedPath.has(n.parent);
        const onBest = bestPath.has(n.idx) && bestPath.has(n.parent);
        scene.appendChild(el("line", {{
          x1,y1,x2,y2,
          stroke: onSelected ? (selected === DATA.best_idx ? "#16a34a" : "#f59e0b") : (onBest ? "#86efac" : "#cbd5e1"),
          "stroke-width": onSelected ? 3 : 1.2
        }}));
        if (onSelected || onBest || DATA.nodes.length <= 250) {{
          drawEdgeLabel(scene, edgeDecisionLabels(parent, n), x1, y1, x2, y2, onSelected, occupiedLabels);
        }}
      }}
      for (const n of displayNodes) {{
        const [x,y] = pos.get(n.idx);
        if (n.omitted) {{
          const box = el("rect", {{x:x-37, y:y-15, width:74, height:30, rx:6, fill:"#f1f5f9", stroke:"#64748b", "stroke-width":1.2, "stroke-dasharray":"4 2"}});
          box.addEventListener("pointerenter", () => updateTreeMagnifier(x, y, `Omitted: ${{n.path_count}} paths, ${{n.node_count}} nodes`));
          scene.appendChild(box);
          scene.appendChild(el("text", {{x, y:y-2, "text-anchor":"middle", "font-family":"Arial", "font-size":8, "font-weight":700, fill:"#475569"}}, `+${{n.path_count}} paths`));
          scene.appendChild(el("text", {{x, y:y+9, "text-anchor":"middle", "font-family":"Arial", "font-size":7, fill:"#64748b"}}, `${{n.node_count}} nodes omitted`));
          continue;
        }}
        const onSelected = selectedPath.has(n.idx);
        const isBestPath = bestPath.has(n.idx);
        const fill = onSelected ? (selected === DATA.best_idx ? "#dcfce7" : "#fffbeb") : "#f8fafc";
        const stroke = onSelected ? (selected === DATA.best_idx ? "#16a34a" : "#f59e0b") : (isBestPath ? "#86efac" : "#cbd5e1");
        const circle = el("circle", {{cx:x, cy:y, r:13, fill, stroke, "stroke-width":onSelected ? 2.5 : 1}});
        circle.addEventListener("pointerenter", () => updateTreeMagnifier(x, y, `Node ${{n.idx}}${{n.idx===DATA.best_idx ? " * optimal" : ""}} | J=${{n.g.toFixed(3)}} | tw=${{n.tw.toFixed(3)}}`));
        scene.appendChild(circle);
        if (onSelected || isBestPath || DATA.nodes.length <= 250) {{
          scene.appendChild(el("text", {{x, y:y-23, "text-anchor":"middle", "font-family":"Arial", "font-size":7}}, `J=${{n.g.toFixed(2)}}`));
          scene.appendChild(el("text", {{x, y:y-14, "text-anchor":"middle", "font-family":"Arial", "font-size":7}}, `tw=${{n.tw.toFixed(2)}}`));
        }}
        scene.appendChild(el("text", {{x, y:y+3, "text-anchor":"middle", "font-family":"Arial", "font-size":9, "font-weight":700}}, `${{n.idx}}${{n.idx===DATA.best_idx ? "*" : ""}}`));
      }}
      const omittedPaths = Math.max(0, DATA.terminal_count_total - DATA.terminal_count_shown);
      const omittedNodes = Math.max(0, DATA.node_count_total - DATA.node_count_shown);
      const summary = document.getElementById("treeSummary");
      summary.textContent = omittedPaths || omittedNodes
        ? `Visualization capped: displayed ${{DATA.terminal_count_shown}}/${{DATA.terminal_count_total}} terminal paths and ${{DATA.node_count_shown}}/${{DATA.node_count_total}} nodes; omitted ${{omittedPaths}} paths and ${{omittedNodes}} nodes. Full search result retained.`
        : `Full tree displayed: ${{DATA.terminal_count_total}} terminal paths and ${{DATA.node_count_total}} nodes.`;
      summary.style.display = "block";
    }}

    function selectedOptionIndex(pathTree) {{
      const node = nodeById.get(terminals[cursor]);
      if (node && node.route_choices && node.route_choices.length > pathTree.plan_index) return node.route_choices[pathTree.plan_index];
      if (node && node.route_candidates && node.route_candidates[pathTree.plan_index] && node.route_candidates[pathTree.plan_index].length) {{
        return node.route_candidates[pathTree.plan_index][0];
      }}
      if (Object.prototype.hasOwnProperty.call(pathTree, "selected_index")) return pathTree.selected_index;
      return 0;
    }}

    function buildTrie(pathTree) {{
      const root = {{id:0, label:`B${{pathTree.entrance}}`, depth:0, children:new Map(), terminal:false, parent:null}};
      let nextId = 1;
      for (const option of pathTree.options) {{
        let cur = root;
        for (const m of option.intersections) {{
          if (!cur.children.has(m)) {{
            cur.children.set(m, {{id:nextId++, label:`I${{m}}`, intersection:m, depth:cur.depth+1, children:new Map(), terminal:false, parent:cur}});
          }}
          cur = cur.children.get(m);
        }}
        cur.terminal = true;
      }}
      return root;
    }}

    function flattenTrie(root) {{
      const nodes = [], edges = [];
      function walk(n) {{
        nodes.push(n);
        for (const child of n.children.values()) {{
          edges.push([n, child]);
          walk(child);
        }}
      }}
      walk(root);
      return {{nodes, edges}};
    }}

    function selectedEdges(pathTree, optionIndex) {{
      const option = pathTree.options[optionIndex] || pathTree.options[0];
      const selected = new Set();
      let prev = null;
      for (const m of option.intersections) {{
        const cur = `I${{m}}`;
        if (prev !== null) selected.add(`${{prev}}->${{cur}}`);
        prev = cur;
      }}
      return selected;
    }}

    function activeDecisionEdges() {{
      const selected = terminals[cursor];
      const ids = pathTo(selected);
      const active = new Set();
      for (let k = 1; k < ids.length; k++) {{
        const parent = nodeById.get(ids[k - 1]);
        const child = nodeById.get(ids[k]);
        if (!parent || !child) continue;
        for (const label of edgeDecisionLabels(parent, child)) {{
          if (label.type !== "z" || !label.from || !label.to) continue;
          active.add(`${{label.vehicle}}|${{label.from}}->${{label.to}}`);
        }}
      }}
      return active;
    }}

    function activeDecisionExtraByEdge() {{
      const selected = terminals[cursor];
      const ids = pathTo(selected);
      const out = new Map();
      for (let k = 1; k < ids.length; k++) {{
        const parent = nodeById.get(ids[k - 1]);
        const child = nodeById.get(ids[k]);
        if (!parent || !child) continue;
        for (const label of edgeDecisionLabels(parent, child)) {{
          if (label.type !== "z" || !label.from || !label.to) continue;
          const key = `${{label.vehicle}}|${{label.from}}->${{label.to}}`;
          if (Number.isFinite(Number(label.extra))) out.set(key, Number(label.extra));
        }}
      }}
      return out;
    }}

    function optionLabels(pathTree, option) {{
      return option.intersections.map(m => `I${{m}}`);
    }}

    function branchExtraForEdge(pathTree, a, b) {{
      if (a.children.size <= 1) return null;
      const prefix = [];
      let cur = a;
      while (cur) {{
        prefix.unshift(cur.label);
        cur = cur.parent || null;
      }}
      const branchOptions = [];
      const edgeOptions = [];
      for (const option of pathTree.options) {{
        const labels = optionLabels(pathTree, option);
        const prefixOk = prefix.every((label, index) => labels[index] === label);
        if (!prefixOk || labels.length <= prefix.length) continue;
        branchOptions.push(option);
        if (labels[prefix.length] === b.label) edgeOptions.push(option);
      }}
      if (!branchOptions.length || !edgeOptions.length) return null;
      const branchBest = Math.min(...branchOptions.map(option => option.free_time));
      const edgeBest = Math.min(...edgeOptions.map(option => option.free_time));
      return Math.max(0, edgeBest - branchBest);
    }}

    function pathTreeSvg(pathTree, optionIndex) {{
      const nodeLabels = new Set();
      const edgeMap = new Map();
      const outTargets = new Map();
      for (const option of pathTree.options) {{
        const labels = optionLabels(pathTree, option);
        labels.forEach(label => nodeLabels.add(label));
        for (let k = 0; k < labels.length - 1; k++) {{
          const key = `${{labels[k]}}->${{labels[k+1]}}`;
          edgeMap.set(key, [labels[k], labels[k+1]]);
          if (!outTargets.has(labels[k])) outTargets.set(labels[k], new Set());
          outTargets.get(labels[k]).add(labels[k+1]);
        }}
      }}

      const selectedOption = pathTree.options[optionIndex] || pathTree.options[0];
      const selectedLabels = optionLabels(pathTree, selectedOption);

      function selectedPrefixEdgeExtra(from, to) {{
        const k = selectedLabels.findIndex((label, index) =>
          label === from && selectedLabels[index + 1] === to
        );
        if (k < 0) return null;
        const prefix = selectedLabels.slice(0, k + 1);
        const branchOptions = [];
        const edgeOptions = [];
        for (const option of pathTree.options) {{
          const labels = optionLabels(pathTree, option);
          const prefixOk = prefix.every((label, index) => labels[index] === label);
          if (!prefixOk || labels.length <= k + 1) continue;
          branchOptions.push(option);
          if (labels[k + 1] === to) edgeOptions.push(option);
        }}
        if (!branchOptions.length || !edgeOptions.length) return null;
        const nextChoices = new Set(branchOptions.map(option => optionLabels(pathTree, option)[k + 1]));
        if (nextChoices.size <= 1) return null;
        const branchBest = Math.min(...branchOptions.map(option => option.free_time));
        const edgeBest = Math.min(...edgeOptions.map(option => option.free_time));
        return Math.max(0, edgeBest - branchBest);
      }}

      const usedIntersections = [...nodeLabels]
        .filter(label => label.startsWith("I"))
        .map(label => Number(label.slice(1)))
        .filter(id => DATA.coords && DATA.coords[String(id)]);
      const xs = usedIntersections.map(id => DATA.coords[String(id)][0]);
      const ys = usedIntersections.map(id => DATA.coords[String(id)][1]);
      const minX = Math.min(...xs, 0), maxX = Math.max(...xs, 1);
      const minY = Math.min(...ys, 0), maxY = Math.max(...ys, 1);
      const scale = 48, pad = 24;
      const rawPos = new Map();
      for (const id of usedIntersections) {{
        const [gx, gy] = DATA.coords[String(id)];
        rawPos.set(`I${{id}}`, [pad + (gx - minX) * scale, pad + (maxY - gy) * scale]);
      }}
      const allPoints = [...rawPos.values()];
      const minPx = Math.min(...allPoints.map(p => p[0])) - 14;
      const minPy = Math.min(...allPoints.map(p => p[1])) - 16;
      const maxPx = Math.max(...allPoints.map(p => p[0])) + 22;
      const maxPy = Math.max(...allPoints.map(p => p[1])) + 30;
      const w = maxPx - minPx;
      const h = maxPy - minPy;
      const pos = new Map([...rawPos.entries()].map(([label, p]) => [label, [p[0] - minPx, p[1] - minPy]]));
      const selected = selectedEdges(pathTree, optionIndex);
      const displayW = Math.max(220, Math.min(420, w * 2.8));
      const displayH = Math.max(110, Math.min(240, displayW * h / Math.max(1, w)));
      const svg = el("svg", {{viewBox:`0 0 ${{w}} ${{h}}`, width:displayW, height:displayH}});
      const markerSelected = `pathArrowSelected${{pathTree.vehicle_id}}`;
      const defs = el("defs");
      const selectedMarker = el("marker", {{
        id:markerSelected,
        viewBox:"0 0 10 10",
        refX:"8.5",
        refY:"5",
        markerWidth:"4.5",
        markerHeight:"4.5",
        orient:"auto-start-reverse"
      }});
      selectedMarker.appendChild(el("path", {{d:"M 0 0 L 10 5 L 0 10 z", fill:"#16a34a"}}));
      defs.appendChild(selectedMarker);
      svg.appendChild(defs);
      for (const [key, [from, to]] of edgeMap.entries()) {{
        const [x1,y1] = pos.get(from);
        const [x2,y2] = pos.get(to);
        const lineLen = Math.max(1, Math.hypot(x2 - x1, y2 - y1));
        const ux = (x2 - x1) / lineLen;
        const uy = (y2 - y1) / lineLen;
        const selectedEdge = selected.has(key);
        svg.appendChild(el("line", {{
          x1:x1 + ux * 9,
          y1:y1 + uy * 9,
          x2:x2 - ux * 10,
          y2:y2 - uy * 10,
          stroke:selectedEdge ? "#16a34a" : "#94a3b8",
          "stroke-width":selectedEdge ? 2 : 1,
          "marker-end":selectedEdge ? `url(#${{markerSelected}})` : "none"
        }}));
        const extra = selectedPrefixEdgeExtra(from, to);
        if (selectedEdge && extra !== null) {{
          const lx = (x1 + x2) / 2;
          const ly = (y1 + y2) / 2 - 6;
          svg.appendChild(el("rect", {{
            x:lx-18, y:ly-8, width:36, height:11, rx:2,
            fill:"#ffffff",
            stroke: extra > 1e-8 ? "#f59e0b" : "#16a34a",
            "stroke-width":1
          }}));
          svg.appendChild(el("text", {{
            x:lx, y:ly,
            "text-anchor":"middle",
            "dominant-baseline":"middle",
            "font-family":"Arial",
            "font-size":7,
            "font-weight":700,
            fill: extra > 1e-8 ? "#92400e" : "#166534"
          }}, `+${{extra.toFixed(2)}}s`));
        }}
      }}
      for (const label of nodeLabels) {{
        if (!pos.has(label)) continue;
        const [x,y] = pos.get(label);
        const isBranch = outTargets.has(label) && outTargets.get(label).size > 1;
        const isTerminal = !outTargets.has(label) && label.startsWith("I");
        svg.appendChild(el("circle", {{cx:x, cy:y, r:isBranch ? 9 : 8, fill:isTerminal ? "#fef3c7" : "#f8fafc", stroke:isBranch ? "#2563eb" : "#64748b", "stroke-width":isBranch ? 2 : 1}}));
        svg.appendChild(el("text", {{x, y:y+2, "text-anchor":"middle", "font-family":"Arial", "font-size":7, "font-weight":700}}, label));
      }}
      return svg;
    }}

    function expandedPathTreeSvg(pathTree, optionIndex, activeEdges=new Set(), activeExtraByEdge=new Map()) {{
      const root = buildTrie(pathTree);
      const flat = flattenTrie(root);
      const selectedOption = pathTree.options[optionIndex] || pathTree.options[0];
      const selectedLabels = selectedOption ? optionLabels(pathTree, selectedOption) : [];
      const selected = new Set();
      let previous = `B${{pathTree.entrance}}`;
      for (const label of selectedLabels) {{
        selected.add(`${{previous}}->${{label}}`);
        previous = label;
      }}

      const leafOrder = new Map();
      let leafIndex = 0;
      function assignY(node) {{
        if (!node.children.size) {{
          leafOrder.set(node.id, leafIndex++);
          return leafOrder.get(node.id);
        }}
        const childYs = [...node.children.values()].map(assignY);
        const y = childYs.reduce((sum, item) => sum + item, 0) / childYs.length;
        leafOrder.set(node.id, y);
        return y;
      }}
      assignY(root);

      const maxDepth = Math.max(...flat.nodes.map(node => node.depth), 1);
      const xGap = 46;
      const yGap = 24;
      const margin = 18;
      const width = margin * 2 + maxDepth * xGap;
      const height = margin * 2 + Math.max(1, leafIndex - 1) * yGap;
      const pos = new Map(flat.nodes.map(node => [
        node.id,
        [margin + node.depth * xGap, margin + (leafOrder.get(node.id) || 0) * yGap]
      ]));
      const displayW = Math.max(260, Math.min(520, width * 1.6));
      const displayH = Math.max(130, Math.min(360, displayW * height / Math.max(1, width)));
      const svg = el("svg", {{viewBox:`0 0 ${{width}} ${{height}}`, width:displayW, height:displayH}});

      for (const [a, b] of flat.edges) {{
        const [x1, y1] = pos.get(a.id);
        const [x2, y2] = pos.get(b.id);
        const edgeKey = `${{a.label}}->${{b.label}}`;
        const activeKey = `${{pathTree.vehicle_id}}|${{edgeKey}}`;
        const activeEdge = activeEdges.has(activeKey);
        const selectedEdge = selected.has(edgeKey);
        svg.appendChild(el("line", {{
          x1:x1 + 8,
          y1,
          x2:x2 - 9,
          y2,
          stroke:activeEdge ? "#f97316" : (selectedEdge ? "#16a34a" : "#cbd5e1"),
          "stroke-width":activeEdge ? 3.2 : (selectedEdge ? 2 : 1.1)
        }}));
        const extra = activeExtraByEdge.get(activeKey);
        if (activeEdge && Number.isFinite(extra)) {{
          const lx = (x1 + x2) / 2;
          const ly = (y1 + y2) / 2 - 7;
          svg.appendChild(el("rect", {{
            x:lx-18, y:ly-8, width:36, height:11, rx:2,
            fill:"#ffffff",
            stroke:"#f97316",
            "stroke-width":1
          }}));
          svg.appendChild(el("text", {{
            x:lx, y:ly,
            "text-anchor":"middle",
            "dominant-baseline":"middle",
            "font-family":"Arial",
            "font-size":7,
            "font-weight":700,
            fill:"#9a3412"
          }}, `+${{extra.toFixed(2)}}s`));
        }}
      }}

      for (const node of flat.nodes) {{
        const [x, y] = pos.get(node.id);
        const selectedNode = node.label === `B${{pathTree.entrance}}` || selectedLabels.includes(node.label);
        const fill = selectedNode ? "#dcfce7" : "#f8fafc";
        const stroke = node.children.size > 1 ? "#2563eb" : "#64748b";
        svg.appendChild(el("circle", {{cx:x, cy:y, r:7.5, fill, stroke, "stroke-width":1.2}}));
        svg.appendChild(el("text", {{x, y:y+2, "text-anchor":"middle", "font-family":"Arial", "font-size":6.5, "font-weight":700}}, node.label));
      }}
      return svg;
    }}

    function roundedTrafficPath(points, radius=12) {{
      if (!points.length) return "";
      if (points.length === 1) return `M ${{points[0][0]}} ${{points[0][1]}}`;
      let path = `M ${{points[0][0]}} ${{points[0][1]}}`;
      for (let i = 1; i < points.length - 1; i += 1) {{
        const previous = points[i - 1], current = points[i], next = points[i + 1];
        const inDx = current[0] - previous[0], inDy = current[1] - previous[1];
        const outDx = next[0] - current[0], outDy = next[1] - current[1];
        const inLength = Math.max(1e-9, Math.hypot(inDx, inDy));
        const outLength = Math.max(1e-9, Math.hypot(outDx, outDy));
        const inRadius = Math.min(radius, inLength * 0.35);
        const outRadius = Math.min(radius, outLength * 0.35);
        const before = [current[0] - inDx / inLength * inRadius, current[1] - inDy / inLength * inRadius];
        const after = [current[0] + outDx / outLength * outRadius, current[1] + outDy / outLength * outRadius];
        path += ` L ${{before[0]}} ${{before[1]}} Q ${{current[0]}} ${{current[1]}} ${{after[0]}} ${{after[1]}}`;
      }}
      const last = points[points.length - 1];
      return path + ` L ${{last[0]}} ${{last[1]}}`;
    }}

    function renderBasicMap() {{
      const panel = document.getElementById("basicMapPanel");
      panel.innerHTML = "";
      const coordEntries = Object.entries(DATA.coords || {{}});
      if (!coordEntries.length) {{
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "No map geometry";
        panel.appendChild(empty);
        return;
      }}

      const xs = coordEntries.map(([, xy]) => xy[0]);
      const ys = coordEntries.map(([, xy]) => xy[1]);
      const minX = Math.min(...xs), maxX = Math.max(...xs);
      const minY = Math.min(...ys), maxY = Math.max(...ys);
      const scale = 78;
      const margin = 68;
      const roadWidth = 38;
      const intersectionSize = 40;
      const portReach = 52;
      const legendRows = Math.max(1, (DATA.path_trees || []).length);
      const legendHeight = 18 + legendRows * 14;
      const mapWidth = (maxX - minX) * scale + margin * 2;
      const mapHeight = (maxY - minY) * scale + margin * 2;
      const width = mapWidth;
      const height = mapHeight + legendHeight;
      const point = (x, y) => [margin + (x - minX) * scale, margin + (maxY - y) * scale];
      const screenDelta = {{L:[-1,0], D:[0,1], R:[1,0], U:[0,-1]}};
      const portsById = new Map((DATA.ports || []).map(port => [Number(port.id), port]));
      const portGeometry = port => {{
        const xy = DATA.coords[String(port.intersection)];
        const direction = screenDelta[port.direction] || [0, 0];
        if (!xy) return null;
        const center = point(xy[0], xy[1]);
        return {{
          center,
          direction,
          end:[center[0] + direction[0] * portReach, center[1] + direction[1] * portReach]
        }};
      }};
      const displayW = Math.max(440, Math.min(780, width * 2.05));
      const displayH = Math.max(300, Math.min(650, displayW * height / Math.max(1, width)));
      const svg = el("svg", {{viewBox:`0 0 ${{width}} ${{height}}`, width:displayW, height:displayH}});
      svg.appendChild(el("rect", {{x:0, y:0, width, height, fill:"#ffffff"}}));

      const defs = el("defs");
      const marker = (id, color, size=5) => {{
        const item = el("marker", {{id, viewBox:"0 0 10 10", refX:8.5, refY:5, markerWidth:size, markerHeight:size, orient:"auto-start-reverse"}});
        item.appendChild(el("path", {{d:"M 0 0 L 10 5 L 0 10 z", fill:color}}));
        defs.appendChild(item);
      }};
      marker("trafficIncoming", "#2e7d32", 4.5);
      marker("trafficOutgoing", "#9a4d2e", 4.5);
      const routeColors = ["#0ea5e9", "#22c55e", "#d946ef", "#f97316", "#8b5cf6", "#ef4444", "#14b8a6", "#ca8a04"];
      routeColors.forEach((color, index) => marker(`robotRouteArrow${{index}}`, color, 5.5));
      svg.appendChild(defs);

      const drawRoad = (start, end) => {{
        svg.appendChild(el("line", {{x1:start[0], y1:start[1], x2:end[0], y2:end[1], stroke:"#b8bdc5", "stroke-width":roadWidth, "stroke-linecap":"butt"}}));
        svg.appendChild(el("line", {{x1:start[0], y1:start[1], x2:end[0], y2:end[1], stroke:"#eab308", "stroke-width":1.2, "stroke-dasharray":"7 6"}}));
      }};

      for (const road of DATA.roads || []) {{
        const a = DATA.coords[String(road.a)], b = DATA.coords[String(road.b)];
        if (!a || !b) continue;
        drawRoad(point(a[0], a[1]), point(b[0], b[1]));
      }}
      for (const port of DATA.ports || []) {{
        const geometry = portGeometry(port);
        if (geometry) drawRoad(geometry.center, geometry.end);
      }}

      for (const item of coordEntries) {{
        const [gx, gy] = item[1];
        const [x, y] = point(gx, gy);
        svg.appendChild(el("rect", {{
          x:x-intersectionSize/2, y:y-intersectionSize/2,
          width:intersectionSize, height:intersectionSize,
          fill:"#b8bdc5", stroke:"#1e40af", "stroke-width":1.5, "stroke-dasharray":"5 3"
        }}));
      }}

      for (const port of DATA.ports || []) {{
        const geometry = portGeometry(port);
        if (!geometry) continue;
        const [dx, dy] = geometry.direction;
        const perpendicular = [-dy, dx];
        const incomingStart = [geometry.center[0] + dx*45 + perpendicular[0]*7, geometry.center[1] + dy*45 + perpendicular[1]*7];
        const incomingEnd = [geometry.center[0] + dx*25 + perpendicular[0]*7, geometry.center[1] + dy*25 + perpendicular[1]*7];
        const outgoingStart = [geometry.center[0] + dx*25 - perpendicular[0]*7, geometry.center[1] + dy*25 - perpendicular[1]*7];
        const outgoingEnd = [geometry.center[0] + dx*45 - perpendicular[0]*7, geometry.center[1] + dy*45 - perpendicular[1]*7];
        svg.appendChild(el("line", {{x1:incomingStart[0], y1:incomingStart[1], x2:incomingEnd[0], y2:incomingEnd[1], stroke:"#2e7d32", "stroke-width":1.8, "marker-end":"url(#trafficIncoming)"}}));
        svg.appendChild(el("line", {{x1:outgoingStart[0], y1:outgoingStart[1], x2:outgoingEnd[0], y2:outgoingEnd[1], stroke:"#9a4d2e", "stroke-width":1.8, "marker-end":"url(#trafficOutgoing)"}}));
      }}

      const routeItems = (DATA.path_trees || []).map((pathTree, index) => {{
        const shortestIndex = Number.isInteger(pathTree.shortest_index) ? pathTree.shortest_index : 0;
        return {{pathTree, option:pathTree.options[shortestIndex] || pathTree.options[0], index}};
      }}).filter(item => item.option);
      const dashPatterns = ["", "9 5", "3 4", "12 4 3 4", "2 3", "14 5"];
      routeItems.forEach((item, routeIndex) => {{
        const entrance = portsById.get(Number(item.pathTree.entrance));
        const exitPort = portsById.get(Number(item.pathTree.exit));
        const entranceGeometry = entrance ? portGeometry(entrance) : null;
        const exitGeometry = exitPort ? portGeometry(exitPort) : null;
        if (!entranceGeometry || !exitGeometry) return;
        const routePoints = [
          entranceGeometry.end,
          ...item.option.intersections.map(id => {{
            const xy = DATA.coords[String(id)];
            return point(xy[0], xy[1]);
          }}),
          exitGeometry.end
        ];
        const color = routeColors[routeIndex % routeColors.length];
        const pathData = roundedTrafficPath(routePoints, 13);
        svg.appendChild(el("path", {{d:pathData, fill:"none", stroke:"#ffffff", "stroke-width":6.5, "stroke-linejoin":"round", "stroke-linecap":"round", opacity:0.82}}));
        svg.appendChild(el("path", {{
          d:pathData, fill:"none", stroke:color, "stroke-width":3.2,
          "stroke-linejoin":"round", "stroke-linecap":"round",
          "stroke-dasharray":dashPatterns[routeIndex % dashPatterns.length],
          "marker-end":`url(#robotRouteArrow${{routeIndex % routeColors.length}})`
        }}));
        const [labelX, labelY] = entranceGeometry.end;
        svg.appendChild(el("rect", {{x:labelX-9, y:labelY-8, width:18, height:12, rx:3, fill:"#ffffff", stroke:color, "stroke-width":1.4}}));
        svg.appendChild(el("text", {{x:labelX, y:labelY+0.5, "text-anchor":"middle", "font-family":"Arial", "font-size":6.5, "font-weight":700, fill:color}}, `N${{item.pathTree.vehicle_id}}`));
      }});

      for (const item of coordEntries) {{
        const id = Number(item[0]);
        const [x, y] = point(item[1][0], item[1][1]);
        svg.appendChild(el("text", {{x, y:y+4, "text-anchor":"middle", "font-family":"Arial", "font-size":12, "font-weight":700, fill:"#1f2937", stroke:"#b8bdc5", "stroke-width":3, "paint-order":"stroke fill"}}, `I${{id}}`));
      }}
      for (const port of DATA.ports || []) {{
        const geometry = portGeometry(port);
        if (!geometry) continue;
        const [dx, dy] = geometry.direction;
        const labelX = geometry.end[0] + dx * 11;
        const labelY = geometry.end[1] + dy * 11;
        svg.appendChild(el("text", {{
          x:labelX, y:labelY+3, "text-anchor":"middle",
          "font-family":"Arial", "font-size":9, "font-weight":700, fill:"#111827",
          stroke:"#ffffff", "stroke-width":3, "paint-order":"stroke fill"
        }}, `B${{port.id}}`));
      }}

      const legendY = mapHeight + 9;
      svg.appendChild(el("text", {{x:8, y:legendY, "font-family":"Arial", "font-size":7.5, "font-weight":700, fill:"#475569"}}, "Robot shortest paths (reference)"));
      routeItems.forEach((item, index) => {{
        const y = legendY + 12 + index * 14;
        const color = routeColors[index % routeColors.length];
        svg.appendChild(el("line", {{x1:10, y1:y-2, x2:34, y2:y-2, stroke:color, "stroke-width":3, "stroke-dasharray":dashPatterns[index % dashPatterns.length]}}));
        const routeText = `N${{item.pathTree.vehicle_id}}: B${{item.pathTree.entrance}} → ${{item.option.intersections.map(id => `I${{id}}`).join(" → ")}} → B${{item.pathTree.exit}}`;
        svg.appendChild(el("text", {{x:39, y, "font-family":"Arial", "font-size":7.5, "font-weight":700, fill:"#334155"}}, routeText));
      }});
      panel.appendChild(svg);
    }}

    function pathCostText(option) {{
      if (!option) return "";
      const free = Number(option.free_time || 0);
      const extra = Number(option.extra_time || 0);
      const lambda = Number(DATA.lambda_path || 1);
      return `free=${{free.toFixed(3)}}s, extra=${{extra.toFixed(3)}}s, obj=${{(lambda * extra).toFixed(3)}}`;
    }}

    function pathText(option, includeCost=false) {{
      if (!option) return "";
      const base = `[${{option.intersections.map(m => `I${{m}}`).join(",")}}]`;
      return includeCost ? `${{base}} (${{pathCostText(option)}})` : base;
    }}

    function renderPathSummary() {{
      const panel = document.getElementById("pathSummaryPanel");
      panel.innerHTML = "";
      if (!DATA.path_trees || !DATA.path_trees.length) {{
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "No route options";
        panel.appendChild(empty);
        return;
      }}

      const table = document.createElement("table");
      table.className = "path-table";
      table.innerHTML = "<thead><tr><th>Vehicle</th><th>Ent</th><th>Ext</th><th>Selected Path</th><th>Candidate Paths with Path Cost</th></tr></thead>";
      const tbody = document.createElement("tbody");
      for (const pathTree of DATA.path_trees) {{
        const optionIndex = selectedOptionIndex(pathTree);
        const selected = pathTree.options[optionIndex] || pathTree.options[0];
        const tr = document.createElement("tr");
        const candidateList = pathTree.options.map((option, index) => {{
          const cls = index === optionIndex ? "path-pill selected" : "path-pill";
          return `<span class="${{cls}}"><span>${{index + 1}}: ${{pathText(option)}}</span><span class="path-cost">${{pathCostText(option)}}</span></span>`;
        }}).join("");
        tr.innerHTML = `
          <td>N${{pathTree.vehicle_id}}</td>
          <td>B${{pathTree.entrance}}</td>
          <td>B${{pathTree.exit}}</td>
          <td>${{selected ? pathText(selected, true) : ""}}</td>
          <td><div class="path-list">${{candidateList}}</div></td>
        `;
        tbody.appendChild(tr);
      }}
      table.appendChild(tbody);
      panel.appendChild(table);
    }}

    function branchChoices(pathTree) {{
      const byPrefix = new Map();
      for (const option of pathTree.options || []) {{
        const labels = option.intersections.map(m => `I${{m}}`);
        for (let k = 0; k < labels.length - 1; k++) {{
          const prefix = labels.slice(0, k + 1);
          const key = prefix.join("|");
          if (!byPrefix.has(key)) byPrefix.set(key, {{prefix, choices:new Set()}});
          byPrefix.get(key).choices.add(labels[k + 1]);
        }}
      }}
      return [...byPrefix.values()]
        .filter(item => item.choices.size > 1)
        .sort((a, b) => a.prefix.length - b.prefix.length || a.prefix.join(",").localeCompare(b.prefix.join(",")));
    }}

    function renderPathBranches() {{
      const panel = document.getElementById("pathBranchPanel");
      panel.innerHTML = "";
      if (!DATA.path_trees || !DATA.path_trees.length) {{
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "No path-selection branches";
        panel.appendChild(empty);
        return;
      }}

      const note = document.createElement("div");
      note.className = "path-note";
      note.textContent = "Each row is one actual route-choice point. The green choice is the next intersection used by the currently selected path.";
      panel.appendChild(note);

      const table = document.createElement("table");
      table.className = "path-table";
      table.innerHTML = "<thead><tr><th>Vehicle</th><th>Ent</th><th>Ext</th><th>Reached Prefix</th><th>Next Choices</th></tr></thead>";
      const tbody = document.createElement("tbody");
      for (const pathTree of DATA.path_trees) {{
        const optionIndex = selectedOptionIndex(pathTree);
        const selected = pathTree.options[optionIndex] || pathTree.options[0];
        const selectedLabels = selected ? selected.intersections.map(m => `I${{m}}`) : [];
        for (const item of branchChoices(pathTree)) {{
          const selectedNext = selectedLabels.length > item.prefix.length &&
            item.prefix.every((label, index) => selectedLabels[index] === label)
              ? selectedLabels[item.prefix.length]
              : "";
          const choices = [...item.choices].sort((a, b) => Number(a.slice(1)) - Number(b.slice(1)));
          const choiceText = choices.map(choice => {{
            const cls = choice === selectedNext ? "path-pill selected" : "path-pill";
            return `<span class="${{cls}}">${{choice}}</span>`;
          }}).join("");
          const tr = document.createElement("tr");
          tr.innerHTML = `
            <td>N${{pathTree.vehicle_id}}</td>
            <td>B${{pathTree.entrance}}</td>
            <td>B${{pathTree.exit}}</td>
            <td>[${{item.prefix.join(",")}}]</td>
            <td><div class="path-list">${{choiceText}}</div></td>
          `;
          tbody.appendChild(tr);
        }}
      }}
      table.appendChild(tbody);
      panel.appendChild(table);
    }}

    function renderPathTrees() {{
      const panel = document.getElementById("pathTreePanel");
      panel.innerHTML = "";
      const activeEdges = activeDecisionEdges();
      const activeExtra = activeDecisionExtraByEdge();
      for (const pathTree of DATA.path_trees) {{
        const optionIndex = selectedOptionIndex(pathTree);
        const option = pathTree.options[optionIndex] || pathTree.options[0];
        const activeText = [...activeEdges]
          .filter(key => key.startsWith(`${{pathTree.vehicle_id}}|`))
          .map(key => {{
            const edge = key.split("|")[1];
            const extra = activeExtra.get(key);
            return Number.isFinite(extra) ? `${{edge}}(+${{extra.toFixed(2)}}s)` : edge;
          }})
          .join(", ");
        const card = document.createElement("section");
        card.className = "path-card";
        card.innerHTML = `<h3>N${{pathTree.vehicle_id}} selected: ${{option ? pathText(option) : ""}}${{activeText ? ` | z edge: ${{activeText}}` : ""}}</h3>`;
        const visuals = document.createElement("div");
        visuals.className = "path-card-visuals";
        const dagWrap = document.createElement("div");
        dagWrap.innerHTML = '<div class="path-visual-title">System DAG</div>';
        dagWrap.appendChild(pathTreeSvg(pathTree, optionIndex));
        const treeWrap = document.createElement("div");
        treeWrap.innerHTML = '<div class="path-visual-title">Expanded Path Selection Tree</div>';
        treeWrap.appendChild(expandedPathTreeSvg(pathTree, optionIndex, activeEdges, activeExtra));
        visuals.appendChild(dagWrap);
        visuals.appendChild(treeWrap);
        card.appendChild(visuals);
        panel.appendChild(card);
      }}
    }}

    function turnLabel(planIndex, taskIndex, node) {{
      const plan = DATA.plans[planIndex];
      if (!plan) return "";
      if (plan.turns && plan.turns.length) return plan.turns[Math.max(0, taskIndex - 1)] || "";
      if (plan.turns_by_option && node.route_candidates && node.route_candidates[planIndex]) {{
        const optionIndex = node.route_candidates[planIndex].length ? node.route_candidates[planIndex][0] : 0;
        const turns = plan.turns_by_option[optionIndex] || [];
        return turns[Math.max(0, taskIndex - 1)] || "";
      }}
      return "";
    }}

    function movementRouteId(planIndex, taskIndex, node) {{
      const plan = DATA.plans[planIndex];
      if (!plan) return null;
      const offset = Math.max(0, taskIndex - 1);
      if (plan.route_ids && plan.route_ids.length) return plan.route_ids[offset] ?? null;
      if (plan.route_ids_by_option && node.route_candidates && node.route_candidates[planIndex]) {{
        const candidates = node.route_candidates[planIndex];
        const optionIndex = candidates.length ? candidates[0] : 0;
        const routeIds = plan.route_ids_by_option[optionIndex] || [];
        return routeIds[offset] ?? null;
      }}
      return null;
    }}

    const conflictingRoutePairKeys = new Set(
      (DATA.conflicting_route_pairs || []).map(pair => `${{pair[0]}}|${{pair[1]}}`)
    );

    function routePairConflicts(left, right) {{
      const a = Math.min(left, right);
      const b = Math.max(left, right);
      return conflictingRoutePairKeys.has(`${{a}}|${{b}}`);
    }}

    function conflictingSegmentPairs(segs, node) {{
      const visits = segs.map(seg => {{
        const planIndex = DATA.plans.findIndex(p => p.vehicle_id === seg.vehicle_id);
        return {{...seg, route_id:movementRouteId(planIndex, seg.task_index, node)}};
      }});
      const pairs = [];
      for (let i = 0; i < visits.length; i += 1) {{
        for (let j = i + 1; j < visits.length; j += 1) {{
          const left = visits[i], right = visits[j];
          if (!Number.isInteger(left.route_id) || !Number.isInteger(right.route_id)) continue;
          if (!routePairConflicts(left.route_id, right.route_id)) continue;
          pairs.push({{left, right}});
        }}
      }}
      pairs.sort((a, b) =>
        Math.min(a.left.route_id, a.right.route_id) - Math.min(b.left.route_id, b.right.route_id) ||
        Math.max(a.left.route_id, a.right.route_id) - Math.max(b.left.route_id, b.right.route_id) ||
        a.left.vehicle_id - b.left.vehicle_id || a.right.vehicle_id - b.right.vehicle_id
      );
      return pairs;
    }}

    function turnCode(turn) {{
      const key = String(turn || "").toLowerCase();
      if (key === "left") return "1";
      if (key === "straight") return "2";
      if (key === "right") return "3";
      return "?";
    }}

    function niceTickStep(maxT, targetTicks) {{
      const raw = Math.max(maxT / Math.max(1, targetTicks), 1e-9);
      const power = Math.pow(10, Math.floor(Math.log10(raw)));
      const scaled = raw / power;
      let nice = 10;
      if (scaled <= 1) nice = 1;
      else if (scaled <= 2) nice = 2;
      else if (scaled <= 5) nice = 5;
      return nice * power;
    }}

    function timeTickText(t) {{
      if (Math.abs(t - Math.round(t)) < 1e-8) return String(Math.round(t));
      return t.toFixed(1);
    }}

    function timeEventTicks(segs, maxT, step) {{
      const ticks = [];
      for (let t = 0; t <= maxT + 1e-9; t += step) {{
        ticks.push({{t, kind:"base", label:timeTickText(t)}});
      }}
      for (const s of segs) {{
        ticks.push({{t:s.requested_time, kind:"alpha", label:fmtTime(s.requested_time)}});
        ticks.push({{t:s.start_time, kind:"beta", label:fmtTime(s.start_time)}});
        ticks.push({{t:s.end_time, kind:"gamma", label:fmtTime(s.end_time)}});
      }}
      if (!ticks.some(item => Math.abs(item.t - maxT) < 1e-8)) {{
        ticks.push({{t:maxT, kind:"base", label:timeTickText(maxT)}});
      }}
      ticks.sort((a,b) => a.t - b.t || a.kind.localeCompare(b.kind));
      const out = [];
      for (const tick of ticks) {{
        const previous = out[out.length - 1];
        if (previous && Math.abs(previous.t - tick.t) < 1e-7 && previous.kind === tick.kind) continue;
        out.push(tick);
      }}
      return out;
    }}

    function localScheduleSvg(resource, segs, attempts, node, globalMaxT) {{
      const pairMode = Boolean(DATA.trajectory_conflict_filter);
      const conflictPairs = pairMode ? conflictingSegmentPairs(segs, node) : [];
      const width = pairMode ? 640 : 520;
      const rowH = 26;
      const rows = [...new Set([...segs.map(s => s.vehicle_id), ...attempts.map(a => a.vehicle_id)])].sort((a,b)=>a-b);
      const validationRowCount = pairMode ? Math.max(1, conflictPairs.length) : 1;
      const height = 74 + Math.max(1, rows.length + validationRowCount) * rowH;
      const maxT = Math.max(1, globalMaxT || 1);
      const x0 = pairMode ? 126 : 58, x1 = width - 14;
      const xOf = t => x0 + t / maxT * (x1 - x0);
      const svg = el("svg", {{viewBox:`0 0 ${{width}} ${{height}}`, width:"100%"}});
      rows.forEach((vehicleId, row) => {{
        const y = 20 + row * rowH;
        svg.appendChild(el("text", {{x:12, y:y+15, "font-family":"Arial", "font-size":9, "font-weight":700}}, `N${{vehicleId}}`));
        svg.appendChild(el("rect", {{x:x0, y:y, width:x1-x0, height:18, fill:"#f8fafc", stroke:"#dbe3ef"}}));
      }});
      const resourceY = 20 + rows.length * rowH;
      if (!pairMode) {{
        svg.appendChild(el("text", {{x:12, y:resourceY+15, "font-family":"Arial", "font-size":9, "font-weight":700}}, `I${{resource}}`));
        svg.appendChild(el("rect", {{x:x0, y:resourceY, width:x1-x0, height:18, fill:"#f8fafc", stroke:"#dbe3ef"}}));
      }}
      segs.forEach(s => {{
        const row = rows.indexOf(s.vehicle_id);
        const y = 20 + row * rowH;
        const xs = xOf(s.start_time);
        const xe = xOf(s.end_time);
        const xa = xOf(s.requested_time);
        if (s.start_time > s.requested_time + 1e-9) {{
          svg.appendChild(el("rect", {{x:xa, y:y, width:Math.max(1, xs-xa), height:18, fill:"#e5e7eb", stroke:"#94a3b8"}}));
        }}
        svg.appendChild(el("rect", {{x:xs, y:y, width:Math.max(2, xe-xs), height:18, fill:"#4ade80"}}));
        const turn = turnLabel(DATA.plans.findIndex(p => p.vehicle_id === s.vehicle_id), s.task_index, node);
        svg.appendChild(el("text", {{x:(xs+xe)/2, y:y+12, "text-anchor":"middle", "font-family":"Arial", "font-size":8, "font-weight":700}}, `C${{turnCode(turn)}}, K${{s.task_index}}`));
        if (!pairMode) {{
          svg.appendChild(el("rect", {{x:xs, y:resourceY, width:Math.max(2, xe-xs), height:18, fill:"none", stroke:"#22c55e", "stroke-width":1.5}}));
          svg.appendChild(el("text", {{x:(xs+xe)/2, y:resourceY+12, "text-anchor":"middle", "font-family":"Arial", "font-size":8, "font-weight":700}}, `N${{s.vehicle_id}}`));
        }}
      }});
      if (pairMode && !conflictPairs.length) {{
        svg.appendChild(el("text", {{x:12, y:resourceY+15, "font-family":"Arial", "font-size":8, "font-weight":700, fill:"#15803d"}}, `I${{resource}}: no active conflict pair`));
        svg.appendChild(el("rect", {{x:x0, y:resourceY, width:x1-x0, height:18, fill:"#f0fdf4", stroke:"#86efac"}}));
      }}
      conflictPairs.forEach((pair, pairIndex) => {{
        const y = resourceY + pairIndex * rowH;
        const left = pair.left, right = pair.right;
        const routeA = Math.min(left.route_id, right.route_id);
        const routeB = Math.max(left.route_id, right.route_id);
        const overlapStart = Math.max(left.start_time, right.start_time);
        const overlapEnd = Math.min(left.end_time, right.end_time);
        const violated = overlapEnd > overlapStart + 1e-9;
        const label = `I${{resource}}, R${{routeA}}–R${{routeB}} ${{violated ? "✗" : "✓"}}`;
        svg.appendChild(el("text", {{x:12, y:y+15, "font-family":"Arial", "font-size":8, "font-weight":700, fill:violated ? "#b91c1c" : "#15803d"}}, label));
        svg.appendChild(el("rect", {{x:x0, y, width:x1-x0, height:18, fill:violated ? "#fef2f2" : "#f0fdf4", stroke:violated ? "#fca5a5" : "#86efac"}}));
        [left, right].forEach((visit, visitIndex) => {{
          const xs = xOf(visit.start_time), xe = xOf(visit.end_time);
          const barY = y + 1 + visitIndex * 8;
          svg.appendChild(el("rect", {{x:xs, y:barY, width:Math.max(2, xe-xs), height:7, fill:"#4ade80", stroke:"#22c55e", "stroke-width":0.8}}));
          svg.appendChild(el("text", {{x:(xs+xe)/2, y:barY+6, "text-anchor":"middle", "font-family":"Arial", "font-size":5.5, "font-weight":700}}, `N${{visit.vehicle_id}},K${{visit.task_index}},R${{visit.route_id}}`));
        }});
        if (violated) {{
          svg.appendChild(el("rect", {{x:xOf(overlapStart), y, width:Math.max(2, xOf(overlapEnd)-xOf(overlapStart)), height:18, fill:"rgba(239,68,68,0.45)", stroke:"#dc2626", "stroke-width":1.2}}));
        }}
      }});
      attempts.forEach(a => {{
        const row = rows.indexOf(a.vehicle_id);
        if (row < 0) return;
        const y = 20 + row * rowH;
        const xs = xOf(a.start_time);
        const xe = xOf(a.end_time);
        svg.appendChild(el("rect", {{
          x:xs,
          y:y,
          width:Math.max(2, xe-xs),
          height:18,
          fill:"rgba(251,146,60,0.26)",
          stroke:"#f97316",
          "stroke-width":1.4,
          "stroke-dasharray":"4 2"
        }}));
      }});
      segs.forEach(s => {{
        const row = rows.indexOf(s.vehicle_id);
        const y = 20 + row * rowH;
        const xa = xOf(s.requested_time);
        svg.appendChild(el("line", {{x1:xa, y1:y-1, x2:xa, y2:y+19, stroke:"#0000ff", "stroke-width":1.6, "stroke-dasharray":"3 2"}}));
      }});
      const axisY = height - 24;
      svg.appendChild(el("line", {{x1:x0, y1:axisY, x2:x1, y2:axisY, stroke:"#111827"}}));
      const step = niceTickStep(maxT, 5);
      const tickRows = new Map();
      for (const tick of timeEventTicks(segs, maxT, step)) {{
        const x = xOf(tick.t);
        const color = tick.kind === "alpha" ? "#0000ff" : (tick.kind === "beta" ? "#d97706" : (tick.kind === "gamma" ? "#15803d" : "#111827"));
        const len = tick.kind === "base" ? 4 : 7;
        const recent = [...tickRows.entries()].filter(([px]) => Math.abs(px - x) < 34);
        const row = recent.length ? (Math.max(...recent.map(([,r]) => r)) + 1) % 3 : 0;
        tickRows.set(x, row);
        svg.appendChild(el("line", {{x1:x, y1:axisY, x2:x, y2:axisY+len, stroke:color, "stroke-width":tick.kind === "base" ? 1 : 1.25}}));
        svg.appendChild(el("text", {{
          x,
          y:axisY + 14 + row * 10,
          "text-anchor":"middle",
          "font-family":"Arial",
          "font-size":tick.kind === "base" ? 7 : 6,
          "font-weight":tick.kind === "base" ? 400 : 700,
          fill:color
        }}, tick.label));
      }}
      svg.appendChild(el("text", {{x:(x0+x1)/2, y:height-2, "text-anchor":"middle", "font-family":"Arial", "font-size":8, "font-weight":700}}, "Time (seconds)"));
      return svg;
    }}

    function orderedResources() {{
      if (!DATA.coords || !Object.keys(DATA.coords).length) {{
        return DATA.resources.slice().sort((a,b) => a-b);
      }}
      return Object.keys(DATA.coords).map(Number).sort((a,b) => a-b);
    }}

    function renderSchedule() {{
      const selected = terminals[cursor];
      const node = nodeById.get(selected);
      const isBest = selected === DATA.best_idx;
      document.getElementById("meta").textContent = `Path ${{cursor+1}}/${{terminals.length}}: node ${{selected}}${{isBest ? " * optimal" : ""}} | J = ${{node.g.toFixed(3)}} | shown paths ${{DATA.terminal_count_shown}}/${{DATA.terminal_count_total}}, nodes ${{DATA.node_count_shown}}/${{DATA.node_count_total}}`;
      const validationMode = DATA.trajectory_conflict_filter
        ? "trajectory-pair contention validation"
        : "conservative intersection-occupancy validation";
      document.getElementById("scheduleTitle").textContent = `Schedule for node ${{selected}}${{isBest ? " *" : ""}} — ${{validationMode}}`;
      const panel = document.getElementById("schedulePanel");
      panel.innerHTML = "";
      const globalMaxT = Math.max(
        1,
        ...node.segments.flatMap(s => [s.requested_time, s.end_time]),
        ...(node.attempts || []).flatMap(a => [a.start_time, a.end_time])
      );
      for (const resource of orderedResources()) {{
        const segs = node.segments.filter(s => s.resource === resource);
        const attempts = (node.attempts || []).filter(a => a.resource === resource);
        if (!segs.length && !attempts.length) continue;
        const card = document.createElement("section");
        card.className = "icard";
        card.innerHTML = `<h3>I${{resource}}</h3>`;
        card.appendChild(localScheduleSvg(resource, segs, attempts, node, globalMaxT));
        panel.appendChild(card);
      }}
    }}

    function renderAll() {{
      renderTree();
      renderBasicMap();
      renderPathSummary();
      renderPathBranches();
      renderPathTrees();
      renderSchedule();
    }}

    document.addEventListener("keydown", event => {{
      if (!["ArrowRight","ArrowDown","ArrowLeft","ArrowUp"].includes(event.key)) return;
      event.preventDefault();
      const step = (event.key === "ArrowRight" || event.key === "ArrowDown") ? 1 : -1;
      cursor = (cursor + step + terminals.length) % terminals.length;
      renderAll();
    }});

    const treeSvg = document.getElementById("treeSvg");
    treeSvg.addEventListener("wheel", event => {{
      if (!event.ctrlKey) return;
      event.preventDefault();
      zoomTreeAtCenter(event.deltaY < 0 ? 0.85 : 1.18);
    }}, {{passive:false}});

    treeSvg.addEventListener("dblclick", event => {{
      event.preventDefault();
      resetTreeZoom();
    }});
    treeSvg.addEventListener("pointerdown", event => {{
      if (event.button !== 0 || !treeBaseViewBox || !treeViewBox) return;
      if (treeViewBox[2] >= treeBaseViewBox[2] * 0.995) return;
      treeSvg.setPointerCapture(event.pointerId);
      treeDrag = {{
        pointerId:event.pointerId,
        clientX:event.clientX,
        clientY:event.clientY,
        box:treeViewBox.slice(),
        rect:treeSvg.getBoundingClientRect()
      }};
      treeSvg.classList.add("dragging");
    }});
    treeSvg.addEventListener("pointermove", event => {{
      if (!treeDrag || event.pointerId !== treeDrag.pointerId) return;
      const scaleX = treeDrag.box[2] / Math.max(1, treeDrag.rect.width);
      const scaleY = treeDrag.box[3] / Math.max(1, treeDrag.rect.height);
      const rawX = treeDrag.box[0] - (event.clientX - treeDrag.clientX) * scaleX;
      const rawY = treeDrag.box[1] - (event.clientY - treeDrag.clientY) * scaleY;
      const xPad = treeDrag.box[2] * 0.25, yPad = treeDrag.box[3] * 0.25;
      const minX = treeBaseViewBox[0] - xPad;
      const maxX = treeBaseViewBox[0] + treeBaseViewBox[2] - treeDrag.box[2] + xPad;
      const minY = treeBaseViewBox[1] - yPad;
      const maxY = treeBaseViewBox[1] + treeBaseViewBox[3] - treeDrag.box[3] + yPad;
      treeViewBox = [
        Math.min(maxX, Math.max(minX, rawX)),
        Math.min(maxY, Math.max(minY, rawY)),
        treeDrag.box[2],
        treeDrag.box[3]
      ];
      applyTreeViewBox();
    }});
    function finishTreeDrag(event) {{
      if (!treeDrag || (event && event.pointerId !== treeDrag.pointerId)) return;
      try {{ treeSvg.releasePointerCapture(treeDrag.pointerId); }} catch (_) {{}}
      treeDrag = null;
      treeSvg.classList.remove("dragging");
    }}
    treeSvg.addEventListener("pointerup", finishTreeDrag);
    treeSvg.addEventListener("pointercancel", finishTreeDrag);
    treeSvg.addEventListener("pointerleave", () => {{
      if (!treeDrag) document.getElementById("treeMagnifier").classList.remove("visible");
    }});

    document.getElementById("treeZoomOut").addEventListener("click", () => zoomTreeAtCenter(1.25));
    document.getElementById("treeZoomIn").addEventListener("click", () => zoomTreeAtCenter(0.8));
    document.getElementById("treeFitAll").addEventListener("click", resetTreeZoom);
    document.getElementById("treeFitPath").addEventListener("click", fitSelectedTreePath);
    document.getElementById("treeMagnifierToggle").addEventListener("click", event => {{
      treeMagnifierEnabled = !treeMagnifierEnabled;
      event.currentTarget.classList.toggle("active", treeMagnifierEnabled);
      if (!treeMagnifierEnabled) document.getElementById("treeMagnifier").classList.remove("visible");
    }});

    const mainLayout = document.getElementById("mainLayout");
    const splitter = document.getElementById("splitter");
    splitter.addEventListener("pointerdown", event => {{
      if (event.button !== 0) return;
      splitter.setPointerCapture(event.pointerId);
      splitter.classList.add("dragging");
      const rect = mainLayout.getBoundingClientRect();
      splitterDrag = {{
        pointerId:event.pointerId,
        left:rect.left,
        width:rect.width
      }};
    }});

    splitter.addEventListener("pointermove", event => {{
      if (!splitterDrag) return;
      const raw = (event.clientX - splitterDrag.left) / Math.max(1, splitterDrag.width) * 100;
      const pct = Math.min(84, Math.max(25, raw));
      mainLayout.style.setProperty("--left-pane", `${{pct.toFixed(1)}}%`);
    }});

    function finishSplitterDrag() {{
      if (!splitterDrag) return;
      try {{ splitter.releasePointerCapture(splitterDrag.pointerId); }} catch (_) {{}}
      splitter.classList.remove("dragging");
      splitterDrag = null;
    }}

    splitter.addEventListener("pointerup", finishSplitterDrag);
    splitter.addEventListener("pointercancel", finishSplitterDrag);
    splitter.addEventListener("dblclick", event => {{
      event.preventDefault();
      mainLayout.style.setProperty("--left-pane", "75%");
    }});
    renderAll();
  </script>
</body>
</html>"""
    p.write_text(html, encoding="utf-8")
    return p


def export_schedule_mat(
    result,
    path: str | Path,
    *,
    plans: Sequence,
    tmap: Optional[TrafficMap] = None,
) -> Path:
    p = _ensure_parent(path)
    # Lightweight placeholder export so MATLAB-side filename remains reproducible.
    rows = [
        "vehicle_id,task_index,resource,requested_time,start_time,end_time,delay",
        *[
            f"{seg.vehicle_id},{seg.task_index},{seg.resource},{seg.requested_time},"
            f"{seg.start_time},{seg.end_time},{seg.delay}"
            for seg in result.best_schedule
        ],
    ]
    p.with_suffix(".csv").write_text("\n".join(rows), encoding="utf-8")
    p.write_bytes(b"")
    return p
