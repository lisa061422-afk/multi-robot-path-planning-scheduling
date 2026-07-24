"""FCFS baseline solver for path-selection and scheduling co-design.

This file is intentionally separate from the exact search and keeps the same
public return object shape (`RelaxedSearchResult`) so it can be compared against
`search_dynamic_codesign_dfs_bb` without changing the existing setup.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from scheduler_models import (
    RelaxedNode,
    RelaxedSearchResult,
    RelaxedVehiclePlan,
    ScheduleSegment,
)
from traffic_map import RouteOption

EPS = 1e-9


def _free_time(route: RouteOption, road_time: float) -> float:
    return sum(route.execution_times) + len(route.edges) * road_time


def _dynamic_task_count(plan: RelaxedVehiclePlan, candidates: Sequence[int]) -> int:
    if not candidates:
        return 0
    return max(len(plan.route_options[idx].intersections) for idx in candidates)


def _path_candidates(
    plan: RelaxedVehiclePlan,
    candidates: Sequence[int],
    task_index0: int,
    prefix: Tuple[int, ...],
) -> Tuple[int, ...]:
    out: List[int] = []
    for option_index in candidates:
        option = plan.route_options[option_index]
        if option.intersections[:task_index0] == prefix:
            out.append(option_index)
    return tuple(out)


def _path_decision(
    plan: RelaxedVehiclePlan,
    option_index: int,
    task_index0: int,
    tw: float,
    extra: float,
) -> Optional[Tuple[int, int, int, int, float, float]]:
    option = plan.route_options[option_index]
    traversal = option.traversals[task_index0]
    if task_index0 + 1 >= len(option.intersections):
        return None
    return (
        plan.vehicle_id,
        option.id,
        traversal.intersection,
        option.intersections[task_index0 + 1],
        tw,
        extra,
    )


def _path_extra(plan: RelaxedVehiclePlan, before: Sequence[int], after: Sequence[int]) -> float:
    if not before or not after:
        return 0.0
    best_before = min(_free_time(plan.route_options[idx], plan.road_time) for idx in before)
    best_after = min(_free_time(plan.route_options[idx], plan.road_time) for idx in after)
    return max(0.0, best_after - best_before)


def _dynamic_task_info(
    plan: RelaxedVehiclePlan,
    option_index: int,
    task_index0: int,
) -> Tuple[int, float]:
    option = plan.route_options[option_index]
    traversal = option.traversals[task_index0]
    return traversal.intersection, traversal.execution_time


def _snapshot_node(
    *,
    idx: int,
    parent: int,
    tw: float,
    g_delay: float,
    g_path: float,
    d: Sequence[float],
    r: Sequence[float],
    o: Sequence[float],
    ni: Sequence[int],
    alpha: Sequence[Sequence[float]],
    gamma: Sequence[Sequence[float]],
    segments: Sequence[ScheduleSegment],
    route_choices: Sequence[int],
    route_candidates: Sequence[Sequence[int]],
    priority_queues: Sequence[Tuple[int, Sequence[int]]],
    path_decisions: Sequence[Tuple[int, int, int, int, float, float]],
    U_temp: Sequence[Optional[int]],
) -> RelaxedNode:
    return RelaxedNode(
        idx=idx,
        parent=parent,
        tw=tw,
        g=g_delay + g_path,
        g_delay=g_delay,
        g_path=g_path,
        segments=tuple(segments),
        route_choices=tuple(route_choices),
        attempts=(),
        route_candidates=tuple(tuple(row) for row in route_candidates),
        ni=tuple(ni),
        d=tuple(d),
        r=tuple(r),
        o=tuple(o),
        alpha=tuple(tuple(row) for row in alpha),
        gamma=tuple(tuple(row) for row in gamma),
        U_temp=tuple(U_temp),
        priority_queues=tuple((res, tuple(q)) for res, q in sorted(priority_queues)),
        path_decisions=tuple(path_decisions),
    )


def _all_done(
    plans: Sequence[RelaxedVehiclePlan],
    *,
    ni: Sequence[int],
    d: Sequence[float],
    r: Sequence[float],
    route_candidates: Sequence[Sequence[int]],
) -> bool:
    for i, plan in enumerate(plans):
        if ni[i] >= _dynamic_task_count(plan, route_candidates[i]):
            if r[i] > 0.0 or (math.isfinite(d[i]) and d[i] < math.inf / 2):
                return False
        else:
            return False
    return True


def _select_route(
    plan: RelaxedVehiclePlan,
    candidates: Sequence[int],
    task_index0: int,
    done_prefix: Tuple[int, ...],
) -> Tuple[int, Sequence[int], float]:
    feasible = _path_candidates(plan, candidates, task_index0, done_prefix)
    if not feasible:
        feasible = tuple(candidates)
    if not feasible:
        feasible = tuple(range(len(plan.route_options)))
    best_cost = min(_free_time(plan.route_options[idx], plan.road_time) for idx in feasible)
    selected = min(
        idx for idx in feasible
        if math.isclose(
            _free_time(plan.route_options[idx], plan.road_time),
            best_cost,
            rel_tol=0,
            abs_tol=1e-12,
        )
    )
    before = feasible
    after = (selected,)
    extra = _path_extra(plan, before, after)
    return selected, after, extra


def _is_requesting(
    plans: Sequence[RelaxedVehiclePlan],
    route_candidates: Sequence[Sequence[int]],
    ni: Sequence[int],
    d: Sequence[float],
    r: Sequence[float],
    n: int,
) -> bool:
    task_count = _dynamic_task_count(plans[n], route_candidates[n])
    return (
        ni[n] < task_count
        and d[n] <= EPS
        and r[n] <= EPS
    )


def search_dynamic_codesign_fcfs_dfs_bb(
    plans: Sequence[RelaxedVehiclePlan],
    *,
    verbose: bool = True,
) -> RelaxedSearchResult:
    """Deterministic FCFS co-design rollout.

    Priority on each intersection is fixed by first request arrival time. For equal
    arrival time, smaller vehicle id breaks ties. Priority is preserved while waiting.
    """

    if not plans:
        return RelaxedSearchResult(
            nodes=(
                RelaxedNode(
                    idx=0,
                    parent=-1,
                    tw=0.0,
                    g=0.0,
                    g_delay=0.0,
                    g_path=0.0,
                    segments=(),
                    route_choices=(),
                    attempts=(),
                    route_candidates=(),
                    U_temp=(),
                    ni=(),
                    d=(),
                    r=(),
                    o=(),
                    alpha=(),
                    gamma=(),
                    priority_queues=(),
                    path_decisions=(),
                ),
            ),
            leaves=(0,),
            best_idx=0,
            best_g=0.0,
            log=("empty input; returning empty solution",),
        )

    n_vehicle = len(plans)
    max_tasks = [
        max((len(option.intersections) for option in plan.route_options), default=0)
        for plan in plans
    ]
    alpha = [[math.nan for _ in range(count)] for count in max_tasks]
    gamma = [[math.nan for _ in range(count)] for count in max_tasks]

    d = [float(plan.alpha0) for plan in plans]
    r = [0.0 for _ in plans]
    o = [0.0 for _ in plans]
    ni = [0 for _ in plans]
    route_candidates: List[Tuple[int, ...]] = [
        tuple(range(len(plan.route_options))) for plan in plans
    ]
    route_choices: List[int] = [-1 for _ in plans]
    done_prefix: List[Tuple[int, ...]] = [tuple() for _ in plans]
    path_decisions: List[Tuple[int, int, int, int, float, float]] = []

    g_delay = 0.0
    g_path = 0.0
    tw = 0.0

    running_by_resource: dict[int, int] = {}
    run_start = [math.nan for _ in plans]
    wait_queues: dict[int, List[int]] = {}

    segments: List[ScheduleSegment] = []
    U_temp: List[Optional[int]] = [None for _ in plans]
    nodes: List[RelaxedNode] = []
    log: List[str] = ["[FCFS_CoDesign] start"]

    nodes.append(
        _snapshot_node(
            idx=0,
            parent=-1,
            tw=tw,
            g_delay=0.0,
            g_path=0.0,
            d=d,
            r=r,
            o=o,
            ni=ni,
            alpha=alpha,
            gamma=gamma,
            segments=segments,
        route_choices=route_choices,
        route_candidates=route_candidates,
        U_temp=U_temp,
        priority_queues=(),
        path_decisions=path_decisions,
    )
    )

    def request_key(vehicle_idx: int) -> Tuple[float, int]:
        requested = alpha[vehicle_idx][ni[vehicle_idx]]
        if math.isnan(requested):
            return (math.inf, plans[vehicle_idx].vehicle_id)
        return (requested, plans[vehicle_idx].vehicle_id)

    steps = 0
    while not _all_done(
        plans,
        ni=ni,
        d=d,
        r=r,
        route_candidates=route_candidates,
    ):
        steps += 1
        if steps > 2_000_000:
            raise RuntimeError("FCFS_CoDesign step limit reached; simulation may be looping")

        # 1) current request arrivals and deterministic route choice per request
        for n, plan in enumerate(plans):
            task_count = _dynamic_task_count(plan, route_candidates[n])
            if ni[n] >= task_count:
                d[n] = math.inf
                continue
            if d[n] > EPS or r[n] > EPS:
                continue

            if math.isnan(alpha[n][ni[n]]):
                alpha[n][ni[n]] = tw

            if route_choices[n] < 0:
                selected, after, extra = _select_route(
                    plan,
                    route_candidates[n],
                    ni[n],
                    done_prefix[n],
                )
                g_path += extra
                route_choices[n] = selected
                route_candidates[n] = after

                decision = _path_decision(plan, selected, ni[n], tw, extra)
                if decision is not None:
                    path_decisions.append(decision)

            selected = route_choices[n]
            if selected < 0:
                selected, after, extra = _select_route(
                    plan,
                    route_candidates[n],
                    ni[n],
                    done_prefix[n],
                )
                route_choices[n] = selected
                route_candidates[n] = after
                g_path += extra
                decision = _path_decision(plan, selected, ni[n], tw, extra)
                if decision is not None:
                    path_decisions.append(decision)

            resource, _ = _dynamic_task_info(plan, route_choices[n], ni[n])
            if U_temp[n] is None:
                U_temp[n] = resource
            queue = wait_queues.setdefault(resource, [])
            if n in queue:
                queue.remove(n)
            queue.append(n)

        # 2) grant one vehicle per resource by FCFS order
        for resource in sorted(wait_queues.keys()):
            if running_by_resource.get(resource) is not None:
                continue

            queue = wait_queues.get(resource, [])
            queue = [
                item
                for item in queue
                if _is_requesting(plans, route_candidates, ni, d, r, item)
            ]
            if not queue:
                wait_queues[resource] = queue
                continue

            queue.sort(key=request_key)
            winner = queue.pop(0)
            task_count = _dynamic_task_count(plans[winner], route_candidates[winner])
            if ni[winner] >= task_count:
                wait_queues[resource] = queue
                continue

            if route_choices[winner] < 0:
                selected_candidates = route_candidates[winner]
                if not selected_candidates:
                    selected_candidates = tuple(range(len(plans[winner].route_options)))
                selected = selected_candidates[0]
                route_choices[winner] = selected
                route_candidates[winner] = (selected,)

            request_resource, duration = _dynamic_task_info(
                plans[winner],
                route_choices[winner],
                ni[winner],
            )
            if request_resource != resource:
                # stale/misaligned request entry; re-queue by true resource.
                U_temp[winner] = request_resource
                route_candidates[winner] = (
                    route_choices[winner],
                ) if route_choices[winner] >= 0 else route_candidates[winner]
                target_queue = wait_queues.setdefault(request_resource, [])
                if winner not in target_queue:
                    target_queue.append(winner)
                wait_queues[resource] = queue
                continue

            running_by_resource[resource] = winner
            run_start[winner] = tw
            r[winner] = duration
            wait_queues[resource] = queue

        # 3) next event time
        dt_candidates = []
        for value in r:
            if value > EPS and math.isfinite(value):
                dt_candidates.append(value)
        for value in d:
            if value > EPS and math.isfinite(value):
                dt_candidates.append(value)

        if not dt_candidates:
            raise RuntimeError("FCFS_CoDesign reached a non-advancing state")

        dt = min(dt_candidates)
        tw_next = tw + dt

        # 4) advance system clock
        for n in range(n_vehicle):
            if d[n] > 0.0 and math.isfinite(d[n]):
                d[n] = max(0.0, d[n] - dt)
            if r[n] > 0.0:
                r[n] = max(0.0, r[n] - dt)
            if r[n] > 0.0:
                o[n] += dt

        # 5) finalize completed tasks
        completed: List[int] = []
        for resource, winner in list(running_by_resource.items()):
            if r[winner] > 0.0:
                continue
            completed.append(winner)
            task_index0 = ni[winner]
            requested_time = alpha[winner][task_index0]
            start_time = run_start[winner]
            end_time = tw_next
            delay = max(0.0, start_time - requested_time) if not math.isnan(requested_time) else 0.0

            g_delay += delay
            segments.append(
                ScheduleSegment(
                    vehicle_id=plans[winner].vehicle_id,
                    task_index=task_index0 + 1,
                    resource=resource,
                    requested_time=requested_time,
                    start_time=start_time,
                    end_time=end_time,
                    delay=delay,
                )
            )
            gamma[winner][task_index0] = end_time
            ni[winner] += 1
            done_prefix[winner] = done_prefix[winner] + (resource,)
            U_temp[winner] = None

            task_count = _dynamic_task_count(plans[winner], route_candidates[winner])
            if ni[winner] < task_count:
                d[winner] = plans[winner].road_time
            else:
                d[winner] = math.inf

            r[winner] = 0.0
            run_start[winner] = math.nan
            route_choices[winner] = -1
            running_by_resource.pop(resource)

        for n in completed:
            for queue_resource in list(wait_queues.keys()):
                wait_queues[queue_resource] = [m for m in wait_queues[queue_resource] if m != n]

        tw = tw_next

        # 6) keep node/state record for tracing
        queue_state: List[Tuple[int, Tuple[int, ...]]] = []
        for resource, queue in sorted(wait_queues.items()):
            if queue:
                queue_state.append((resource, tuple(queue)))

        nodes.append(
            _snapshot_node(
                idx=len(nodes),
                parent=len(nodes) - 1,
                tw=tw,
                g_delay=g_delay,
                g_path=g_path,
                d=d,
                r=r,
                o=o,
                ni=ni,
                alpha=alpha,
                gamma=gamma,
                segments=segments,
                route_choices=route_choices,
                route_candidates=route_candidates,
                U_temp=U_temp,
                priority_queues=queue_state,
                path_decisions=path_decisions,
            )
        )

        if verbose and steps % 50 == 0:
            log.append(
                f"[FCFS_CoDesign] step={steps} t={tw:.6f} g_delay={g_delay:.6f} "
                f"g_path={g_path:.6f} running={len(running_by_resource)}"
            )

    best_idx = len(nodes) - 1 if nodes else -1
    best_g = g_delay + g_path

    if verbose:
        log.append(
            f"[FCFS_CoDesign] done: expanded={steps}, nodes={len(nodes)}, "
            f"best_g={best_g:.6f}"
        )

    return RelaxedSearchResult(
        nodes=tuple(nodes),
        leaves=(best_idx,),
        best_idx=best_idx,
        best_g=best_g,
        log=tuple(log),
    )
