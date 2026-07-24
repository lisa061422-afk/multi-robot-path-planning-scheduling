"""Independent fixed-shortest-path + FCFS scheduler.

This module is intentionally separate from existing scheduler files and exposes
one solver that returns the same public result shape (`RelaxedSearchResult`) as the
co-design solvers, for direct objective comparison.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from scheduler_models import RelaxedNode, RelaxedSearchResult, RelaxedVehiclePlan, ScheduleSegment

EPS = 1e-9


def _route_free_time(route, road_time: float) -> float:
    return sum(route.execution_times) + len(route.edges) * road_time


def _select_shortest_route(plan: RelaxedVehiclePlan) -> int:
    if not plan.route_options:
        raise ValueError(f"vehicle {plan.vehicle_id} has no route options")
    if len(plan.route_options) == 1:
        return 0

    best_cost = _route_free_time(plan.route_options[0], plan.road_time)
    best_idx = 0
    for idx in range(1, len(plan.route_options)):
        cost = _route_free_time(plan.route_options[idx], plan.road_time)
        if cost + 1e-12 < best_cost:
            best_cost = cost
            best_idx = idx
    return best_idx


def _snapshot_node(
    *,
    idx: int,
    parent: int,
    tw: float,
    g_delay: float,
    g_path: float,
    segments: Sequence[ScheduleSegment],
    d: Sequence[float],
    r: Sequence[float],
    o: Sequence[float],
    ni: Sequence[int],
    route_choices: Sequence[int],
    route_candidates: Sequence[Sequence[int]],
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
        route_candidates=tuple(route_candidates),
        U_temp=tuple(None for _ in d),
        ni=tuple(ni),
        d=tuple(d),
        r=tuple(r),
        o=tuple(o),
        alpha=tuple(() for _ in d),
        gamma=tuple(() for _ in d),
        priority_queues=(),
        path_decisions=(),
    )


def search_fixed_shortest_fcfs_dfs_bb(
    plans: Sequence[RelaxedVehiclePlan],
    *,
    verbose: bool = True,
) -> RelaxedSearchResult:
    """Deterministic FCFS over fixed shortest paths."""

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

    # Pick one fixed path per vehicle: shortest by free-flow time.
    fixed_choice = [_select_shortest_route(plan) for plan in plans]
    task_counts = [len(plans[i].route_options[fixed_choice[i]].intersections) for i in range(len(plans))]

    d = [float(plan.alpha0) for plan in plans]
    r = [0.0 for _ in plans]
    o = [0.0 for _ in plans]
    ni = [0 for _ in plans]
    run_start = [math.nan for _ in plans]
    running_by_resource: dict[int, int] = {}
    requested_time = [math.nan for _ in plans]
    segments: List[ScheduleSegment] = []

    # Record chosen path index per vehicle for all time for traceability.
    route_candidates = [tuple([choice]) for choice in fixed_choice]
    route_choices = [choice for choice in fixed_choice]

    # Priority queues by resource for FCFS tie-break: earliest request time then
    # smaller vehicle id.
    wait_queues: dict[int, List[int]] = {}

    def request_key(vid: int) -> Tuple[float, int]:
        if math.isfinite(requested_time[vid]):
            return (requested_time[vid], plans[vid].vehicle_id)
        return (math.inf, plans[vid].vehicle_id)

    nodes: List[RelaxedNode] = []
    log: List[str] = [("[FCFS_StaticShortest] start")]
    nodes.append(
            _snapshot_node(
                idx=0,
                parent=-1,
                tw=0.0,
                g_delay=0.0,
                g_path=0.0,
                segments=(),
                d=d,
                r=r,
                o=o,
                ni=ni,
                route_choices=route_choices,
            route_candidates=tuple(route_candidates),
        )
    )

    tw = 0.0
    g_delay = 0.0
    g_path = 0.0
    steps = 0

    # Snapshot of all state updates.
    while True:
        done = True
        for i, plan in enumerate(plans):
            task_count = task_counts[i]
            if ni[i] < task_count:
                done = False
            else:
                if r[i] > 0.0 or (math.isfinite(d[i]) and d[i] < math.inf / 2):
                    done = False
        if done:
            break

        steps += 1
        if steps > 2_000_000:
            raise RuntimeError("FCFS_StaticShortest step limit reached; simulation may be looping")

        # 1) vehicles that are requesting next task at this time enter queues
        for n, plan in enumerate(plans):
            if ni[n] >= task_counts[n]:
                d[n] = math.inf
                continue
            if d[n] > EPS or r[n] > EPS:
                continue

            current_plan = plan.route_options[fixed_choice[n]]
            if ni[n] >= len(current_plan.traversals):
                # Defensive: route length mismatch should be avoided, but avoid hard crash.
                d[n] = math.inf
                continue
            resource = current_plan.traversals[ni[n]].intersection
            queue = wait_queues.setdefault(resource, [])
            if n in queue:
                queue.remove(n)
            queue.append(n)
            if math.isnan(requested_time[n]):
                requested_time[n] = tw

        # 2) grant each free resource by FCFS order
        for resource in sorted(wait_queues.keys()):
            if running_by_resource.get(resource) is not None:
                continue

            queue = wait_queues.get(resource, [])
            if not queue:
                continue

            queue.sort(key=request_key)
            winner = queue.pop(0)
            if ni[winner] >= task_counts[winner] or r[winner] > EPS or d[winner] > EPS:
                wait_queues[resource] = queue
                continue

            current_plan = plans[winner].route_options[fixed_choice[winner]]
            if ni[winner] >= len(current_plan.traversals):
                continue

            request_resource = current_plan.traversals[ni[winner]].intersection
            if request_resource != resource:
                target = wait_queues.setdefault(request_resource, [])
                if winner not in target:
                    target.append(winner)
                wait_queues[resource] = queue
                continue

            # grant
            running_by_resource[resource] = winner
            run_start[winner] = tw
            r[winner] = current_plan.traversals[ni[winner]].execution_time
            wait_queues[resource] = queue

        # 3) advance next event
        dt_candidates: List[float] = []
        for val in r:
            if val > EPS and math.isfinite(val):
                dt_candidates.append(val)
        for val in d:
            if val > EPS and math.isfinite(val):
                dt_candidates.append(val)
        if not dt_candidates:
            raise RuntimeError("FCFS_StaticShortest reached a non-advancing state")

        dt = min(dt_candidates)
        tw_next = tw + dt

        # 4) update timers
        for n in range(len(plans)):
            if d[n] > 0.0 and math.isfinite(d[n]):
                d[n] = max(0.0, d[n] - dt)
            if r[n] > 0.0:
                r[n] = max(0.0, r[n] - dt)
            if r[n] > 0.0:
                o[n] += dt

        # 5) finalize completed crossings
        completed: List[int] = []
        for resource, winner in list(running_by_resource.items()):
            if r[winner] > 0.0:
                continue
            completed.append(winner)
            task_index = ni[winner]
            current_plan = plans[winner].route_options[fixed_choice[winner]]
            request_r = requested_time[winner]
            start_time = run_start[winner]
            end_time = tw_next
            delay = max(0.0, start_time - request_r) if not math.isnan(request_r) else 0.0
            g_delay += delay
            segments.append(
                ScheduleSegment(
                    vehicle_id=plans[winner].vehicle_id,
                    task_index=task_index + 1,
                    resource=resource,
                    requested_time=request_r,
                    start_time=start_time,
                    end_time=end_time,
                    delay=delay,
                )
            )
            if task_index < len(current_plan.traversals):
                if task_index >= len(current_plan.intersections):
                    raise RuntimeError(
                        "FCFS_StaticShortest path layout mismatch for completed traversal"
                    )
            if task_index < task_counts[winner]:
                ni[winner] += 1
            if ni[winner] < task_counts[winner]:
                d[winner] = plans[winner].road_time
            else:
                d[winner] = math.inf

            r[winner] = 0.0
            run_start[winner] = math.nan
            requested_time[winner] = math.nan
            running_by_resource.pop(resource)

        # completed vehicles should not remain in other queues
        if completed:
            for n in completed:
                for q_resource in list(wait_queues.keys()):
                    wait_queues[q_resource] = [m for m in wait_queues[q_resource] if m != n]

        tw = tw_next
        nodes.append(
            _snapshot_node(
                idx=len(nodes),
                parent=len(nodes) - 1,
                tw=tw,
                g_delay=g_delay,
                g_path=g_path,
                segments=segments,
                d=d,
                r=r,
                o=o,
                ni=ni,
                route_choices=route_choices,
                route_candidates=tuple(route_candidates),
            )
        )

        if verbose and steps % 100 == 0:
            log.append(
                f"[FCFS_StaticShortest] step={steps} t={tw:.6f} g_delay={g_delay:.6f} "
                f"running={len(running_by_resource)}"
            )

    best_idx = len(nodes) - 1 if nodes else -1
    best_g = g_delay + g_path
    if verbose:
        log.append(
            f"[FCFS_StaticShortest] done: expanded={steps}, nodes={len(nodes)}, "
            f"best_g={best_g:.6f}"
        )

    return RelaxedSearchResult(
        nodes=tuple(nodes),
        leaves=(best_idx,),
        best_idx=best_idx,
        best_g=best_g,
        log=tuple(log),
    )


__all__ = ["search_fixed_shortest_fcfs_dfs_bb"]
