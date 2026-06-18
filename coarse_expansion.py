"""Node expansion logic for the coarse centralized scheduler.

This file is the Python version of the old ``expand_array`` idea: given the
current decision-tree node, it computes the next significant moment and returns
all valid child nodes.  Keeping this logic separate makes it easier to inspect
the timing model ``d, r, o`` and the contention branches ``U_c -> U_temp``.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Sequence, Tuple

from coarse_scheduler import (
    BIG_M,
    EPS,
    AttemptSegment,
    CoarseNode,
    ScheduleSegment,
    VehiclePlan,
)


def expand_node(
    nodes: Sequence[CoarseNode],
    c_idx: int,
    plans: Sequence[VehiclePlan],
) -> Tuple[List[CoarseNode], bool]:
    """Expand one decision-tree node at its current significant moment."""

    c = nodes[c_idx]
    n_vehicle = len(plans)
    da = [math.inf for _ in plans]
    ra = [0.0 for _ in plans]
    oa = [0.0 for _ in plans]
    ni2 = list(c.ni)
    alpha = [list(row) for row in c.alpha]

    for n, plan in enumerate(plans):
        if ni2[n] >= plan.task_count and c.r[n] <= EPS:
            ni2[n] = plan.task_count
            da[n] = math.inf
            ra[n] = 0.0
            oa[n] = 0.0
            continue

        if c.d[n] <= EPS and c.ni[n] < plan.task_count:
            next_task = c.ni[n] + 1
            ti = next_task - 1
            ni2[n] = next_task
            da[n] = BIG_M
            ra[n] = plan.durations[ti]
            oa[n] = 0.0
            if math.isnan(alpha[n][ti]):
                alpha[n][ti] = c.tw
        else:
            da[n] = c.d[n]
            ra[n] = c.r[n]
            oa[n] = c.o[n]

    if all(ni2[n] >= plans[n].task_count and ra[n] <= EPS for n in range(n_vehicle)):
        return [], True

    U_c = _current_requests(ra, ni2, plans)
    active = [n for n in range(n_vehicle) if U_c[n] is not None]

    if not active:
        U_temp = tuple(None for _ in plans)
        d2, r2, o2, tw1 = _next_sig_m(c.tw, da, ra, oa, U_temp)
        child = _make_child(c, plans, d2, r2, o2, tw1, ni2, U_c, U_temp, ra, alpha)
        return [child], False

    children: List[CoarseNode] = []
    for U_temp in _valid_running_choices(c, U_c, ra, ni2, plans):
        ra_for_step = _reset_interrupted_repeat_tasks(c, U_c, U_temp, ra, ni2, plans)
        d2, r2, o2, tw1 = _next_sig_m(c.tw, da, ra_for_step, oa, U_temp)
        child = _make_child(c, plans, d2, r2, o2, tw1, ni2, U_c, U_temp, ra_for_step, alpha)
        children.append(child)

    unique: Dict[Tuple, CoarseNode] = {}
    for child in children:
        sig = (
            round(child.tw, 9),
            child.ni,
            tuple(round(x, 9) for x in child.d),
            tuple(round(x, 9) for x in child.r),
            child.U_temp,
            round(child.g, 9),
        )
        unique.setdefault(sig, child)
    return list(unique.values()), False


# Alias for the paper/old-code terminology.
expand_array = expand_node


def _current_requests(
    ra: Sequence[float],
    ni2: Sequence[int],
    plans: Sequence[VehiclePlan],
) -> Tuple[Optional[int], ...]:
    requests: List[Optional[int]] = []
    for n, plan in enumerate(plans):
        if ra[n] <= EPS or ni2[n] < 1:
            requests.append(None)
        else:
            requests.append(plan.resources[ni2[n] - 1])
    return tuple(requests)


def _valid_running_choices(
    c: CoarseNode,
    U_c: Tuple[Optional[int], ...],
    ra: Sequence[float],
    ni2: Sequence[int],
    plans: Sequence[VehiclePlan],
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
    products = itertools.product(*(item[1] for item in per_resource_options))
    for queues in products:
        U_temp: List[Optional[int]] = [None for _ in plans]
        for (resource, _options), queue in zip(per_resource_options, queues):
            if queue:
                U_temp[queue[0]] = resource
        choices.append(tuple(U_temp))

    return tuple(choices) if choices else (tuple(None for _ in plans),)


def _reset_interrupted_repeat_tasks(
    c: CoarseNode,
    U_c: Tuple[Optional[int], ...],
    U_temp: Tuple[Optional[int], ...],
    ra: Sequence[float],
    ni2: Sequence[int],
    plans: Sequence[VehiclePlan],
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
            out[n] = plans[n].durations[ni2[n] - 1]
    return tuple(out)


def _next_sig_m(
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


def _make_child(
    c: CoarseNode,
    plans: Sequence[VehiclePlan],
    d2: Tuple[float, ...],
    r2: Tuple[float, ...],
    o2: Tuple[float, ...],
    tw1: float,
    ni2: Sequence[int],
    U_c: Tuple[Optional[int], ...],
    U_temp: Tuple[Optional[int], ...],
    ra: Sequence[float],
    alpha_in: Sequence[Sequence[float]],
) -> CoarseNode:
    alpha = [list(row) for row in alpha_in]
    gamma = [list(row) for row in c.gamma]
    d_new = list(d2)
    g_new = c.g
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
            ti = ni2[n] - 1
            duration = plans[n].durations[ti]
            progress = max(0.0, duration - c.r[n])
            if progress > EPS:
                attempts.append(
                    AttemptSegment(
                        vehicle_id=plans[n].vehicle_id,
                        task_index=ni2[n],
                        resource=plans[n].resources[ti],
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
            gamma[n][ti] = tw1
            duration = plan.durations[ti]
            start_time = tw1 - duration
            requested_time = alpha[n][ti]
            delay = max(0.0, start_time - requested_time)
            g_new += delay
            segments.append(
                ScheduleSegment(
                    vehicle_id=plan.vehicle_id,
                    task_index=ni2[n],
                    resource=plan.resources[ti],
                    requested_time=requested_time,
                    start_time=start_time,
                    end_time=tw1,
                    delay=delay,
                )
            )
            if ni2[n] < plan.task_count:
                d_new[n] = plan.road_time
            else:
                d_new[n] = math.inf

    return CoarseNode(
        idx=-1,
        parent=c.idx,
        tw=tw1,
        d=tuple(d_new),
        r=r2,
        o=o2,
        ni=tuple(ni2),
        U_c=U_c,
        U_temp=U_temp,
        g=g_new,
        alpha=tuple(tuple(row) for row in alpha),
        gamma=tuple(tuple(row) for row in gamma),
        segments=tuple(segments),
        attempts=tuple(attempts),
        priority_queues=tuple(
            (resource, tuple(queue))
            for resource, queue in sorted(queue_by_resource.items())
            if queue
        ),
    )
