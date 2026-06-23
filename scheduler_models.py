"""Shared scheduler data models.

The RL environment should be able to depend on these stable request/result
shapes without knowing which resource model produced the schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from traffic_map import PortId, RouteOption


EPS = 1e-9
BIG_M = 1000.0


@dataclass(frozen=True)
class VehiclePlan:
    """One vehicle with one fixed route."""

    vehicle_id: int
    entrance: PortId
    exit: PortId
    route: RouteOption
    alpha0: float = 0.0
    road_time: float = 3.0

    @property
    def resources(self) -> Tuple[int, ...]:
        return self.route.resource_sequence

    @property
    def durations(self) -> Tuple[float, ...]:
        return self.route.execution_times

    @property
    def task_count(self) -> int:
        return len(self.resources)

    @property
    def free_flow_time(self) -> float:
        return sum(self.durations) + len(self.route.edges) * self.road_time


@dataclass(frozen=True)
class ScheduleSegment:
    """One completed resource occupation in the selected schedule."""

    vehicle_id: int
    task_index: int
    resource: int
    requested_time: float
    start_time: float
    end_time: float
    delay: float


@dataclass(frozen=True)
class AttemptSegment:
    """One interrupted execution attempt that must be repeated from scratch."""

    vehicle_id: int
    task_index: int
    resource: int
    start_time: float
    end_time: float


@dataclass
class CoarseNode:
    idx: int
    parent: int
    tw: float
    d: Tuple[float, ...]
    r: Tuple[float, ...]
    o: Tuple[float, ...]
    ni: Tuple[int, ...]
    U_c: Tuple[Optional[int], ...]
    U_temp: Tuple[Optional[int], ...]
    g: float
    alpha: Tuple[Tuple[float, ...], ...]
    gamma: Tuple[Tuple[float, ...], ...]
    segments: Tuple[ScheduleSegment, ...]
    attempts: Tuple[AttemptSegment, ...] = ()
    priority_queues: Tuple[Tuple[int, Tuple[int, ...]], ...] = ()


@dataclass(frozen=True)
class SearchResult:
    nodes: Tuple[CoarseNode, ...]
    leaves: Tuple[int, ...]
    best_idx: int
    best_g: float
    log: Tuple[str, ...]

    @property
    def best_node(self) -> CoarseNode:
        if self.best_idx < 0:
            raise ValueError("search did not find a feasible leaf")
        return self.nodes[self.best_idx]

    @property
    def best_schedule(self) -> Tuple[ScheduleSegment, ...]:
        return self.best_node.segments


@dataclass(frozen=True)
class RelaxedVehiclePlan:
    vehicle_id: int
    entrance: PortId
    exit: PortId
    route_options: Tuple[RouteOption, ...]
    alpha0: float = 0.0
    road_time: float = 3.0


@dataclass(frozen=True)
class RelaxedNode:
    idx: int
    parent: int
    tw: float
    g: float
    g_delay: float
    g_path: float
    segments: Tuple[ScheduleSegment, ...]
    route_choices: Tuple[int, ...]
    attempts: Tuple[AttemptSegment, ...] = ()
    route_candidates: Tuple[Tuple[int, ...], ...] = ()
    U_temp: Tuple[Optional[int], ...] = ()
    ni: Tuple[int, ...] = ()
    d: Tuple[float, ...] = ()
    r: Tuple[float, ...] = ()
    o: Tuple[float, ...] = ()
    alpha: Tuple[Tuple[float, ...], ...] = ()
    gamma: Tuple[Tuple[float, ...], ...] = ()
    priority_queues: Tuple[Tuple[int, Tuple[int, ...]], ...] = ()
    path_decisions: Tuple[Tuple[int, int, int, int, float, float], ...] = ()


@dataclass(frozen=True)
class RelaxedSearchResult:
    nodes: Tuple[RelaxedNode, ...]
    leaves: Tuple[int, ...]
    best_idx: int
    best_g: float
    log: Tuple[str, ...]

    @property
    def best_node(self) -> RelaxedNode:
        return self.nodes[self.best_idx]

    @property
    def best_schedule(self) -> Tuple[ScheduleSegment, ...]:
        return self.best_node.segments
