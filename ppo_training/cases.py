"""Three-by-three training-case generation with configurable robot count."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Tuple

from coarse_scheduler import apply_relaxed_entrance_headway, build_relaxed_vehicle_plan
from scheduler_models import RelaxedVehiclePlan
from traffic_map import TrafficMap
from trajectory_conflicts import set_trajectory_conflict_filter


@dataclass(frozen=True)
class TrainingCase:
    name: str
    traffic_map: TrafficMap
    requests: Tuple[Tuple[int, int, int, float], ...]
    plans: Tuple[RelaxedVehiclePlan, ...]


class ThreeByThreeCaseFactory:
    """Create fixed-3x3 map cases for the strict one-resource-per-intersection model."""

    FIXED_REQUESTS_3: Tuple[Tuple[int, int, int, float], ...] = (
        (1, 3, 6, 0.0),
        (2, 4, 8, 0.0),
        (3, 12, 5, 0.0),
    )

    def __init__(
        self,
        *,
        seed: int = 20260721,
        n_robots: int = 3,
        randomize: bool = True,
        intersection_time_scale: float = 2.0,
        road_time: float = 2.0,
        entrance_headway: float = 2.0,
        max_initial_release: float = 2.0,
        min_initial_release: float = 0.0,
        initial_release_step: float = 0.5,
        fix_shortest_paths: bool = False,
    ) -> None:
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.randomize = bool(randomize)
        self.n_robots = int(n_robots)
        if self.n_robots <= 0:
            raise ValueError("n_robots must be positive")
        self.road_time = float(road_time)
        self.entrance_headway = float(entrance_headway)
        self.min_initial_release = float(min_initial_release)
        self.max_initial_release = float(max_initial_release)
        self.initial_release_step = float(initial_release_step)
        if self.initial_release_step <= 0:
            raise ValueError("initial_release_step must be positive")
        if self.min_initial_release > self.max_initial_release:
            raise ValueError("min_initial_release must not exceed max_initial_release")
        self.traffic_map = TrafficMap.paper_3x3(
            intersection_time_scale=intersection_time_scale
        )
        self.fix_shortest_paths = bool(fix_shortest_paths)
        set_trajectory_conflict_filter(False)
        self.case_index = 0
        self.ports = tuple(self.traffic_map.port_ids)
        if self.randomize:
            self._fixed_requests = None
        else:
            self._fixed_requests = self._build_fixed_requests()

    def __call__(self) -> TrainingCase:
        self.case_index += 1
        requests = self._random_requests() if self.randomize else self._fixed_requests
        plans = tuple(
            build_relaxed_vehicle_plan(
                self.traffic_map,
                vehicle_id=vehicle_id,
                entrance=entrance,
                exit=exit_port,
                alpha0=alpha0,
                road_time=self.road_time,
            )
            for vehicle_id, entrance, exit_port, alpha0 in requests
        )
        if self.fix_shortest_paths:
            plans = tuple(
                _single_shortest_route_plan(plan, road_time=self.road_time)
                for plan in plans
            )
        plans = tuple(
            apply_relaxed_entrance_headway(
                plans,
                headway=self.entrance_headway,
            )
        )
        return TrainingCase(
            name=f"paper_3x3_n3_{self.case_index:06d}",
            traffic_map=self.traffic_map,
            requests=tuple(requests),
            plans=plans,
        )

    def _random_requests(self) -> Tuple[Tuple[int, int, int, float], ...]:
        ports = self.ports
        releases = [
            self.initial_release_step
            * round(
                self.rng.uniform(self.min_initial_release, self.max_initial_release)
                / self.initial_release_step
            )
            for _ in range(self.n_robots)
        ]
        rows = []
        for vehicle_id in range(1, self.n_robots + 1):
            entrance, exit_port = self.rng.sample(ports, 2)
            rows.append((vehicle_id, entrance, exit_port, releases[vehicle_id - 1]))
        return tuple(rows)

    def _build_fixed_requests(self) -> Tuple[Tuple[int, int, int, float], ...]:
        if self.n_robots == 3:
            return self.FIXED_REQUESTS_3
        rng = random.Random(self.seed)
        requests = []
        for vehicle_id in range(1, self.n_robots + 1):
            entrance, exit_port = rng.sample(self.ports, 2)
            initial_release = self.initial_release_step * round(
                self.min_initial_release / self.initial_release_step
            )
            requests.append((vehicle_id, entrance, exit_port, initial_release))
        return tuple(requests)


def _single_shortest_route_plan(
    plan: RelaxedVehiclePlan,
    *,
    road_time: float | None = None,
) -> RelaxedVehiclePlan:
    if not plan.route_options:
        return plan
    road_time = float(plan.road_time if road_time is None else road_time)
    shortest_idx = min(
        range(len(plan.route_options)),
        key=lambda i: (
            len(plan.route_options[i].intersections),
            sum(plan.route_options[i].execution_times)
            + len(plan.route_options[i].edges) * road_time,
            plan.route_options[i].id,
        ),
    )
    return RelaxedVehiclePlan(
        vehicle_id=plan.vehicle_id,
        entrance=plan.entrance,
        exit=plan.exit,
        route_options=(plan.route_options[shortest_idx],),
        alpha0=plan.alpha0,
        road_time=road_time,
    )
