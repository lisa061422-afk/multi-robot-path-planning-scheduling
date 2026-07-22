"""Three-by-three, three-robot training-case generation."""

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
    """Create fixed-N=3 cases for the strict one-resource-per-intersection model."""

    FIXED_REQUESTS: Tuple[Tuple[int, int, int, float], ...] = (
        (1, 3, 6, 0.0),
        (2, 4, 8, 0.0),
        (3, 12, 5, 0.0),
    )

    def __init__(
        self,
        *,
        seed: int = 20260721,
        randomize: bool = True,
        intersection_time_scale: float = 2.0,
        road_time: float = 2.0,
        entrance_headway: float = 2.0,
        max_initial_release: float = 2.0,
    ) -> None:
        self.rng = random.Random(seed)
        self.randomize = bool(randomize)
        self.road_time = float(road_time)
        self.entrance_headway = float(entrance_headway)
        self.max_initial_release = float(max_initial_release)
        self.traffic_map = TrafficMap.paper_3x3(
            intersection_time_scale=intersection_time_scale
        )
        set_trajectory_conflict_filter(False)
        self.case_index = 0

    def __call__(self) -> TrainingCase:
        self.case_index += 1
        requests = self._random_requests() if self.randomize else self.FIXED_REQUESTS
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
        ports = tuple(self.traffic_map.port_ids)
        releases = [
            0.5 * round(self.rng.uniform(0.0, self.max_initial_release) / 0.5)
            for _ in range(3)
        ]
        rows = []
        for vehicle_id in range(1, 4):
            entrance, exit_port = self.rng.sample(ports, 2)
            rows.append((vehicle_id, entrance, exit_port, releases[vehicle_id - 1]))
        return tuple(rows)
