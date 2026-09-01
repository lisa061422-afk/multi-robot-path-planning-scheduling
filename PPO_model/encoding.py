"""Fixed-size state and variable-branch encodings for the 3x3 PPO model."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from scheduler_models import EPS, RelaxedNode, RelaxedVehiclePlan


ROBOT_STATE_SCALARS = 12
ROBOT_ACTION_SCALARS = 4


@dataclass(frozen=True)
class EncodingConfig:
    n_robots: int = 3
    n_resources: int = 9
    n_ports: int = 12
    max_route_options: int = 12
    time_scale: float = 10.0
    horizon_scale: float = 100.0

    # Compatibility properties: historical callers use these on EncodingConfig directly.
    @property
    def state_dim(self) -> int:
        per_robot = (
            ROBOT_STATE_SCALARS
            + 2 * (self.n_resources + 1)
            + 3 * self.n_resources
            + 2 * self.n_ports
            + self.max_route_options
        )
        return 1 + self.n_robots * per_robot

    @property
    def action_dim(self) -> int:
        per_robot = (
            (self.n_resources + 1)
            + 2 * (self.n_resources + 1)
            + ROBOT_ACTION_SCALARS
        )
        return self.n_robots * per_robot


class BranchEncoder:
    """Encode one node and each of its legal outgoing branches."""

    ROBOT_STATE_SCALARS = ROBOT_STATE_SCALARS
    ROBOT_ACTION_SCALARS = ROBOT_ACTION_SCALARS

    def __init__(
        self,
        plans: Sequence[RelaxedVehiclePlan],
        config: EncodingConfig = EncodingConfig(),
    ) -> None:
        self.plans = tuple(plans)
        self.config = config
        if len(self.plans) != config.n_robots:
            raise ValueError(
                f"expected {config.n_robots} plans, received {len(self.plans)}"
            )
        self._vehicle_index = {
            plan.vehicle_id: index for index, plan in enumerate(self.plans)
        }
        if len(self._vehicle_index) != len(self.plans):
            raise ValueError("vehicle IDs must be unique")

    @property
    def state_dim(self) -> int:
        cfg = self.config
        per_robot = (
            self.ROBOT_STATE_SCALARS
            + 2 * (cfg.n_resources + 1)
            + 3 * cfg.n_resources
            + 2 * cfg.n_ports
            + cfg.max_route_options
        )
        return 1 + cfg.n_robots * per_robot

    @property
    def action_dim(self) -> int:
        cfg = self.config
        per_robot = (
            (cfg.n_resources + 1)
            + 2 * (cfg.n_resources + 1)
            + self.ROBOT_ACTION_SCALARS
        )
        return cfg.n_robots * per_robot

    def encode_state(self, node: RelaxedNode) -> torch.Tensor:
        cfg = self.config
        values: list[float] = [self._clip(node.tw / cfg.horizon_scale)]

        for n, plan in enumerate(self.plans):
            candidates = node.route_candidates[n]
            task_count = self._task_count(plan, candidates)
            current_task = node.ni[n] - 1
            current_duration = self._current_duration(node, n, plan, candidates)
            done = node.ni[n] >= task_count and node.r[n] <= EPS
            active = node.r[n] > EPS
            d_finite = math.isfinite(node.d[n])
            free_times = [self._free_time(plan, index) for index in candidates]
            min_free = min(free_times, default=0.0)
            spread = max(free_times, default=min_free) - min_free
            alpha = self._event_time(node.alpha, n, current_task)
            gamma = self._event_time(node.gamma, n, current_task)

            d_norm = (
                self._clip(node.d[n] / cfg.time_scale)
                if d_finite
                else 0.0
            )
            r_fraction = (
                self._clip(node.r[n] / current_duration)
                if current_duration > EPS
                else 0.0
            )
            scalars = [
                d_norm,
                r_fraction,
                self._clip(node.o[n] / cfg.time_scale),
                node.ni[n] / max(1, task_count),
                self._clip(current_duration / cfg.time_scale),
                len(candidates) / max(1, len(plan.route_options)),
                self._clip(min_free / cfg.horizon_scale),
                self._clip(spread / cfg.horizon_scale),
                float(active),
                float(done),
                self._clip(alpha / cfg.horizon_scale),
                self._clip(gamma / cfg.horizon_scale),
            ]
            values.extend(scalars)

            requested = self._requested_resource(node, n, plan, candidates)
            values.extend(self._resource_one_hot(requested))
            running_resource = node.U_temp[n] if active else None
            values.extend(self._resource_one_hot(running_resource))

            current_mask, upcoming_mask, next_mask = self._route_masks(
                node,
                n,
                plan,
                candidates,
            )
            values.extend(current_mask)
            values.extend(upcoming_mask)
            values.extend(next_mask)
            values.extend(self._port_one_hot(plan.entrance))
            values.extend(self._port_one_hot(plan.exit))
            values.extend(self._route_candidate_mask(plan, candidates))

        out = torch.tensor(values, dtype=torch.float32)
        if out.numel() != self.state_dim:
            raise RuntimeError(f"state encoder produced {out.numel()} != {self.state_dim}")
        return out

    def encode_actions(
        self,
        parent: RelaxedNode,
        branches: Sequence[RelaxedNode],
    ) -> torch.Tensor:
        if not branches:
            return torch.empty((0, self.action_dim), dtype=torch.float32)
        rows = [self._encode_action(parent, child) for child in branches]
        return torch.stack(rows)

    def _encode_action(self, parent: RelaxedNode, child: RelaxedNode) -> torch.Tensor:
        cfg = self.config
        new_path_decisions = child.path_decisions[len(parent.path_decisions) :]
        decisions_by_vehicle = {
            decision[0]: decision for decision in new_path_decisions
        }
        values: list[float] = []

        for n, plan in enumerate(self.plans):
            selected_resource = child.U_temp[n]
            values.extend(self._resource_one_hot(selected_resource))

            decision = decisions_by_vehicle.get(plan.vehicle_id)
            if decision is None:
                from_i = None
                next_i = None
                path_extra = 0.0
                path_active = 0.0
            else:
                from_i = int(decision[2])
                next_i = int(decision[3])
                path_extra = float(decision[5])
                path_active = 1.0
            values.extend(self._resource_one_hot(from_i))
            values.extend(self._resource_one_hot(next_i))

            previous_resource = parent.U_temp[n]
            was_running = previous_resource is not None and parent.r[n] > EPS
            continues = was_running and selected_resource == previous_resource
            interrupts = was_running and selected_resource != previous_resource
            values.extend(
                [
                    self._clip(path_extra / cfg.time_scale),
                    path_active,
                    float(continues),
                    float(interrupts),
                ]
            )

        out = torch.tensor(values, dtype=torch.float32)
        if out.numel() != self.action_dim:
            raise RuntimeError(f"action encoder produced {out.numel()} != {self.action_dim}")
        return out

    def _requested_resource(
        self,
        node: RelaxedNode,
        n: int,
        plan: RelaxedVehiclePlan,
        candidates: Sequence[int],
    ) -> int | None:
        if node.r[n] <= EPS or node.ni[n] < 1 or not candidates:
            return None
        task_index0 = node.ni[n] - 1
        option = plan.route_options[candidates[0]]
        if task_index0 >= len(option.intersections):
            return None
        return int(option.intersections[task_index0])

    def _current_duration(
        self,
        node: RelaxedNode,
        n: int,
        plan: RelaxedVehiclePlan,
        candidates: Sequence[int],
    ) -> float:
        if node.ni[n] < 1 or not candidates:
            return 0.0
        task_index0 = node.ni[n] - 1
        option = plan.route_options[candidates[0]]
        if task_index0 >= len(option.execution_times):
            return 0.0
        return float(option.execution_times[task_index0])

    def _route_masks(
        self,
        node: RelaxedNode,
        n: int,
        plan: RelaxedVehiclePlan,
        candidates: Sequence[int],
    ) -> tuple[list[float], list[float], list[float]]:
        cfg = self.config
        current = [0.0] * cfg.n_resources
        upcoming = [0.0] * cfg.n_resources
        next_after = [0.0] * cfg.n_resources

        current_index0 = node.ni[n] - 1
        upcoming_index0 = node.ni[n]
        for option_index in candidates:
            option = plan.route_options[option_index]
            if 0 <= current_index0 < len(option.intersections):
                current[option.intersections[current_index0] - 1] = 1.0
            if 0 <= upcoming_index0 < len(option.intersections):
                upcoming[option.intersections[upcoming_index0] - 1] = 1.0
            if 0 <= upcoming_index0 + 1 < len(option.intersections):
                next_after[option.intersections[upcoming_index0 + 1] - 1] = 1.0
        return current, upcoming, next_after

    def _resource_one_hot(self, resource: int | None) -> list[float]:
        cfg = self.config
        index = 0 if resource is None else int(resource)
        if not 0 <= index <= cfg.n_resources:
            raise ValueError(f"resource {resource} outside 1..{cfg.n_resources}")
        values = [0.0] * (cfg.n_resources + 1)
        values[index] = 1.0
        return values

    def _port_one_hot(self, port: int) -> list[float]:
        cfg = self.config
        if not 1 <= int(port) <= cfg.n_ports:
            raise ValueError(f"port {port} outside 1..{cfg.n_ports}")
        values = [0.0] * cfg.n_ports
        values[int(port) - 1] = 1.0
        return values

    def _route_candidate_mask(
        self,
        plan: RelaxedVehiclePlan,
        candidates: Sequence[int],
    ) -> list[float]:
        limit = self.config.max_route_options
        if len(plan.route_options) > limit:
            raise ValueError(
                f"OD P{plan.entrance}->P{plan.exit} has {len(plan.route_options)} "
                f"route options, exceeding configured limit {limit}"
            )
        values = [0.0] * limit
        for option_index in candidates:
            values[int(option_index)] = 1.0
        return values

    @staticmethod
    def _task_count(plan: RelaxedVehiclePlan, candidates: Sequence[int]) -> int:
        return max(
            (len(plan.route_options[index].intersections) for index in candidates),
            default=0,
        )

    @staticmethod
    def _event_time(
        table: Sequence[Sequence[float] | None],
        n: int,
        task_index0: int,
    ) -> float:
        if n >= len(table) or task_index0 < 0:
            return 0.0
        row = table[n]
        if row is None or task_index0 >= len(row):
            return 0.0
        value = float(row[task_index0])
        return value if math.isfinite(value) else 0.0

    @staticmethod
    def _free_time(plan: RelaxedVehiclePlan, option_index: int) -> float:
        option = plan.route_options[option_index]
        return sum(option.execution_times) + len(option.edges) * plan.road_time

    @staticmethod
    def _clip(value: float, limit: float = 10.0) -> float:
        return max(-limit, min(limit, float(value)))
