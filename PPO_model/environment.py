"""Decision-epoch environment backed by the exact co-design transition model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from coarse_scheduler import (
    expand_dynamic_codesign_node,
    make_dynamic_codesign_root,
)
from scheduler_models import RelaxedNode, RelaxedVehiclePlan
from trajectory_conflicts import set_trajectory_conflict_filter


@dataclass(frozen=True)
class DecisionStep:
    """Result of one PPO decision followed by all forced transitions."""

    node: RelaxedNode
    branches: Tuple[RelaxedNode, ...]
    reward: float
    terminated: bool
    forced_edges: int
    total_cost: float


class DecisionTreeEnv:
    """Expose only genuine branching nodes to PPO.

    The underlying transition model generates all feasible children without an
    incumbent-cost bound.  Nodes with one child are traversed automatically.
    One environment action selects one immediate legal child at the current
    decision node.
    """

    def __init__(
        self,
        plans: Sequence[RelaxedVehiclePlan],
        *,
        max_forced_edges: int = 100_000,
    ) -> None:
        if not plans:
            raise ValueError("DecisionTreeEnv requires at least one vehicle plan")
        self.plans = tuple(plans)
        self.max_forced_edges = int(max_forced_edges)
        if self.max_forced_edges <= 0:
            raise ValueError("max_forced_edges must be positive")

        # Initial PPO scope: one indivisible resource per intersection.
        set_trajectory_conflict_filter(False)

        self.current_node: RelaxedNode
        self.branches: Tuple[RelaxedNode, ...]
        self.terminated = False
        self.decision_count = 0
        self.edge_count = 0
        self.initial_forced_cost = 0.0
        self._trajectory: list[RelaxedNode] = []

    @property
    def trajectory(self) -> Tuple[RelaxedNode, ...]:
        return tuple(self._trajectory)

    def reset(self) -> tuple[RelaxedNode, Tuple[RelaxedNode, ...], bool]:
        root = make_dynamic_codesign_root(self.plans)
        node, branches, terminated, forced_edges = self._advance_forced(root)
        self.current_node = node
        self.branches = branches
        self.terminated = terminated
        self.decision_count = 0
        self.edge_count = forced_edges
        self.initial_forced_cost = node.g - root.g
        self._trajectory = [root]
        if node is not root:
            self._trajectory.append(node)
        return node, branches, terminated

    def step(self, action_index: int) -> DecisionStep:
        if self.terminated:
            raise RuntimeError("cannot step a terminated environment; call reset()")
        if not 0 <= action_index < len(self.branches):
            raise IndexError(
                f"action {action_index} outside [0, {len(self.branches)})"
            )

        start = self.current_node
        selected = self.branches[action_index]
        node, branches, terminated, forced_edges = self._advance_forced(selected)

        self.current_node = node
        self.branches = branches
        self.terminated = terminated
        self.decision_count += 1
        self.edge_count += 1 + forced_edges
        self._trajectory.append(selected)
        if node is not selected:
            self._trajectory.append(node)

        return DecisionStep(
            node=node,
            branches=branches,
            reward=-(node.g - start.g),
            terminated=terminated,
            forced_edges=forced_edges,
            total_cost=node.g,
        )

    def _advance_forced(
        self,
        start: RelaxedNode,
    ) -> tuple[RelaxedNode, Tuple[RelaxedNode, ...], bool, int]:
        node = start
        forced_edges = 0
        while True:
            children, is_leaf = expand_dynamic_codesign_node(node, self.plans)
            if is_leaf:
                return node, (), True, forced_edges
            if not children:
                raise RuntimeError("nonterminal co-design node has no children")

            ordered = tuple(sorted(children, key=self._branch_sort_key))
            self._assert_strict_resource_mutex(ordered)
            if len(ordered) >= 2:
                return node, ordered, False, forced_edges

            node = ordered[0]
            forced_edges += 1
            if forced_edges > self.max_forced_edges:
                raise RuntimeError("forced-transition chain exceeded safety limit")

    @staticmethod
    def _branch_sort_key(node: RelaxedNode) -> tuple:
        u_key = tuple(0 if resource is None else int(resource) for resource in node.U_temp)
        return (
            u_key,
            node.route_candidates,
            node.path_decisions,
            round(node.tw, 12),
            round(node.g, 12),
        )

    @staticmethod
    def _assert_strict_resource_mutex(branches: Sequence[RelaxedNode]) -> None:
        for child in branches:
            occupied = [resource for resource in child.U_temp if resource is not None]
            if len(occupied) != len(set(occupied)):
                raise RuntimeError(
                    "transition model generated multiple robots on one intersection"
                )
