"""Scheduler facades for swappable resource models.

RL code should depend on this module's small interface instead of importing the
large implementation modules directly.  The current optimal baseline is the
coarse one-resource-per-intersection scheduler; the five-space scheduler will
fill the same shape later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

from scheduler_models import RelaxedSearchResult, RelaxedVehiclePlan, SearchResult, VehiclePlan


class FixedPathScheduler(Protocol):
    """Resource-model interface for fixed-route optimal scheduling."""

    name: str

    def schedule_fixed(
        self,
        plans: Sequence[VehiclePlan],
        *,
        deadline: Optional[float] = None,
        max_nodes: Optional[int] = None,
        branch_and_bound: bool = True,
        verbose: bool = True,
    ) -> SearchResult:
        """Return an optimal fixed-route schedule for the model."""


class PathSelectionScheduler(Protocol):
    """Resource-model interface for joint path selection and scheduling."""

    name: str

    def schedule_path_selection(
        self,
        plans: Sequence[RelaxedVehiclePlan],
        *,
        lambda_path: float = 1.0,
        branch_and_bound: bool = True,
        verbose: bool = True,
    ) -> RelaxedSearchResult:
        """Return an optimal co-design schedule for the model."""


@dataclass(frozen=True)
class CoarseIntersectionScheduler:
    """Optimal baseline where each intersection is one conservative resource."""

    name: str = "coarse_intersection"

    def schedule_fixed(
        self,
        plans: Sequence[VehiclePlan],
        *,
        deadline: Optional[float] = None,
        max_nodes: Optional[int] = None,
        branch_and_bound: bool = True,
        verbose: bool = True,
    ) -> SearchResult:
        from coarse_scheduler import search_dfs_bb

        return search_dfs_bb(
            plans,
            deadline=deadline,
            max_nodes=max_nodes,
            branch_and_bound=branch_and_bound,
            verbose=verbose,
        )

    def schedule_path_selection(
        self,
        plans: Sequence[RelaxedVehiclePlan],
        *,
        lambda_path: float = 1.0,
        branch_and_bound: bool = True,
        verbose: bool = True,
    ) -> RelaxedSearchResult:
        from coarse_scheduler import search_dynamic_codesign_dfs_bb

        return search_dynamic_codesign_dfs_bb(
            plans,
            lambda_path=lambda_path,
            branch_and_bound=branch_and_bound,
            verbose=verbose,
        )


@dataclass(frozen=True)
class FiveSpaceScheduler:
    """Placeholder for the CDC-style per-intersection five-space resource model."""

    name: str = "five_space"

    def schedule_fixed(
        self,
        plans: Sequence[VehiclePlan],
        *,
        deadline: Optional[float] = None,
        max_nodes: Optional[int] = None,
        branch_and_bound: bool = True,
        verbose: bool = True,
    ) -> SearchResult:
        raise NotImplementedError("FiveSpaceScheduler is not implemented yet")

    def schedule_path_selection(
        self,
        plans: Sequence[RelaxedVehiclePlan],
        *,
        lambda_path: float = 1.0,
        branch_and_bound: bool = True,
        verbose: bool = True,
    ) -> RelaxedSearchResult:
        raise NotImplementedError("FiveSpaceScheduler is not implemented yet")
