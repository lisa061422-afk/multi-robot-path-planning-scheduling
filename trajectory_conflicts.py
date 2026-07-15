"""Static movement-level contention rules for one-lane intersections.

The map uses one global, non-rotating coordinate convention.  Route IDs are
the 12 fixed ``entry_dir -> exit_dir`` movements defined in ``traffic_map``.
The scheduler remains conservative: a pair conflicts unless it appears in the
explicit non-conflicting whitelist below.
"""

from __future__ import annotations

import os
from itertools import combinations
from typing import Iterable, Sequence, Tuple

from traffic_map import ROUTE_ID_BY_ENTRY_EXIT


TRAJECTORY_FILTER_ENV = "PATHPLANNING_TRAJECTORY_CONFLICT_FILTER"

ENTRY_EXIT_BY_ROUTE_ID = {
    route_id: entry_exit
    for entry_exit, route_id in ROUTE_ID_BY_ENTRY_EXIT.items()
}

# Route IDs 3, 6, 9, 12 are the four right turns at four different corners.
_RIGHT_TURNS = (3, 6, 9, 12)

# Unordered route-ID pairs whose swept trajectories are allowed concurrently.
# This is deliberately a whitelist: an omitted pair remains safely conflicting.
NON_CONFLICTING_ROUTE_ID_PAIRS = frozenset(
    {
        *(
            frozenset(pair)
            for pair in combinations(_RIGHT_TURNS, 2)
        ),
        # Opposing straight movements use different lanes.
        frozenset((2, 8)),
        frozenset((5, 11)),
        # A straight movement and the two remote-corner right turns.
        frozenset((2, 9)),
        frozenset((2, 12)),
        frozenset((8, 3)),
        frozenset((8, 6)),
        frozenset((5, 3)),
        frozenset((5, 12)),
        frozenset((11, 6)),
        frozenset((11, 9)),
        # User-defined static-geometry pattern A: (1, 12), plus its three
        # 90-degree rotational counterparts in the fixed global route IDs.
        frozenset((1, 12)),
        frozenset((4, 3)),
        frozenset((7, 6)),
        frozenset((10, 9)),
        # User-defined static-geometry pattern B: (1, 6), plus rotations.
        frozenset((1, 6)),
        frozenset((4, 9)),
        frozenset((7, 12)),
        frozenset((10, 3)),
    }
)


def set_trajectory_conflict_filter(enabled: bool) -> None:
    """Set the process-level experiment switch (inherited by worker processes)."""

    os.environ[TRAJECTORY_FILTER_ENV] = "1" if enabled else "0"


def trajectory_conflict_filter_enabled() -> bool:
    return os.environ.get(TRAJECTORY_FILTER_ENV, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def route_ids_conflict(route_id_i: int, route_id_j: int) -> bool:
    """Return whether two movements require an ordering at one intersection."""

    if route_id_i == route_id_j:
        return True
    entry_i, exit_i = ENTRY_EXIT_BY_ROUTE_ID[route_id_i]
    entry_j, exit_j = ENTRY_EXIT_BY_ROUTE_ID[route_id_j]
    # One lane per side: same-origin vehicles cannot enter abreast, and
    # different-origin vehicles cannot merge into the same exit concurrently.
    if entry_i == entry_j or exit_i == exit_j:
        return True
    return frozenset((route_id_i, route_id_j)) not in NON_CONFLICTING_ROUTE_ID_PAIRS


def simultaneous_prefix(
    priority_order: Sequence[int],
    route_ids_by_vehicle: Sequence[int],
) -> Tuple[int, ...]:
    """Greedily admit every vehicle compatible with higher-priority runners.

    With filtering disabled this returns only the first vehicle, exactly
    reproducing the old one-runner-per-intersection model.
    """

    if not priority_order:
        return ()
    if not trajectory_conflict_filter_enabled():
        return (priority_order[0],)

    running = []
    for vehicle in priority_order:
        route_id = route_ids_by_vehicle[vehicle]
        if all(
            not route_ids_conflict(route_id, route_ids_by_vehicle[other])
            for other in running
        ):
            running.append(vehicle)
    return tuple(running)
