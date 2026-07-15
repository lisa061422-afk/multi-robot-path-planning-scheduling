# Fixed-Priority Reservation Scheduler

This directory is reserved for the new exact scheduling algorithm. It is kept
separate from the existing resetting-rule implementation to avoid mixing state,
assumptions, or experimental results.

## Intended architecture

1. Precompute all potential vehicle conflicts by intersection and conflict space.
2. Enumerate or branch on fixed priority decisions.
3. Convert fixed priorities into reservation/time-difference constraints.
4. Compute a complete conflict-free reservation plan before execution.
5. Preserve the original Timing Model state structure (`ddl`, `remain`,
   `response`, and significant moments) to replay/evaluate the plan.
6. Do not use resetting rules or rewrite previously executed resource history.

## Planned modules

- `models.py`: vehicles, intersection visits, conflict spaces, and priorities.
- `conflicts.py`: static conflict preprocessing.
- `reservation_solver.py`: fixed-priority timing/reservation solver.
- `priority_search.py`: exact DFS/branch-and-bound priority search.
- `timing_model.py`: Timing Model replay using the reservation plan.
- `main.py`: experiment entry point.
- `tests/`: correctness and optimality cross-checks.

The existing scheduler files outside this directory should remain unchanged
while this implementation is developed and validated.
