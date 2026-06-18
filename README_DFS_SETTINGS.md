# DFS, Pruning, and Parallel Settings

This note records how to run the scheduler modes in this project.

## 1. Core Terms

### DFS

DFS means depth-first search over the decision tree.

In fixed-path mode, branches come from scheduling contention:

```text
v_in(tw) = 1
```

In dynamic path-selection co-design mode, branches can come from both:

```text
v_in(tw) = 1              scheduling priority
z_nq(i,j)(tw) = 1        path-selection decision
```

### Branch-and-Bound Pruning

Pruning means:

```text
if current node cost J >= current best feasible J:
    stop expanding this node
```

This is safe because the current implementation only adds nonnegative future
costs:

```text
J = delay cost + lambda_path * extra path cost
```

So once a partial node is already worse than the current best complete solution,
none of its children can become better.

## 2. Main Parameters in `main.py`

The main switches are inside `demo_2x2()`.

```python
enable_path_selection = True
show_all_branches = True
use_parallel_dynamic = False
parallel_frontier_depth = 2
parallel_max_workers = 4
lambda_path = 1.0
```

Recommended default while debugging the code and inspecting the full tree:

```python
enable_path_selection = True
show_all_branches = True
use_parallel_dynamic = False
parallel_frontier_depth = 2
parallel_max_workers = 4
lambda_path = 1.0
```

This runs true dynamic co-design in one process and keeps all branches for
visualization. It is much easier to debug manually because breakpoints and local
variables stay in the main Python process.

Recommended default for larger experiments:

```python
enable_path_selection = True
show_all_branches = False
use_parallel_dynamic = True
parallel_frontier_depth = 2
parallel_max_workers = 4
lambda_path = 1.0
```

This runs true dynamic co-design with parallel subtree DFS and branch-and-bound
pruning.

## 3. Fixed-Path Mode

Use this when every vehicle route is predetermined before scheduling.

```python
enable_path_selection = False
show_all_branches = False
```

Then the code calls:

```python
search_parallel_dfs_bb(...)
```

Important note: the current `search_parallel_dfs_bb` is a compatibility wrapper.
It currently calls serial DFS with pruning to avoid overloading the machine.

For a full tree preview, turn pruning off:

```python
show_all_branches = True
```

This is useful for small examples and debugging, but it can explode quickly.

## 4. Dynamic Path-Selection Co-Design Mode

Use this for the real algorithm:

```python
enable_path_selection = True
use_parallel_dynamic = False
show_all_branches = True
```

Then the code calls:

```python
search_dynamic_codesign_dfs_bb(...)
```

This is different from the old relaxed solver.

The dynamic solver does **not** choose complete paths at the root. Instead, each
vehicle carries a current feasible route set:

```text
route_candidates[n]
```

When vehicle `n` reaches a task-generation moment and the next traversal is not
unique among `route_candidates[n]`, the decision tree creates a path-selection
branch at that `tw`.

This supports larger maps where path choices happen in the middle of the route.

## 5. How to Turn Pruning On or Off

Pruning on:

```python
show_all_branches = False
```

This passes:

```python
branch_and_bound=True
```

Pruning off:

```python
show_all_branches = True
```

This passes:

```python
branch_and_bound=False
```

Use pruning on for actual experiments.

Use pruning off only when you want to visually inspect every branch in a small
toy example.

## 6. Old Parallel Relaxed Solver

There is an older function:

```python
search_relaxed_parallel_dfs_bb(...)
```

It parallelizes by assigning each full path-choice combination to a worker.

This is **not** the true dynamic co-design algorithm, because it fixes complete
paths before scheduling begins.

It is kept in `coarse_scheduler.py` only as a legacy benchmark/reference.
It is no longer exposed as a switch in `main.py`.

Do not use it as the main co-design result for 3x3 or larger maps.

## 7. Correct Parallel DFS for Dynamic Co-Design

The correct parallel strategy is:

1. Expand the true dynamic decision tree from the root for a few layers.
2. Collect the frontier nodes.
3. Give each frontier node to a worker.
4. Each worker runs DFS + branch-and-bound inside its subtree.
5. Compare all worker results and choose the smallest global `J`.

This is basically:

```text
small BFS/frontier expansion first
then parallel DFS on each subtree
```

It is still globally optimal if:

```text
all frontier subtrees are searched
and pruning only removes nodes with J >= known best J
```

This is now implemented as:

```python
search_dynamic_codesign_parallel_dfs_bb(...)
```

Use it when the algorithm logic is stable:

```python
use_parallel_dynamic = True
```

Keep it off when manually debugging:

```python
use_parallel_dynamic = False
```

## 8. Recommended Parallel Parameters

The dynamic parallel function uses this interface:

```python
search_dynamic_codesign_parallel_dfs_bb(
    relaxed_plans,
    lambda_path=lambda_path,
    frontier_depth=2,
    max_workers=4,
    branch_and_bound=True,
    verbose=True,
)
```

Recommended starting values:

```python
frontier_depth = 2
max_workers = 4
branch_and_bound = True
```

How to tune:

```text
frontier_depth = 1
    Safer for small examples. Low overhead, but may not create enough work.

frontier_depth = 2
    Good default. Usually enough independent subtrees for 4 workers.

frontier_depth = 3
    Try only if there are many vehicles and the tree branches early.
    It can create too many frontier nodes.

max_workers = 4
    Good default for this machine. Avoid very large values because Python
    process parallelism can consume memory quickly.
```

## 9. Current Implementation Status

Implemented:

```text
fixed-path DFS + branch-and-bound
dynamic path-selection co-design DFS + branch-and-bound
dynamic path-selection co-design parallel DFS
legacy full-path-combination relaxed parallel baseline
```

So the main real algorithm is:

```python
search_dynamic_codesign_parallel_dfs_bb(...)
```

The serial version is still available for debugging:

```python
search_dynamic_codesign_dfs_bb(...)
```
