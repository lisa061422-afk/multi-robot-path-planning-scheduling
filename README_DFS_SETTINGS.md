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
J = delay cost + extra path cost
```

So once a partial node is already worse than the current best complete solution,
none of its children can become better.

## 2. Main Parameters in `main.py`

The main switches are inside `demo_fixed_map()`.

```python
fixed_map = "paper_3x3"          # "paper_2x2" or "paper_3x3"
enable_path_selection = True
use_baseline_path_filter = False # show all enumerated path-selection options
keep_min_hop_route_options = True
show_all_branches = True
use_parallel_dynamic = False
parallel_frontier_depth = 2
parallel_max_workers = 4
```

One second of extra travel time is added directly to one second of scheduling
delay; there is no separate path-cost weight.

`fixed_map` and `show_all_branches` are intentionally independent. Use
`fixed_map` only to choose the map, and use `show_all_branches` only to choose
whether to draw/search the full tree for debugging.

Recommended default while debugging the code and inspecting the full tree:

```python
enable_path_selection = True
use_baseline_path_filter = False
keep_min_hop_route_options = True
show_all_branches = True
use_parallel_dynamic = False
parallel_frontier_depth = 2
parallel_max_workers = 4
```

This runs true dynamic co-design in one process and keeps all branches for
visualization. It is much easier to debug manually because breakpoints and local
variables stay in the main Python process. Keep this for small cases only.

Recommended default for larger experiments:

```python
enable_path_selection = True
use_baseline_path_filter = True
keep_min_hop_route_options = True
show_all_branches = False
use_parallel_dynamic = True
parallel_frontier_depth = 2
parallel_max_workers = 4
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

## 5. Baseline Upper-Bound Path Filter

This optional filter is controlled by:

```python
use_baseline_path_filter = False  # validation/visualization default
use_baseline_path_filter = True   # larger-experiment pruning mode
keep_min_hop_route_options = True
```

When enabled, `main.py` first solves a feasible shortest-path fixed scheduling
case:

```python
J_ub = shortest_path_delay_upper_bound(...)
```

Then it filters each vehicle's route options before co-design:

```text
keep path p if (T_p - T_shortest) <= J_ub
```

The current demo also keeps all minimum-hop route options:

```python
keep_min_hop_route_options = True
```

This makes the viewer and training candidate set retain same-intersection-count
paths, even when different turn sequences make their free-flow costs slightly
larger than the fastest route.

The upper-bound pruning rule is safe because `J_ub` is the true cost of a
feasible incumbent solution, not an estimate. The optimum must satisfy:

```text
J_star <= J_ub
```

Any single route option whose path-extra penalty already exceeds `J_ub` cannot
appear in a globally better solution, even if it eliminates all future delay.
The `keep_min_hop_route_options` setting deliberately relaxes this display/training
candidate set so that equal-hop route choices are still visible.

To recover the original full candidate set:

```python
use_baseline_path_filter = False
```

This is the setting to use when you want to debug whether the filter is hiding a
route from the visualization.

## 6. How to Turn Pruning On or Off

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

## 7. Old Parallel Relaxed Solver

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

## 8. Correct Parallel DFS for Dynamic Co-Design

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

## 9. Recommended Parallel Parameters

The dynamic parallel function uses this interface:

```python
search_dynamic_codesign_parallel_dfs_bb(
    relaxed_plans,
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

## 10. Current Implementation Status

Implemented:

```text
fixed-path DFS + branch-and-bound
dynamic path-selection co-design DFS + branch-and-bound
dynamic path-selection co-design parallel DFS
baseline upper-bound route-candidate filtering
paper_3x3 fixed training map
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

Current correctness checks:

```bash
python -m unittest test_traffic_map.py test_coarse_scheduler.py test_algorithm_examples.py
```

The most important current 3x3 test is:

```text
test_dynamic_codesign_matches_enumerated_route_choices_on_3x3
```

It compares the dynamic path-selection solver against the exhaustive
full-route-combination baseline on the 3x3 map, then verifies no resource
overlap and checks `J = delay + path_extra`.

## 11. Interactive HTML Viewer

The dynamic co-design HTML output is:

```text
output/relaxed_interactive_solution.html
```

Current viewer panels:

```text
Decision Tree
Basic Map
Vehicle Path Options
Path Selection Branches
Vehicle Path Branch Trees
Schedule
```

`Path Selection Branches` is important for 3x3 and larger maps. It explicitly
lists choices such as:

```text
N1, P1 -> P7, reached [I1], next choices I2 / I4
```

This table is separate from the terminal-path display. A terminal path count is
not the same thing as the number of available route-choice branches.

## 12. Scheduler/Resource-Model Boundary

The code now keeps the stable scheduler data shapes in:

```text
scheduler_models.py
```

The swappable scheduler facade lives in:

```text
resource_schedulers.py
```

For the first RL environment, depend on:

```python
from resource_schedulers import CoarseIntersectionScheduler
```

This keeps the environment coupled to a small interface instead of directly to
the large `coarse_scheduler.py` implementation.  The current baseline is:

```text
CoarseIntersectionScheduler
    each intersection is one conservative resource
```

The future CDC-style model is reserved as:

```text
FiveSpaceScheduler
    each intersection is divided into conflict spaces
```

`FiveSpaceScheduler` is intentionally a placeholder for now.  It should fill
the same fixed-path and path-selection scheduling methods so the RL environment
can switch resource models without changing its observation/action plumbing.

## 13. Random Exact Validation

Use this script to generate deterministic random small cases and check the
current dynamic co-design solver:

```bash
python random_validation.py
```

Each completed case is printed immediately with its elapsed solver time and is
also appended to:

```text
output/random_validation_times.csv
```

The timing CSV records:

```text
elapsed_seconds, optimal_cost, delay, path_extra
```

Each case also keeps an interactive HTML viewer:

```text
output/random_validation_html/case_XX.html
```

For larger multi-robot batches, the default case HTML is compact and only shows
the selected optimal co-design schedule.  If you need the decision tree, use:

```bash
python random_validation.py --full-tree-html --max-terminal-paths 50
```

`--max-terminal-paths` limits how many terminal paths are drawn in the full tree.
The optimal path is always included.

The default run creates 30 cases, uses:

```text
map=paper_2x2
search_dynamic_codesign_parallel_dfs_bb
branch_and_bound=True
frontier_depth=2
max_workers=min(4, cpu_count)
```

and compares each answer against the exhaustive full-path-choice baseline:

```text
search_relaxed_parallel_dfs_bb
```

Default complexity caps:

```text
option_product <= 81
max potential vehicles per resource <= 3
max route-option visits per resource <= 8
```

These caps are for exact-search tractability, not physical feasibility.  With
finite vehicles and finite routes, the one-resource scheduler normally has a
finite feasible schedule; the practical problem is that the decision tree can
grow combinatorially.  If a sampled case violates the caps, the script rejects
it before solving and samples another case.

For a quick run without the cross-check baseline:

```bash
python random_validation.py --no-baseline-compare
```

For a larger stress run, raise the caps carefully:

```bash
python random_validation.py --cases 100 --max-resource-vehicles 4
```

To explicitly include a larger map later:

```bash
python random_validation.py --maps paper_2x2,grid_3x3
```

Use `--deadline` or `--max-nodes` only as a watchdog.  If either cutoff is hit,
the script treats the result as incomplete because it is only best-so-far, not
an exact optimum.
