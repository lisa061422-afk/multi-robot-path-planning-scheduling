# Path Planning and Schedule Co-Design

Python prototype for centralized traffic scheduling and path-planning/scheduling
co-design.

Current scope:

- arbitrary coarse traffic maps built from graph-style intersection layouts;
- entrance/exit port generation;
- fixed-path DFS scheduling baseline;
- dynamic path-selection co-design DFS;
- optional baseline-upper-bound path filtering before co-design;
- dynamic parallel DFS using frontier subtrees;
- interactive HTML visualization of decision trees, maps, path options, path
  selection branches, path DAGs, and schedules.

## Run

```bash
python main.py
```

The main demo writes outputs under `output/`, including:

```text
output/relaxed_interactive_solution.html
```

## Main Controls

The main switches are in `demo_fixed_map()` inside `main.py`.

To choose the fixed training map:

```python
fixed_map = "paper_3x3"  # "paper_2x2" or "paper_3x3"
```

`paper_2x2` keeps the original small example. `paper_3x3` adds the second
typical fixed map, where OD pairs such as `P1 -> P7` have middle-route path
selection after the entrance intersection.

To enable or disable the current upper-bound route filter:

```python
use_baseline_path_filter = False  # show all enumerated path-selection options
use_baseline_path_filter = True   # safe pruning by baseline upper bound
keep_min_hop_route_options = True # keep all minimum-hop route choices visible
```

When enabled, the demo first solves the shortest-path fixed scheduling case to
get a feasible upper bound `J_ub`. It then removes route options satisfying:

```text
(T_path - T_shortest) > J_ub
```

This is a safe path-candidate filter, not a heuristic node-count cutoff. Keep it
off while validating the algorithm or inspecting path-selection structure. With
`keep_min_hop_route_options = True`, every route with the minimum number of
intersections is still retained, so same-hop path choices remain visible for
training and inspection even when turn costs differ.

For debugging with all branches visible:

```python
enable_path_selection = True
show_all_branches = True
use_parallel_dynamic = True
parallel_frontier_depth = 2
parallel_max_workers = 4
```

For faster optimization with branch-and-bound pruning:

```python
show_all_branches = False
```

More details are in:

```text
README_DFS_SETTINGS.md
```

## PPO-guided branch selection

The initial PPO implementation is in `PPO_model/`.  It uses the `3 x 3`
map, exactly three robots, strict one-robot-per-intersection mutual exclusion,
and a shared neural scorer for the variable legal branch set.  The original
DFS remains the exact ground-truth solver.

Quick smoke training:

```powershell
python -m PPO_model.train --updates 2 --episodes-per-update 4 --fixed-case
```

训练并直接导出指标曲线（便于看收敛）：

```powershell
python -m PPO_model.train --updates 80 --episodes-per-update 16 --plot-after-train
```

See `PPO_model/README.md` and `PPO_DESIGN.md` for the training commands and
the mathematical design.

## Current 3x3 Demo Notes

The default map is currently:

```python
fixed_map = "paper_3x3"
use_baseline_path_filter = False
keep_min_hop_route_options = True
show_all_branches = True
```

The default 3x3 requests are defined in `default_vehicle_requests()`:

```text
N1: P2 -> P6
N2: P1 -> P9
# N3: P12 -> P5 is kept as a commented cross-traffic option
```

Path-extra seconds and waiting-delay seconds are added directly:
`J = delay + path_extra`. In the current all-path-selection display mode, the
demo prints:

```text
Baseline path filter: disabled; using all route options
```

## Interactive HTML Viewer

`output/relaxed_interactive_solution.html` now includes:

- a decision tree with wheel zoom and double-click reset;
- a compact `Basic Map` with intersection indices and port locations;
- a `Vehicle Path Options` table listing each vehicle's entrance, exit,
  selected path, and candidate paths;
- a `Path Selection Branches` table showing actual branch points such as
  `[I1] -> I2 / I4`;
- compact `Vehicle Path Branch Trees`, where only the selected path has green
  arrows and grey alternatives stay unlabeled;
- local schedule panels on the right.

The default screen split is 3:1, with the left pane at 75%.

## Tests

```bash
python test_coarse_scheduler.py
python test_traffic_map.py
python test_algorithm_examples.py
```
