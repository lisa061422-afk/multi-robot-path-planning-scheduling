# Path Planning and Schedule Co-Design

Python prototype for centralized traffic scheduling and path-planning/scheduling
co-design.

Current scope:

- arbitrary coarse traffic maps built from graph-style intersection layouts;
- entrance/exit port generation;
- fixed-path DFS scheduling baseline;
- dynamic path-selection co-design DFS;
- dynamic parallel DFS using frontier subtrees;
- interactive HTML visualization of decision trees, path DAGs, and schedules.

## Run

```bash
python main.py
```

The main demo writes outputs under `output/`, including:

```text
output/relaxed_interactive_solution.html
```

## Main Controls

The main switches are in `demo_2x2()` inside `main.py`.

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

## Tests

```bash
python test_coarse_scheduler.py
python test_traffic_map.py
python test_algorithm_examples.py
```

