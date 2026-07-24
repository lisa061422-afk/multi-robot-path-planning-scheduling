# PPO training quick start

The first implementation trains a variable-branch Actor on the `3 x 3` map
with exactly three robots and strict one-robot-per-intersection mutual
exclusion.  The exact DFS implementation remains the ground-truth baseline.

Run a short CPU smoke training:

```powershell
python -m PPO_model.train --updates 2 --episodes-per-update 4 --fixed-case
```

Run randomized three-robot training:

```powershell
python -m PPO_model.train --updates 100 --episodes-per-update 16
```

Run and save training curves for convergence check:

 ```powershell
 python -m PPO_model.train `
  --updates 100 --episodes-per-update 16 `
  --group-robots 2-12 `
  --plot-after-train `
  --plots-dir output/ppo_{n_robots}/plots
```

每次训练默认会自动建新 run 文件夹（默认在 `output/ppo_runs/run_时间戳/...`）：

```powershell
python -m PPO_model.train --updates 100 --episodes-per-update 16 --n-robots 3
```

你也可以自定义 run 目录名与重用同一组结果文件名：

```powershell
python -m PPO_model.train --updates 100 --episodes-per-update 16 --n-robots 3 --run-id test_v1 --run-root output/ppo_experiments
```

Use one-vehicle-per-entrance random cases (default):

```powershell
python -m PPO_model.train --updates 100 --episodes-per-update 16 --max-vehicles-per-entrance 1
```

Run grouped training across robot counts:

```powershell
python -m PPO_model.train --updates 40 --episodes-per-update 16 --group-robots 2-12
```

Each N will be trained as an independent group (separate model per N).

Add `--exact-eval` to compare the final greedy policy with exact DFS on the
fixed reference case.  Exact evaluation can be much slower than a PPO rollout.

Evaluate a saved checkpoint independently:

```powershell
python -m PPO_model.evaluate `
  --checkpoint output/ppo_n3/ppo_branch_actor.pt `
  --exact
```

Outputs are written by default to:

```text
output/ppo_n3/ppo_branch_actor.pt
output/ppo_n3/training_metrics.csv
output/ppo_n3/plots/ppo_n3_training_curves.png  # if --plot-after-train is set
```

The environment generates all immediate legal children at a visited decision
node, the shared Actor scores those children, and only one sampled child is
followed.  Single-child nodes are traversed automatically.  PPO rollouts do not
use the DFS incumbent-cost pruning rule.
