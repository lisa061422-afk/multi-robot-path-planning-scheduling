# PPO training quick start

The first implementation trains a variable-branch Actor on the `3 x 3` map
with exactly three robots and strict one-robot-per-intersection mutual
exclusion.  The exact DFS implementation remains the ground-truth baseline.

Run a short CPU smoke training:

```powershell
python -m ppo_training.train --updates 2 --episodes-per-update 4 --fixed-case
```

Run randomized three-robot training:

```powershell
python -m ppo_training.train --updates 100 --episodes-per-update 16
```

Add `--exact-eval` to compare the final greedy policy with exact DFS on the
fixed reference case.  Exact evaluation can be much slower than a PPO rollout.

Evaluate a saved checkpoint independently:

```powershell
python -m ppo_training.evaluate `
  --checkpoint output/ppo_n3/ppo_branch_actor.pt `
  --exact
```

Outputs are written by default to:

```text
output/ppo_n3/ppo_branch_actor.pt
output/ppo_n3/training_metrics.csv
```

The environment generates all immediate legal children at a visited decision
node, the shared Actor scores those children, and only one sampled child is
followed.  Single-child nodes are traversed automatically.  PPO rollouts do not
use the DFS incumbent-cost pruning rule.
