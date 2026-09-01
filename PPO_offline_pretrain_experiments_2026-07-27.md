# PPO offline exact pretrain + fine-tune notes (2026-07-27)

## Scope

- Fixed-shortest-path mode enabled (`--fix-shortest-paths`) for schedule-only PPO experiments.
- Offline supervised pretraining from exact-tree solutions (`offline_exact_pretrain.py`).
- Targeted state-design and training-flow iteration for tree-search aligned branch scheduling.

## Code changes included in this commit

Modified files:

- `PPO_model/compare_ppo_fcfs_exact.py`
- `PPO_model/encoding.py`
- `PPO_model/evaluate.py`
- `PPO_model/mine_fcfs_hard_cases.py`
- `PPO_model/rollout_buffer.py`
- `PPO_model/train.py`
- `PPO_model/trainer.py`
- `coarse_scheduler.py`
- `main.py`
- `PPO_model/offline_exact_pretrain.py` (new)

## Key run commands and outcomes

### Run A (pretrain only)

```bash
python -m PPO_model.offline_exact_pretrain --n-robots 5 --exact-cases 300 --pretrain-epochs 5 --run-ppo-updates 0 --skip-trivial-cases --fix-shortest-paths --run-root output/ppo_pretrain --run-id n5_offline_v1
```

- Final lines:
  - `[pretrain epoch 005] actor_loss=0.599 critic_loss=16.640 entropy=0.032`
  - `pretrain final fixed-case PPO cost=8.995574, exact_cost=5.853982, gap=3.141593`

### Run B (pretrain + PPO fine-tune)

```bash
python -m PPO_model.offline_exact_pretrain --n-robots 5 --exact-cases 500 --pretrain-epochs 10 --skip-trivial-cases --fix-shortest-paths --run-ppo-updates 20 --run-ppo-episodes-per-update 2 --run-root output/ppo_pretrain --run-id n5_offline_v2
```

- Pretrain trend:
  - Actor loss: `0.568 -> 0.372`
  - Critic loss: `28.083 -> 7.246`
  - Entropy: `0.644 -> 0.056`
- PPO update trace (20 updates) produced fluctuating `mean_J` on rollout.
- `fixed_J` on held reference case remained `8.996` and did not improve within this run.
- Final:
  - `final fixed-case PPO cost=8.995574`
  - `final fixed-case exact cost=5.853982, gap=3.141593`

## Notes

- Observed that `fixed_J` is fixed-case based and may remain unchanged even when actor distribution moves on random training cases.
- Current workflow intentionally keeps exact-dataset supervised stage and optional PPO stage in one script.

