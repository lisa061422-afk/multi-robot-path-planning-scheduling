# PPO Design Notes (Current Implementation, 2026-07)

This document reflects the currently shipped PPO implementation in
`ppo_training/` as of now.

## 1) Objective

For each trajectory node \(l\), accumulated objective is
\[
g = g_{\text{delay}} + g_{\text{path}}.
\]
There is no extra weighting factor in this code path.  PPO learns a policy over
branching decisions in the co-design search tree.

## 2) Environment and model interface

- Exact model is used for state transitions; PPO only chooses among legal branches.
- Only the strict one-resource-per-intersection rule is active in current
  training mode.
- PPO is called only at decision nodes with at least 2 legal branches.
- Chains of forced transitions are skipped inside the environment (`_advance_forced`).

### Reward
For one PPO step:
\[
r_t = -(g_{t+1} - g_t)
\]
(`DecisionTreeEnv.step` returns this as `reward`).

### Episode
An episode is one rollout from reset to terminal leaf.

## 3) State and action

### State (`state`)
- Built from current search node using `BranchEncoder.encode_state(node)`.
- Dimensionality:
  - `state_dim = 1 + n_robots * per_robot`
  - \(per\_robot = 10 + 2\*(n\_resources+1) + 3*n\_resources + 2*n\_ports + max\_route\_options\)
- Includes normalized global time, per-robot task/remaining-time features,
  resource/path masks, in/out/next masks, entrance/exit one-hots, and route-candidate mask.

### Action (`a_t`)
- An action is an index into current legal branch list.
- Branch candidates are encoded by `BranchEncoder.encode_actions(parent, branches)`.
- Action encoding per branch concatenates per-robot features:
  selected resource one-hot, previous/next intersection one-hots, extra-time,
  active/continue/interruption flags.
- `action_dim = n_robots * ((n_resources+1) + 2*(n_resources+1) + 4)`.

## 4) Policy / Value parameterization

- **Actor (branch scoring):** `BranchScoringActor` takes one state vector and one
  candidate-branch matrix.
  - Input dim: `state_dim + action_dim`
  - Hidden layers: configurable (default 2), width `hidden_dim=128`
  - Shared weights over all legal branches
  - Output: 1 logit per branch
  - Activations: `SiLU`, with `LayerNorm`; orthogonal init

- **Critic:** `StateValueCritic` takes only state.
  - Input dim: `state_dim`
  - Hidden layers: configurable (default 2), width `hidden_dim=128`
  - Output: scalar value \(V(s)\)
  - Activations: `SiLU`, with `LayerNorm`; orthogonal init

## 5) PPO math used in code

At decision time:
- logits = actor(state, all_legal_branches)
- \(\pi(a_t|s_t)=\text{Categorical}(\text{logits})\)

For each collected transition:
- old log-prob = \(\log \pi_{\text{old}}(a_t|s_t)\) (stored in buffer)
- new log-prob = \(\log \pi_{\text{new}}(a_t|s_t)\)
- probability ratio:
\[
\rho_t = \exp(\log \pi_{\text{new}} - \log \pi_{\text{old}})
\]

GAE advantage (discount 1.0, lambda 0.95 default):
\[
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
\]
\[
A_t = \delta_t + \gamma \lambda A_{t+1}
\]

PPO clipped surrogate (batch):
\[
L^{\text{CLIP}}_t = \min\!\Big(\rho_t A_t,\ \text{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t\Big)
\]
Actor loss is \(-\mathbb{E}[L^{\text{CLIP}}_t]\) and entropy is subtracted with
coefficient `entropy_coef`.

Critic loss is MSE between predicted value and value target
\((A_t + V(s_t))\), scaled by `value_loss_coef`.

## 6) Rollout / training flow

For each update:
1. Collect `episodes-per-update` episodes (`train.py`).
2. Store transitions in one shared buffer (`RolloutBuffer`).
3. Shuffle buffer and do `update-epochs` passes.
4. Each pass uses mini-batches of size `minibatch-size`.

Relevant defaults in `train.py`:
- `learning-rate=3e-4`
- `updates=100`
- `episodes-per-update=16`
- `update-epochs=4`
- `minibatch-size=64`
- `entropy-coef=0.01`
- `clip-epsilon=0.2`
- `discount_factor=1.0`
- `gae-lambda=0.95`
- `value-loss-coef=0.5`
- `max-grad-norm=0.5`

## 7) Implemented optional features

- Reward normalization: `--reward-norm-mode {none, absmax}`
- Optional linear schedules:
  - `--entropy-schedule linear`
  - `--lr-schedule linear`
- Optional behavior cloning warm-start from exact trajectory (`--bc-pretrain-*`).
- Fixed-case or random case generation with `--fixed-case` and `--fix-shortest-paths`.
- Optional skipping of trivial cases (`--skip-trivial-cases`).

## 8) Code entry and key files
- `ppo_training/train.py` (CLI, config, loops)
- `ppo_training/trainer.py` (collect/update)
- `ppo_training/environment.py` (decision tree wrapper)
- `ppo_training/encoding.py` (state/action encoding)
- `ppo_training/networks.py` (actor + critic)
- `ppo_training/rollout_buffer.py` (GAE buffer)
- `ppo_training/evaluate.py` (validation vs exact solver)

---

This file replaces the previous stale design notes. It is intended as the
authoritative in-repo PPO design reference for the current implementation.
