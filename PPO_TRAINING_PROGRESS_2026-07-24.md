# PPO Training Progress — 2026-07-24

This note preserves the current modeling decisions, experiments, conclusions,
and next-step design discussion for the 3x3 intersection scheduling project.
It is a structured technical record rather than a verbatim chat transcript.

## Current problem definition

- The exact scheduling algorithm remains an independent solver.
- PPO and exact evaluation receive the same fixed shortest route for every
  vehicle. The current task is scheduling only, not path/schedule co-design.
- Training is grouped by robot count `N = 2, ..., 12`.
- Each random case uses:
  - a 3x3 network with 12 ports;
  - at most one vehicle entering from each port;
  - an exit different from the entrance;
  - initial release time `alpha0 ~ Uniform(0, 5 s)`;
  - no `--max-resource-vehicles` restriction.
- Actor and critic are MLPs.
- Behavioral cloning and the previous supervised-learning path are disabled so
  the reported baseline remains pure PPO.

## Completed training baseline

The main grouped run is:

```text
output/ppo_runs/fast_parallel_20260723_211520_cont
```

It completed 600 PPO updates and 9,600 episodes for every `N` from 2 through
12, for 105,600 episodes in total.

On a verified ordinary fixed-path comparison set, PPO behaved almost exactly
like FCFS. On 97 nontrivial cases it was better in 4, tied in 89, and worse in
4. The result did not support a claim that raw PPO consistently outperformed
FCFS.

## Pending-delay reward shaping

The original event reward is:

```text
r_k = -(g_{k+1} - g_k)
```

Because delay is booked into `g` mainly when a task completes, the reward can
arrive well after the scheduling decision that caused it.

For an active task `i`, define:

```text
C_i       nominal uninterrupted execution time
r_i       remaining execution time
p_i       C_i - r_i, useful progress since the latest restart
alpha_i   task request/release time
L_i(t)    max(0, t - alpha_i - p_i)
```

The shaped training cost and reward are:

```text
G_train(s) = g(s) + sum_i L_i(s)
r_k = -(G_train(s_{k+1}) - G_train(s_k))
```

This has the intended behavior:

- waiting: time increases while progress does not, so delay is charged
  immediately;
- uninterrupted execution: time and progress increase together, so no delay
  is charged;
- preemptive-repeat interruption: progress resets and the lost progress is
  charged at the interruption event;
- completion: pending delay transfers into `g` without changing the terminal
  objective.

For a ready-but-not-running set `W_k` over an event interval `Delta t`, the
same accounting can be written as:

```text
interval cost
  = |W_k| * Delta t
    + sum(lost progress at restart events)
```

This is the earliest exact causal accounting for preemptive-repeat tasks. Work
cannot be declared wasted before the interruption is known.

The implementation is selectable with:

```text
--reward-cost-mode delta_g
--reward-cost-mode pending_delay
```

The default remains `delta_g` for backward compatibility.

## N=5 reward A/B pilot

Two fresh runs used the same seed and cases:

```text
output/ppo_runs/pilot_n5_delta_g_20260724
output/ppo_runs/pilot_n5_pending_delay_20260724
```

Each run used 100 updates, 32 episodes per update, and 3,200 episodes total.

Reward diagnostics:

- nonzero decision rewards increased from 69.2% to 100%;
- the largest reward jump decreased from 26.39 to 16.73;
- ending-window critic loss decreased from about 19.85 to 12.78;
- ending-window mean rollout cost decreased from about 5.98 to 5.77.

However, the held-out greedy scheduling result did not materially improve:

- both reward versions had a 12.987% mean gap to exact;
- both matched exact in 66 of 94 nontrivial cases;
- both were 1 better, 93 tied, and 0 worse than FCFS.

Conclusion: pending-delay shaping improved reward density and critic fitting,
but did not by itself teach a better actor policy.

## FCFS-hard benchmark

The hard benchmark uses the same N=5 random-case rules but retains only cases
where exact fully solves the tree and:

```text
(J_FCFS - J_exact) / J_exact >= 10%
```

Selection uses only exact and FCFS results; PPO is evaluated afterward to
avoid PPO-dependent selection bias.

From 524 candidates, 100 fully solved hard cases were retained. The mean FCFS
gap to exact on this deliberately difficult set was 54.57%.

Results:

| Model | Mean gap to exact | Better / tie / worse than FCFS | Exact matches |
| --- | ---: | ---: | ---: |
| 600-update PPO | 53.99% | 2 / 98 / 0 | 1 |
| 100-update delta-g PPO | 53.99% | 2 / 98 / 0 | 1 |
| 100-update pending-delay PPO | 53.90% | 3 / 97 / 0 | 1 |

The current PPO policies therefore remain FCFS-like on cases where departing
from FCFS is necessary.

## Position within the complete decision tree

All terminal leaves were exhaustively enumerated for the 100 hard cases with
branch-and-bound disabled. The trees contained:

- 2 terminal leaves at minimum;
- 15.93 terminal leaves on average;
- 8 terminal leaves at the median;
- 245 terminal leaves at maximum.

Define normalized tree quality:

```text
Q_tree = (J_worst - J_method) / (J_worst - J_exact)
```

Here 0% is the worst leaf and 100% is the optimal leaf.

| Method | Mean tree quality | Mean fraction of leaves strictly worse | Worst-leaf matches |
| --- | ---: | ---: | ---: |
| Uniformly selected leaf | 52.77% | 46.89% | 0 |
| FCFS | 70.88% | 61.31% | 7 |
| 100-update delta-g PPO | 71.06% | 61.41% | 7 |
| 600-update PPO | 71.06% | 61.41% | 7 |
| 100-update pending-delay PPO | 71.13% | 61.49% | 7 |

FCFS reduced cost relative to the worst leaf by 47.74% on average and 50.38%
at the median. Thus FCFS is not generally the worst path. PPO learned a
stable, reasonable policy, but its improvement over FCFS was negligible.

## What "learned" means in the current result

The actor parameters changed and entropy fell from roughly 0.69 to 0.15-0.20,
so optimization occurred and the policy became more deterministic. PPO also
ranked above a uniformly selected tree leaf.

The policy-level conclusion is nevertheless limited:

```text
PPO learned a FCFS-like heuristic;
it did not learn the hard non-FCFS priority decisions needed to approach exact.
```

Actor loss staying near zero is not proof of no learning because PPO
normalizes advantages per update. Held-out cost, FCFS win/tie/loss counts,
exact gap, and exact-match rate are the primary learning metrics.

## Why ordinary path-planning PPO is easier

Typical obstacle-avoidance problems provide:

- fixed action semantics such as steering or motion direction;
- many transitions per episode;
- immediate distance/collision feedback;
- local, repeated geometric patterns;
- a state that is close to Markov.

The current scheduler instead has:

- only about 2-3 genuine decisions per N=5 episode in the pilot;
- rare cases where a non-FCFS decision is useful;
- delayed global effects from an early priority choice;
- preemptive-repeat lost work;
- incomplete explicit representation of priority queues and future conflicts;
- branch actions whose physical consequences are not directly encoded.

## Recommended next model interface

Keep the exact event-driven transition model. Event-driven search is a valid
and efficient representation for exact scheduling; fixed time-step simulation
would create many no-op states.

Redefine the PPO layer as a contention-level SMDP:

1. A decision epoch occurs at a genuine resource contention.
2. The state explicitly includes:
   - contenders and requested intersection;
   - priority queue order, queue rank, and queue length;
   - request age and accumulated waiting;
   - current progress fraction and accumulated wasted work;
   - preemption count;
   - remaining route and future shared resources.
3. The action has one stable physical meaning:
   - select which contender receives priority;
   - optionally continue or preempt the currently running task.
4. The environment advances automatically to the next contention, completion,
   or terminal event.
5. Reward is the interval waiting cost plus lost progress at restart.

The actor should score the consequences of each legal action, for example:

```text
actor input = [parent state, child state - parent state, Delta g, Delta L]
```

This keeps the MLP architecture while exposing information needed to
distinguish hard priority choices.

## Supervised-learning option

If exact guidance is allowed, the most informative target is branch-level
exact cost-to-go:

```text
Q*(s, a) = minimum completion cost after taking branch a
```

The actor can learn a soft ranking such as `softmax(-Q*/temperature)`, and the
critic can regress exact value-to-go. PPO can then fine-tune after the
supervised loss is disabled.

This would be an exact-guided or imitation-assisted PPO model, not a pure PPO
model. The pure PPO result should remain as an ablation baseline.

Before large training, run two diagnostics:

1. attempt to overfit 10-20 fixed hard cases;
2. test whether a supervised exact-branch classifier can learn with the
   proposed state.

Failure to overfit indicates a state/action/reward interface problem. Success
to overfit but failure on held-out hard cases indicates a data-distribution or
generalization problem.

## Evaluation standard

Strictly beating FCFS on a majority of all random cases can be mathematically
impossible when FCFS is already exact-optimal in most cases. In the earlier
ordinary set, FCFS matched exact in 65 of 94 nontrivial cases.

The meaningful primary test is therefore an independently seeded
FCFS-suboptimal set. A useful pilot success criterion is:

- raw PPO strictly beats FCFS in at least 60 of 100 hard cases;
- PPO is worse in at most 5 cases;
- paired cost improvement has a confidence interval excluding zero;
- exact gap and exact-match rate improve materially.

For deployment, PPO and FCFS can both be evaluated and the lower-cost complete
schedule returned:

```text
J_safe = min(J_PPO, J_FCFS)
```

This guarantees performance no worse than FCFS, but the raw PPO result must
still be reported separately to demonstrate learning.

## Added analysis tools

- `PPO_model/plot_reward_ab_pilot.py`
  - plots smoothed reward A/B training and held-out results.
- `PPO_model/mine_fcfs_hard_cases.py`
  - mines independently selected FCFS-suboptimal cases and evaluates multiple
    PPO checkpoints.
- `PPO_model/evaluate_tree_position.py`
  - exhaustively enumerates terminal leaves and measures FCFS/PPO position
    between exact and worst.
