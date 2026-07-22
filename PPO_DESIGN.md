# PPO-Guided Tree Search Design Notes

This file records the PPO design decisions for the joint path-selection and
contention-resolving scheduling tree.  It is intentionally separate from the
ground-truth tree-search implementation.

## 1. Ground-truth objective

The exact tree-search objective is

```text
g = g_delay + g_path
```

where `g_delay` is accumulated scheduling delay and `g_path` is accumulated
extra free-flow travel time caused by path selection.  There is no separate
path-cost weight.

### Initial training scope and resource model

The first PPO environment uses the `3 x 3` traffic map.  Each of its nine
intersections is one indivisible shared resource.  The strict mutual-exclusion
rules are

```text
for every intersection i:  sum_n u[n, i](t) <= 1
for every robot n:         sum_i u[n, i](t) <= 1
```

Thus, at most one robot may occupy an intersection at a time, even if two
geometric movements through that intersection would otherwise be compatible.
The trajectory-compatibility/concurrent-movement extension is disabled for
this initial model.  Legal branches are generated under these constraints, so
PPO never receives a branch that violates intersection contention.

## 2. Node and edge timing

The current node `l(t_w)` stores the state produced by the previous edge
decision:

```text
l(t_{w-1}) -- a_{w-1} --> l(t_w) -- a_w --> l(t_{w+1})
```

Therefore, `U_temp` and the realized path-selection information stored in
`l(t_w)` describe the previous decision and its effect.  PPO observes this
current node and selects the next edge action `a_w`.

## 3. Task execution semantics

The environment uses preemptive-restart execution:

- A running task may be interrupted by the next scheduling decision.
- If it is selected again later, it restarts with its full execution time.
- It does not resume from the remaining execution time at interruption.

The current node's `U_temp`, together with `r_n(t_w)`, is sufficient to detect
whether a candidate next scheduling decision continues or interrupts a task.

No separate progress variable `rho_n` is required.  If normalization is useful
for neural-network training, compute it when building the observation:

```text
r_normalized = r_n / C_nk
```

This is a derived input feature, not additional environment state.  The full
task duration `C_nk` is obtained from the current task definition.

## 4. Waiting-task semantics

`U_temp` identifies which robot currently has access to a requested resource.
Robots that request the same resource but are not selected form an unordered
waiting set.  `U_temp` does not define an order among those waiting robots.

The PPO formulation therefore does not require a committed priority queue.
At the next event time, the environment generates the legal scheduling choices
again.

## 5. Path-selection variable

Path selection is event based.  The binary variable

```text
z_n,(i,j)(t_w) = 1
```

means that robot `n`, currently at intersection `i`, selects feasible next
intersection `j`.  It is a local next-intersection decision, not a one-shot
selection of an entire route.  The sequence of local `z` decisions determines
the realized route.

## 6. PPO state / observation

The environment may keep the complete tree node internally.  The first PPO
observation should contain the compact information that affects future
transitions and rewards.

For every robot `n`:

```text
d_n(t_w)       time until its next task is generated
r_n(t_w)       remaining time of its current task
o_n(t_w)       elapsed response time of its current task
ni_n(t_w)      current task index
U_request,n    currently requested resource, if any
U_temp,n       resource access stored in the current node
route state    current intersection/path prefix and feasible next choices
done_n         completed-robot indicator
```

Global information:

```text
t_w            current event time (normalized for the network)
decision mask  robots/resources that currently require a decision
```

The full histories `segments`, `attempts`, `path_decisions`, `idx`, and
`parent` are not Actor inputs.  Accumulated `g`, `g_delay`, and `g_path` are
also excluded when training with incremental rewards.

The full `alpha` and `gamma` matrices remain available to the environment for
cost calculation and reporting.  They do not need to be copied into the Actor
observation when the active-task timing is already represented by `d`, `r`,
and `o`.

## 7. PPO action and legal branches

One PPO action is one legal outgoing tree branch:

```text
a_w^(q) = (U_temp,w^(q), Z_w^(q))
```

`U_temp,w^(q)` is the candidate next scheduling decision and `Z_w^(q)` is the
candidate next-intersection decision.  Forced components may be included in
the complete encoding but are not independent choices.

The environment, not PPO, generates the legal branches.  It enforces resource
mutual exclusion and the one-hot path-selection constraint before the Actor
sees the candidates.  PPO therefore cannot sample an illegal or
contention-violating branch.

### Decision nodes and forced-transition nodes

Not every tree node is a PPO decision point.  A node with exactly one legal
child has no path-selection or scheduling choice.  Typical cases include:

- no active task, so time only advances to the next task-generation event;
- one task is the only request for a resource;
- no path split is currently available;
- all components of `(U_temp, Z)` are forced.

The environment automatically follows chains of one-child nodes.  The Actor is
called only at a node with at least two legal children.  After the Actor selects
one branch, the environment follows any subsequent forced transitions until it
reaches the next decision node or a terminal leaf.

For a PPO step starting at decision node `s_w` and ending at the next decision
node or leaf `s_v`, combine all skipped edge costs into one reward:

```text
reward = -(g_v - g_w).
```

Forced edges do not create Actor log-probability or Advantage entries.  On
environment reset, apply the same automatic advancement from the root to the
first decision node or leaf.

## 8. Actor architecture (initial design)

The number of legal children can vary by node.  Use one shared branch-scoring
network rather than a fixed output neuron with a changing meaning.

For each legal branch `q`, construct

```text
x_q = [current observation, encode(U_temp,w^(q)), encode(Z_w^(q))]
```

and score it with the same two-hidden-layer MLP:

```text
Linear(input_dim, 128)
Tanh
Linear(128, 128)
Tanh
Linear(128, 1)       -> branch logit
```

Softmax is applied across the logits of the legal branches at the current
node.  The resulting values are action-selection probabilities, not calibrated
probabilities that a branch is globally optimal.

The previous `U_temp` is part of the current observation.  A candidate next
`U_temp` is part of the candidate branch encoding.  Comparing them lets the
network learn the consequence of continuing or interrupting a running task.

## 9. Critic architecture (initial design)

The Critic receives only the current observation and outputs one scalar value:

```text
observation -> Linear(128) -> Tanh -> Linear(128) -> Tanh -> Linear(1)
```

Actor and Critic should initially use separate parameters so their behavior is
easy to inspect.

Use incremental reward

```text
reward_w = -(g_{w+1} - g_w)
```

with `discount_factor = 1.0`.  The state value is therefore the negative
expected remaining cost under the current policy:

```text
V^pi(s_w) = -E_pi[g_terminal - g_w | s_w]
```

It is not the negative accumulated cost `-g_w`, and it is not automatically
the exact optimal cost-to-go.  At a terminal leaf, `V(s_terminal) = 0` because
there is no remaining cost.  Train the Critic against sampled returns with a
mean-squared value loss.  No additional terminal reward is needed when all
cost increments have already been issued as step rewards.

In implementation, call the PPO discount parameter `discount_factor` rather
than `gamma` to avoid confusion with the task completion times `gamma_nk`.

## 10. Advantage definition

For a legal branch `b` at state `s`, let `Q_J^pi(s, b)` be the expected final
cost when branch `b` is selected now and the current policy is followed
afterward.  The current-policy cost baseline is the policy-weighted expectation

```text
J^pi(s) = sum_b pi(b | s) Q_J^pi(s, b).
```

This is not an unweighted arithmetic mean unless the Actor probabilities are
uniform.  Define the reward-oriented Advantage for this cost-minimization
problem as

```text
A^pi(s, b) = J^pi(s) - Q_J^pi(s, b).
```

Equivalently, using remaining cost and child state `s'`,

```text
A^pi(s, b)
    = J_rem^pi(s) - [delta_g(s, b) + J_rem^pi(s')].
```

Therefore:

- `A > 0`: the branch is expected to save total travel time relative to the
  current-policy baseline, so its probability should increase.
- `A < 0`: the branch is expected to add total travel time relative to the
  baseline, so its probability should decrease.

The unit of Advantage is time.  During PPO training, estimate this ideal
quantity from rollout rewards and Critic values using TD errors and GAE rather
than evaluating every branch outcome exactly.

## 11. Recommended code organization

Keep the exact tree search as the unchanged ground-truth baseline.  Add PPO in
separate files, for example:

```text
ppo_training/
    __init__.py
    environment.py     wraps node expansion and branch selection
    encoding.py        node/action encoding and normalization
    networks.py        branch-scoring Actor and state-value Critic
    rollout_buffer.py  PPO trajectories
    trainer.py         PPO loss and parameter updates
    train.py           training entry point
    evaluate.py        comparison with exact tree-search ground truth
```

The wrapper should call the existing legal child-generation logic.  It should
not duplicate or silently alter the scheduling/path-transition rules.  Changes
to the existing scheduler should be limited to small public interfaces only if
the wrapper cannot access the required operations cleanly.

## 12. Items still to finalize before implementation

The first fixed-`N=3` implementation now uses a 280-dimensional state encoding
and a 102-dimensional per-branch action encoding.  A leaf terminates an episode;
the implementation includes a safety cap on the number of policy decisions.
Random OD/release-time generation and exact-cost evaluation are available.

Future experimental choices still to finalize include:

- the long-run randomized training distribution and curriculum;
- the number of updates/episodes used for reported results;
- whether to add optimal-trajectory supervised pretraining;
- extension from fixed `N=3` to padded or set-encoded variable robot counts.
