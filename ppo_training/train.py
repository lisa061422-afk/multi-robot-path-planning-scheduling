"""Command-line entry point for variable-N PPO training on the 3x3 map."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
import random
import time
import math

import torch
import torch.nn.functional as F

from .cases import ThreeByThreeCaseFactory
from .encoding import BranchEncoder, EncodingConfig
from .evaluate import evaluate_against_exact
from .networks import BranchScoringActor, BranchScoringActorGNN, StateValueCritic
from .trainer import PPOConfig, PPOTrainer, finite_mean
from .environment import DecisionTreeEnv
from coarse_scheduler import search_dynamic_codesign_dfs_bb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train variable-branch PPO on the 3x3, N=3 co-design tree"
    )
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--episodes-per-update", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--n-robots", type=int, default=3)
    parser.add_argument(
        "--fixed-case",
        action="store_true",
        help="train repeatedly on one fixed case instead of random OD cases",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--critic-hidden-dim", type=int, default=None)
    parser.add_argument(
        "--actor-model",
        choices=["mlp", "gnn"],
        default="mlp",
        help="policy network type: mlp baseline or lightweight GNN-style branch scorer",
    )
    parser.add_argument(
        "--actor-hidden-layers",
        type=int,
        default=2,
        help="actor MLP hidden layer count (default matches legacy 2)",
    )
    parser.add_argument(
        "--critic-hidden-layers",
        type=int,
        default=2,
        help="critic MLP hidden layer count (legacy baseline = 2)",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--gnn-message-layers",
        type=int,
        default=2,
        help="message passing layers when using --actor-model gnn",
    )
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument(
        "--entropy-schedule",
        choices=["none", "linear"],
        default="none",
        help="optional entropy coefficient schedule",
    )
    parser.add_argument("--entropy-start", type=float, default=None)
    parser.add_argument("--entropy-end", type=float, default=None)
    parser.add_argument("--max-decisions", type=int, default=200)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument(
        "--exact-eval",
        action="store_true",
        help="run exact DFS on the fixed evaluation case after training",
    )
    parser.add_argument(
        "--checkpoint",
        default="output/ppo_n3/ppo_branch_actor.pt",
    )
    parser.add_argument("--min-initial-release", type=float, default=0.0)
    parser.add_argument("--max-initial-release", type=float, default=2.0)
    parser.add_argument("--initial-release-step", type=float, default=0.5)
    parser.add_argument(
        "--fix-shortest-paths",
        action="store_true",
        help=(
            "lock each vehicle to its shortest route option before training; "
            "only scheduling choices remain"
        ),
    )
    parser.add_argument(
        "--metrics-csv",
        default="output/ppo_n3/training_metrics.csv",
    )
    parser.add_argument("--bc-pretrain-episodes", type=int, default=0)
    parser.add_argument("--bc-pretrain-epochs", type=int, default=3)
    parser.add_argument("--bc-learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--reward-norm-mode",
        choices=["none", "absmax"],
        default="none",
        help=(
            "reward normalization mode used for training: "
            "none (raw), absmax (divide by max abs reward -> roughly [-1,1])"
        ),
    )
    parser.add_argument(
        "--reward-norm-eps",
        type=float,
        default=1e-12,
        help="epsilon used when reward normalization scale is near zero",
    )
    parser.add_argument(
        "--bc-mix-random",
        action="store_true",
        help=(
            "when set, include random cases in BC pretraining even when the main "
            "training stream is fixed-case"
        ),
    )
    parser.add_argument(
        "--lr-schedule",
        choices=["none", "linear"],
        default="none",
        help="optional learning-rate schedule",
    )
    parser.add_argument(
        "--discount-factor",
        type=float,
        default=1.0,
        help="discount factor gamma for PPO returns/GAE",
    )
    parser.add_argument("--lr-start", type=float, default=None)
    parser.add_argument("--lr-end", type=float, default=None)
    parser.add_argument(
        "--skip-trivial-cases",
        action="store_true",
        help="skip cases with no PPO decision branching (single-path/no-contention trees)",
    )
    parser.add_argument(
        "--max-case-attempts",
        type=int,
        default=2000,
        help="max case-sampling attempts when --skip-trivial-cases is enabled",
    )
    return parser.parse_args()


def _format_resource_tuple(values: tuple[int, ...]) -> str:
    if not values:
        return ""
    return "|".join(str(v) for v in values)


def _linear_schedule(
    update_index: int,
    total_updates: int,
    start_value: float,
    end_value: float,
) -> float:
    if total_updates <= 1:
        return start_value
    clipped = min(max(update_index - 1, 0), total_updates - 1)
    ratio = clipped / (total_updates - 1)
    return float(start_value + (end_value - start_value) * ratio)


def _node_signature(node: object) -> tuple:
    # Keep only stable, mostly hashable fields used by both exact solver and env nodes.
    return (
        round(float(node.tw), 12),
        round(float(node.g), 12),
        node.U_temp,
        node.path_decisions,
        node.route_candidates,
        node.segments,
    )


def _collect_bc_examples(
    actor: BranchScoringActor,
    plans,
    encoding_config: EncodingConfig,
    rng: random.Random,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
    env = DecisionTreeEnv(plans)
    exact = search_dynamic_codesign_dfs_bb(plans, verbose=False)
    if exact.best_idx < 0:
        return []

    exact_nodes = exact.nodes
    path: list[int] = []
    current = int(exact.best_idx)
    while current >= 0:
        path.append(current)
        current = exact_nodes[current].parent
    path = path[::-1]
    if len(path) <= 1:
        return []

    target_by_signature: dict[tuple, tuple] = {}
    signature_to_position: dict[tuple, int] = {}
    for position, node_idx in enumerate(path):
        sig = _node_signature(exact_nodes[node_idx])
        signature_to_position[sig] = position
        if position + 1 < len(path):
            target_by_signature[sig] = _node_signature(exact_nodes[path[position + 1]])

    encoder = BranchEncoder(plans, encoding_config)
    node, branches, terminated = env.reset()
    examples: list[tuple[torch.Tensor, torch.Tensor, int]] = []

    while not terminated:
        cur_sig = _node_signature(node)
        target_sig = target_by_signature.get(cur_sig)
        if target_sig is None:
            break

        action_index = None
        for index, child in enumerate(branches):
            if _node_signature(child) == target_sig:
                action_index = index
                break
        if action_index is None:
            break

        # Keep CPU tensors for lightweight storage; move to device during training.
        state_cpu = encoder.encode_state(node)
        branch_cpu = encoder.encode_actions(node, branches)
        examples.append((state_cpu, branch_cpu, action_index))

        step = env.step(action_index)
        node = step.node
        branches = step.branches
        terminated = step.terminated

    if not examples:
        return []
    # Add a tiny bit of random action jitter so BC never dies on a single mode.
    if rng.random() < 1e-3:
        idx = rng.randrange(len(examples))
        state_cpu, branch_cpu, action = examples[idx]
        if len(branch_cpu) > 1:
            examples[idx] = (state_cpu, branch_cpu, rng.randrange(len(branch_cpu)))

    # No-op shuffle to diversify without additional storage conversion.
    rng.shuffle(examples)
    return examples


def _run_behavior_cloning(
    actor: BranchScoringActor,
    examples: list[tuple[torch.Tensor, torch.Tensor, int]],
    device: torch.device,
    learning_rate: float,
    epochs: int,
    minibatch_size: int = 64,
) -> None:
    if not examples:
        return
    if learning_rate <= 0:
        raise ValueError("bc-learning-rate must be positive")
    optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    target_indices = list(range(len(examples)))
    actor.train()
    for epoch in range(1, max(1, epochs) + 1):
        random.shuffle(target_indices)
        batch_examples = [examples[i] for i in target_indices]
        for start in range(0, len(batch_examples), minibatch_size):
            batch = batch_examples[start : start + minibatch_size]
            if not batch:
                continue
            losses = []
            for state_cpu, branch_cpu, action in batch:
                state = state_cpu.to(device)
                branches = branch_cpu.to(device)
                logits = actor(state, branches)
                target = torch.tensor([action], dtype=torch.long, device=device)
                losses.append(F.cross_entropy(logits.unsqueeze(0), target))
            if not losses:
                continue
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


def main() -> None:
    args = parse_args()
    if args.updates <= 0 or args.episodes_per_update <= 0:
        raise ValueError("updates and episodes-per-update must be positive")
    if args.n_robots <= 0:
        raise ValueError("--n-robots must be positive")
    if args.hidden_dim <= 0:
        raise ValueError("--hidden-dim must be positive")
    if args.critic_hidden_dim is None:
        args.critic_hidden_dim = args.hidden_dim
    if args.critic_hidden_dim <= 0:
        raise ValueError("--critic-hidden-dim must be positive")
    if args.actor_hidden_layers <= 0:
        raise ValueError("--actor-hidden-layers must be positive")
    if args.actor_model == "gnn" and args.gnn_message_layers <= 0:
        raise ValueError("--gnn-message-layers must be positive")
    if args.critic_hidden_layers <= 0:
        raise ValueError("--critic-hidden-layers must be positive")
    if args.bc_pretrain_episodes < 0:
        raise ValueError("--bc-pretrain-episodes must be non-negative")
    if args.bc_pretrain_epochs < 0:
        raise ValueError("--bc-pretrain-epochs must be non-negative")
    if args.bc_learning_rate <= 0:
        raise ValueError("--bc-learning-rate must be positive")
    if args.reward_norm_eps <= 0:
        raise ValueError("--reward-norm-eps must be positive")

    entropy_schedule = args.entropy_schedule
    if entropy_schedule == "linear":
        entropy_start = (
            args.entropy_start
            if args.entropy_start is not None
            else args.entropy_coef
        )
        entropy_end = (
            args.entropy_end
            if args.entropy_end is not None
            else args.entropy_coef
        )
        if entropy_start <= 0 or entropy_end < 0:
            raise ValueError("entropy-start and entropy-end must be non-negative")
    else:
        entropy_start = entropy_end = args.entropy_coef

    lr_schedule = args.lr_schedule
    if lr_schedule == "linear":
        lr_start = args.lr_start if args.lr_start is not None else args.learning_rate
        lr_end = args.lr_end if args.lr_end is not None else args.learning_rate
        if not (math.isfinite(lr_start) and math.isfinite(lr_end) and lr_start > 0 and lr_end > 0):
            raise ValueError("lr-start and lr-end must be positive finite values")
    else:
        lr_start = lr_end = args.learning_rate

    effective_learning_rate = args.learning_rate

    random.seed(args.seed)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, int(args.torch_threads)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoding_config = EncodingConfig(n_robots=args.n_robots, n_resources=9, n_ports=12)
    case_factory = ThreeByThreeCaseFactory(
        seed=args.seed,
        randomize=not args.fixed_case,
        n_robots=args.n_robots,
        max_initial_release=args.max_initial_release,
        min_initial_release=args.min_initial_release,
        initial_release_step=args.initial_release_step,
        fix_shortest_paths=args.fix_shortest_paths,
    )
    reference_case_factory = ThreeByThreeCaseFactory(
        seed=args.seed,
        randomize=False,
        n_robots=args.n_robots,
        max_initial_release=args.max_initial_release,
        min_initial_release=args.min_initial_release,
        initial_release_step=args.initial_release_step,
        fix_shortest_paths=args.fix_shortest_paths,
    )
    shape_case = reference_case_factory()
    shape_encoder = BranchEncoder(shape_case.plans, encoding_config)

    if args.actor_model == "gnn":
        actor = BranchScoringActorGNN(
            shape_encoder.state_dim,
            shape_encoder.action_dim,
            n_robots=args.n_robots,
            hidden_dim=args.hidden_dim,
            hidden_layers=args.actor_hidden_layers,
            message_layers=args.gnn_message_layers,
        )
    else:
        actor = BranchScoringActor(
            shape_encoder.state_dim,
            shape_encoder.action_dim,
            hidden_dim=args.hidden_dim,
            hidden_layers=args.actor_hidden_layers,
        )
    critic = StateValueCritic(
        shape_encoder.state_dim,
        hidden_dim=args.critic_hidden_dim,
        hidden_layers=args.critic_hidden_layers,
    )
    ppo_config = PPOConfig(
        discount_factor=args.discount_factor,
        learning_rate=args.learning_rate,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        entropy_coef=args.entropy_coef,
        max_decisions_per_episode=args.max_decisions,
        reward_norm_mode=args.reward_norm_mode,
        reward_norm_minmax_eps=args.reward_norm_eps,
    )
    trainer = PPOTrainer(
        actor,
        critic,
        encoding_config=encoding_config,
        config=ppo_config,
        device=device,
        seed=args.seed,
    )

    checkpoint_path = Path(args.checkpoint)
    metrics_path = Path(args.metrics_csv)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        "PPO training: "
        f"map=3x3 N={args.n_robots} "
        f"alpha0=[{args.min_initial_release},{args.max_initial_release}] "
        f"step={args.initial_release_step} "
        f"device={device} threads={torch.get_num_threads()} "
        f"state_dim={shape_encoder.state_dim} action_dim={shape_encoder.action_dim}"
    )
    print(
        f"discount_factor={args.discount_factor} gae_lambda={trainer.config.gae_lambda}"
    )
    print(
        "reward normalization: "
        f"mode={args.reward_norm_mode} eps={args.reward_norm_eps}"
    )
    print(
        f"updates={args.updates} episodes/update={args.episodes_per_update} "
        f"cases={'fixed' if args.fixed_case else 'random'}"
    )

    fieldnames = [
        "update",
        "episodes",
        "rollout_steps",
        "mean_cost",
        "mean_reward",
        "mean_reward_raw",
        "mean_decisions",
        "mean_branches",
        "actor_loss",
        "critic_loss",
        "entropy",
        "scheduled_entropy",
        "learning_rate",
        "approx_kl",
        "clip_fraction",
        "fixed_greedy_cost",
        "mean_contention_records",
        "mean_contention_pairs",
        "contention_case_rate",
        "elapsed_seconds",
    ]
    rows = []
    contention_rows_path = metrics_path.with_name(
        f"{metrics_path.stem}_case_contentions.csv"
    )
    contention_fields = [
        "update",
        "episode_in_update",
        "case_name",
        "contention_records",
        "contention_pairs",
        "contention_record_count",
    ]
    contention_rows_path.parent.mkdir(parents=True, exist_ok=True)
    with contention_rows_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=contention_fields).writeheader()
    start_time = time.perf_counter()

    # Optional BC warm-start from exact optimal decision trajectories.
    if args.bc_pretrain_episodes > 0:
        print(
            "behavior cloning warm-start: "
            f"episodes={args.bc_pretrain_episodes} "
            f"epochs={args.bc_pretrain_epochs} "
            f"lr={args.bc_learning_rate}"
        )
        all_examples: list[tuple[torch.Tensor, torch.Tensor, int]] = []
        bc_case_factory = case_factory
        if args.bc_mix_random:
            bc_case_factory = ThreeByThreeCaseFactory(
                seed=args.seed + 9999,
                randomize=True,
                n_robots=args.n_robots,
                max_initial_release=args.max_initial_release,
                min_initial_release=args.min_initial_release,
                initial_release_step=args.initial_release_step,
                fix_shortest_paths=args.fix_shortest_paths,
            )
        for ep in range(args.bc_pretrain_episodes):
            case = bc_case_factory()
            plans = getattr(case, "plans", case)
            all_examples.extend(
                _collect_bc_examples(
                    actor,
                    plans,
                    encoding_config=encoding_config,
                    rng=rng,
                    device=device,
                )
            )
        _run_behavior_cloning(
            actor=actor,
            examples=all_examples,
            device=device,
            learning_rate=args.bc_learning_rate,
            epochs=args.bc_pretrain_epochs,
            minibatch_size=args.minibatch_size,
        )
        print(f"BC pretrain collected {len(all_examples)} state-action examples")
        trainer.save_checkpoint(
            str(checkpoint_path),
            extra={
                "completed_updates": 0,
                "completed_episodes": 0,
                "bc_pretrain_episodes": args.bc_pretrain_episodes,
                "bc_pretrain_examples": len(all_examples),
            },
        )

    for update_index in range(1, args.updates + 1):
        current_entropy = _linear_schedule(
            update_index,
            args.updates,
            entropy_start,
            entropy_end,
        ) if entropy_schedule == "linear" else args.entropy_coef
        effective_learning_rate = (
            _linear_schedule(
                update_index,
                args.updates,
                lr_start,
                lr_end,
            )
            if lr_schedule == "linear"
            else args.learning_rate
        )
        if lr_schedule == "linear":
            trainer.set_learning_rates(effective_learning_rate)
        rollout_stats = trainer.collect_rollouts(
            case_factory,
            args.episodes_per_update,
            skip_trivial=args.skip_trivial_cases,
            max_attempts=args.max_case_attempts,
        )
        with contention_rows_path.open("a", newline="", encoding="utf-8") as handle:
            content_writer = csv.DictWriter(handle, fieldnames=contention_fields)
            for local_ep, stat in enumerate(rollout_stats, start=1):
                content_writer.writerow(
                    {
                        "update": update_index,
                        "episode_in_update": local_ep,
                        "case_name": stat.case_name,
                        "contention_records": _format_resource_tuple(
                            stat.contention_records
                        ),
                        "contention_pairs": stat.contention_pair_count,
                        "contention_record_count": len(stat.contention_records),
                    }
                )
        update_stats = trainer.update_with_entropy(current_entropy)
        fixed_case = reference_case_factory()
        fixed_eval = evaluate_against_exact(
            actor,
            fixed_case.plans,
            encoding_config=encoding_config,
            device=device,
            run_exact=False,
        )
        elapsed = time.perf_counter() - start_time
        row = {
            "update": update_index,
            "episodes": update_index * args.episodes_per_update,
            "rollout_steps": update_stats.rollout_steps,
            "mean_cost": finite_mean([item.total_cost for item in rollout_stats]),
            "mean_reward": finite_mean([item.total_reward_normalized for item in rollout_stats]),
            "mean_reward_raw": finite_mean([item.total_reward for item in rollout_stats]),
            "mean_decisions": finite_mean([item.decisions for item in rollout_stats]),
            "mean_branches": finite_mean([item.mean_branches for item in rollout_stats]),
            "mean_contention_records": finite_mean(
                [len(item.contention_records) for item in rollout_stats]
            ),
            "mean_contention_pairs": finite_mean(
                [item.contention_pair_count for item in rollout_stats]
            ),
            "contention_case_rate": finite_mean(
                [1.0 if item.contention_pair_count > 0 else 0.0 for item in rollout_stats]
            ),
            "actor_loss": update_stats.actor_loss,
            "critic_loss": update_stats.critic_loss,
            "entropy": update_stats.entropy,
            "scheduled_entropy": current_entropy,
            "learning_rate": effective_learning_rate,
            "approx_kl": update_stats.approx_kl,
            "clip_fraction": update_stats.clip_fraction,
            "fixed_greedy_cost": fixed_eval.ppo_cost,
            "elapsed_seconds": elapsed,
        }
        rows.append(row)
        print(
            f"[{update_index:04d}] episodes={row['episodes']} "
                f"steps={row['rollout_steps']} mean_J={row['mean_cost']:.3f} "
                f"fixed_J={row['fixed_greedy_cost']:.3f} "
                f"entropy={row['entropy']:.3f} "
                f"sched_ent={row['scheduled_entropy']:.3f} "
                f"lr={row['learning_rate']:.1e} "
                f"kl={row['approx_kl']:.5f} "
                f"elapsed={elapsed:.1f}s"
        )

        trainer.save_checkpoint(
            str(checkpoint_path),
            extra={
                "completed_updates": update_index,
                "completed_episodes": update_index * args.episodes_per_update,
                "fixed_greedy_cost": fixed_eval.ppo_cost,
            },
        )
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    final_case = reference_case_factory()
    final_eval = evaluate_against_exact(
        actor,
        final_case.plans,
        encoding_config=encoding_config,
        device=device,
        run_exact=args.exact_eval,
    )
    print(f"checkpoint: {checkpoint_path}")
    print(f"metrics: {metrics_path}")
    print(f"fixed-case PPO cost: {final_eval.ppo_cost:.6f}")
    if final_eval.exact_cost is not None:
        print(
            f"fixed-case exact cost: {final_eval.exact_cost:.6f}; "
            f"absolute gap={final_eval.absolute_gap:.6f}; "
            f"relative gap={final_eval.relative_gap:.3%}"
        )


if __name__ == "__main__":
    main()
