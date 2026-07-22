"""Command-line entry point for variable-N PPO training on the 3x3 map."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
import random
import time

import torch

from .cases import ThreeByThreeCaseFactory
from .encoding import BranchEncoder, EncodingConfig
from .evaluate import evaluate_against_exact
from .networks import BranchScoringActor, StateValueCritic
from .trainer import PPOConfig, PPOTrainer, finite_mean


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
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
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
        "--metrics-csv",
        default="output/ppo_n3/training_metrics.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.updates <= 0 or args.episodes_per_update <= 0:
        raise ValueError("updates and episodes-per-update must be positive")
    if args.n_robots <= 0:
        raise ValueError("--n-robots must be positive")

    random.seed(args.seed)
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
    )
    reference_case_factory = ThreeByThreeCaseFactory(
        seed=args.seed,
        randomize=False,
        n_robots=args.n_robots,
        max_initial_release=args.max_initial_release,
        min_initial_release=args.min_initial_release,
        initial_release_step=args.initial_release_step,
    )
    shape_case = reference_case_factory()
    shape_encoder = BranchEncoder(shape_case.plans, encoding_config)

    actor = BranchScoringActor(
        shape_encoder.state_dim,
        shape_encoder.action_dim,
        hidden_dim=args.hidden_dim,
    )
    critic = StateValueCritic(
        shape_encoder.state_dim,
        hidden_dim=args.hidden_dim,
    )
    ppo_config = PPOConfig(
        learning_rate=args.learning_rate,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        entropy_coef=args.entropy_coef,
        max_decisions_per_episode=args.max_decisions,
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
        f"updates={args.updates} episodes/update={args.episodes_per_update} "
        f"cases={'fixed' if args.fixed_case else 'random'}"
    )

    fieldnames = [
        "update",
        "episodes",
        "rollout_steps",
        "mean_cost",
        "mean_reward",
        "mean_decisions",
        "mean_branches",
        "actor_loss",
        "critic_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "fixed_greedy_cost",
        "elapsed_seconds",
    ]
    rows = []
    start_time = time.perf_counter()

    for update_index in range(1, args.updates + 1):
        rollout_stats = trainer.collect_rollouts(
            case_factory,
            args.episodes_per_update,
        )
        update_stats = trainer.update()
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
            "mean_reward": finite_mean([item.total_reward for item in rollout_stats]),
            "mean_decisions": finite_mean([item.decisions for item in rollout_stats]),
            "mean_branches": finite_mean([item.mean_branches for item in rollout_stats]),
            "actor_loss": update_stats.actor_loss,
            "critic_loss": update_stats.critic_loss,
            "entropy": update_stats.entropy,
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
            f"entropy={row['entropy']:.3f} kl={row['approx_kl']:.5f} "
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
