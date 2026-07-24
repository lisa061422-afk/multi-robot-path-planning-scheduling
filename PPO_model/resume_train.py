"""Resume PPO training from an existing checkpoint and continue updates.

This script keeps the same model architecture and PPO configuration from the
source checkpoint, then runs additional on-policy updates from fresh random cases.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import random
import time
from typing import Sequence

import torch

from .cases import ThreeByThreeCaseFactory
from .encoding import EncodingConfig
from .evaluate import evaluate_against_exact
from .networks import BranchScoringActor, BranchScoringActorGNN, StateValueCritic
from .trainer import PPOConfig, PPOTrainer, finite_mean


def _default_run_id() -> str:
    return f"resumed_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


def _parse_group_robots(spec: str) -> list[int]:
    if not spec:
        return []
    values: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo_str, hi_str = token.split("-", 1)
            lo_i = int(lo_str.strip())
            hi_i = int(hi_str.strip())
            if lo_i <= 0 or hi_i <= 0:
                raise ValueError(f"robot counts in --group-robots must be positive: '{token}'")
            if hi_i < lo_i:
                raise ValueError(f"invalid range in --group-robots: '{token}'")
            values.extend(range(lo_i, hi_i + 1))
        else:
            value = int(token)
            if value <= 0:
                raise ValueError(f"robot counts in --group-robots must be positive: '{token}'")
            values.append(value)
    # keep stable unique order
    seen = set()
    out: list[int] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _load_networks(
    checkpoint_path: Path,
    n_robots_fallback: int,
    *,
    device: torch.device,
):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    actor_type = payload.get("actor_type", "BranchScoringActor")

    if actor_type == "BranchScoringActorGNN":
        actor = BranchScoringActorGNN(
            int(payload["state_dim"]),
            int(payload["action_dim"]),
            n_robots=int(payload.get("actor_n_robots", n_robots_fallback)),
            hidden_dim=int(payload["hidden_dim"]),
            hidden_layers=int(payload.get("actor_hidden_layers", 2)),
            message_layers=int(payload.get("actor_gnn_message_layers", 2)),
        )
    else:
        actor = BranchScoringActor(
            int(payload["state_dim"]),
            int(payload["action_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            hidden_layers=int(payload.get("actor_hidden_layers", 2)),
        )

    actor.load_state_dict(payload["actor_state_dict"])
    actor.to(device)

    critic = StateValueCritic(
        int(payload["state_dim"]),
        hidden_dim=int(payload["critic_hidden_dim"]),
        hidden_layers=int(payload.get("critic_hidden_layers", 2)),
    )
    critic.load_state_dict(payload["critic_state_dict"])
    critic.to(device)

    encoding = EncodingConfig(**payload["encoding_config"])
    ppo_config = PPOConfig(**payload["ppo_config"])

    completed_updates = int(
        (payload.get("extra", {}) or {}).get("completed_updates", 0)
    )
    completed_episodes = int(
        (payload.get("extra", {}) or {}).get("completed_episodes", 0)
    )

    return actor, critic, encoding, ppo_config, completed_updates, completed_episodes


def _resume_one_n(
    n_robots: int,
    source_path: Path,
    *,
    target_run_id: str,
    run_root: Path,
    seed: int,
    updates: int,
    episodes_per_update: int,
    max_decisions: int,
    max_case_attempts: int,
    skip_trivial_cases: bool,
    device: torch.device,
    fix_shortest_paths: bool,
    max_vehicles_per_entrance: int,
    min_initial_release: float,
    max_initial_release: float,
    initial_release_step: float,
    extra_prefix: str,
    torch_threads: int,
) -> None:
    if updates <= 0 or episodes_per_update <= 0:
        raise ValueError("updates and episodes-per-update must be positive")
    if max_vehicles_per_entrance < 0:
        raise ValueError("--max-vehicles-per-entrance must be non-negative")
    if initial_release_step <= 0:
        raise ValueError("--initial-release-step must be positive")

    n_seed = int(seed + n_robots * 97)
    random.seed(n_seed)
    torch.manual_seed(n_seed)

    actor, critic, encoding_config, ppo_config, completed_updates, completed_episodes = _load_networks(
        source_path,
        n_robots,
        device=device,
    )

    trainer = PPOTrainer(
        actor,
        critic,
        encoding_config=encoding_config,
        config=ppo_config,
        device=device,
        seed=n_seed,
    )
    # continue optimizer states where available
    source_payload = torch.load(source_path, map_location=device, weights_only=False)
    trainer.actor_optimizer.load_state_dict(source_payload["actor_optimizer_state_dict"])
    trainer.critic_optimizer.load_state_dict(source_payload["critic_optimizer_state_dict"])

    case_factory = ThreeByThreeCaseFactory(
        seed=n_seed,
        randomize=True,
        n_robots=n_robots,
        max_vehicles_per_entrance=max_vehicles_per_entrance,
        max_initial_release=max_initial_release,
        min_initial_release=min_initial_release,
        initial_release_step=initial_release_step,
        fix_shortest_paths=fix_shortest_paths,
    )

    reference_case_factory = ThreeByThreeCaseFactory(
        seed=n_seed,
        randomize=False,
        n_robots=n_robots,
        max_vehicles_per_entrance=max_vehicles_per_entrance,
        max_initial_release=max_initial_release,
        min_initial_release=min_initial_release,
        initial_release_step=initial_release_step,
        fix_shortest_paths=fix_shortest_paths,
    )

    run_dir = run_root / target_run_id / f"n{n_robots}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "ppo_branch_actor.pt"
    metrics_path = run_dir / "training_metrics.csv"
    contention_path = run_dir / "training_metrics_case_contentions.csv"

    with contention_path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(
            [
                "n_robots",
                "update",
                "episode_in_update",
                "case_name",
                "contention_records",
                "contention_pairs",
                "contention_record_count",
            ]
        )

    # If target is new, write header. If user manually runs again, append.
    fieldnames = [
        "n_robots",
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

    existing_rows: list[dict[str, float]] = []
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            existing_rows = [dict(row) for row in csv.DictReader(handle)]

    start_update = completed_updates + 1
    end_update = completed_updates + updates
    start = time.perf_counter()

    print(
        f"=== RESUME n={n_robots} === "
        f"source={source_path} updates={start_update}-{end_update}"
    )
    print(
        f"state_dim={actor.state_dim} action_dim={actor.action_dim} "
        f"seed={n_seed}"
    )

    for update_index in range(start_update, end_update + 1):
        rollout_stats = trainer.collect_rollouts(
            case_factory,
            episodes_per_update,
            skip_trivial=skip_trivial_cases,
            max_attempts=max_case_attempts,
        )

        with contention_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            for local_ep, stat in enumerate(rollout_stats, start=1):
                writer.writerow(
                    [
                        n_robots,
                        update_index,
                        local_ep,
                        stat.case_name,
                        "|".join(str(v) for v in stat.contention_records),
                        stat.contention_pair_count,
                        len(stat.contention_records),
                    ]
                )

        update_stats = trainer.update_with_entropy(ppo_config.entropy_coef)
        fixed_case = reference_case_factory()
        fixed_eval = evaluate_against_exact(
            actor,
            fixed_case.plans,
            encoding_config=encoding_config,
            device=device,
            run_exact=False,
        )
        elapsed = time.perf_counter() - start
        row = {
            "n_robots": n_robots,
            "update": update_index,
            "episodes": update_index * episodes_per_update,
            "rollout_steps": update_stats.rollout_steps,
            "mean_cost": finite_mean([item.total_cost for item in rollout_stats]),
            "mean_reward": finite_mean(
                [item.total_reward_normalized for item in rollout_stats]
            ),
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
                [
                    1.0 if item.contention_pair_count > 0 else 0.0
                    for item in rollout_stats
                ]
            ),
            "actor_loss": update_stats.actor_loss,
            "critic_loss": update_stats.critic_loss,
            "entropy": update_stats.entropy,
            "scheduled_entropy": ppo_config.entropy_coef,
            "learning_rate": ppo_config.learning_rate,
            "approx_kl": update_stats.approx_kl,
            "clip_fraction": update_stats.clip_fraction,
            "fixed_greedy_cost": fixed_eval.ppo_cost,
            "elapsed_seconds": elapsed,
        }
        existing_rows.append(row)
        print(
            f"[N={n_robots}] [{update_index:04d}] "
            f"mean_J={row['mean_cost']:.3f} fixed_J={row['fixed_greedy_cost']:.3f} "
            f"entropy={row['entropy']:.3f} kl={row['approx_kl']:.5f} "
            f"elapsed={elapsed:.1f}s"
        )

        trainer.save_checkpoint(
            str(checkpoint_path),
            extra={
                "completed_updates": update_index,
                "completed_episodes": update_index * episodes_per_update,
                "fixed_greedy_cost": row["fixed_greedy_cost"],
            },
        )
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_rows)

    final_case = reference_case_factory()
    final_eval = evaluate_against_exact(
        actor,
        final_case.plans,
        encoding_config=encoding_config,
        device=device,
        run_exact=False,
    )
    print(f"checkpoint: {checkpoint_path}")
    print(f"metrics: {metrics_path}")
    print(f"final fixed-case PPO cost (N={n_robots}): {final_eval.ppo_cost:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continue PPO training from an existing checkpoint"
    )
    parser.add_argument(
        "--source-run-root",
        default="output/ppo_runs",
        help="root directory containing the source checkpoint run",
    )
    parser.add_argument(
        "--source-run-id",
        required=True,
        help="run id containing source checkpoints, e.g. fast_parallel_20260723_211520",
    )
    parser.add_argument(
        "--source-n-robots",
        required=True,
        help="comma/range list of source robot counts, e.g. '2-5,6-8,9-12'",
    )
    parser.add_argument(
        "--target-run-id",
        default="",
        help="run id for resumed outputs; defaults to timestamp",
    )
    parser.add_argument(
        "--run-root",
        default="output/ppo_runs",
        help="root directory for resumed outputs",
    )
    parser.add_argument("--updates", type=int, default=100, help="additional updates")
    parser.add_argument("--episodes-per-update", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--skip-trivial-cases", action="store_true")
    parser.add_argument("--max-case-attempts", type=int, default=2000)
    parser.add_argument("--max-decisions", type=int, default=500)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--fix-shortest-paths", action="store_true")
    parser.add_argument("--max-vehicles-per-entrance", type=int, default=1)
    parser.add_argument("--min-initial-release", type=float, default=0.0)
    parser.add_argument("--max-initial-release", type=float, default=5.0)
    parser.add_argument("--initial-release-step", type=float, default=0.5)
    parser.add_argument("--extra-prefix", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.updates <= 0 or args.episodes_per_update <= 0:
        raise ValueError("updates and episodes-per-update must be positive")
    if args.max_case_attempts <= 0:
        raise ValueError("--max-case-attempts must be positive")

    group = _parse_group_robots(args.source_n_robots)
    if not group:
        raise ValueError("--source-n-robots must specify at least one value")

    torch.set_num_threads(max(1, int(args.torch_threads)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_run_id = args.target_run_id.strip() or _default_run_id()
    run_root = Path(args.run_root)

    print(f"resume run id: {target_run_id}")
    print(f"resume run root: {run_root}")

    source_root = Path(args.source_run_root)
    source_root.mkdir(parents=True, exist_ok=True)

    for n_robots in group:
        source_checkpoint = (
            source_root / args.source_run_id / f"n{n_robots}" / "ppo_branch_actor.pt"
        )
        if not source_checkpoint.exists():
            print(f"warning: missing source checkpoint: {source_checkpoint}, skip n={n_robots}")
            continue

        # Keep the same output naming style as train.py.
        _resume_one_n(
            n_robots,
            source_checkpoint,
            target_run_id=target_run_id,
            run_root=run_root,
            seed=args.seed,
            updates=args.updates,
            episodes_per_update=args.episodes_per_update,
            max_decisions=args.max_decisions,
            max_case_attempts=args.max_case_attempts,
            skip_trivial_cases=args.skip_trivial_cases,
            device=device,
            fix_shortest_paths=args.fix_shortest_paths,
            max_vehicles_per_entrance=args.max_vehicles_per_entrance,
            min_initial_release=args.min_initial_release,
            max_initial_release=args.max_initial_release,
            initial_release_step=args.initial_release_step,
            extra_prefix=args.extra_prefix,
            torch_threads=args.torch_threads,
        )


if __name__ == "__main__":
    main()
