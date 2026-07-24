"""Command-line entry point for variable-N PPO training on the 3x3 map."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import random
import re
import time
import math

import torch

from .cases import ThreeByThreeCaseFactory
from .encoding import BranchEncoder, EncodingConfig
from .evaluate import evaluate_against_exact
from .networks import BranchScoringActor, StateValueCritic
from .trainer import PPOConfig, PPOTrainer, finite_mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train variable-branch PPO on the 3x3 co-design tree"
    )
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--episodes-per-update", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--n-robots", type=int, default=3)
    parser.add_argument(
        "--group-robots",
        default="",
        help=(
            "optional comma- or range-separated robot-count groups to train "
            "sequentially, e.g. '2-4,6' or '2,3,4'. If set, it overrides --n-robots."
        ),
    )
    parser.add_argument(
        "--fixed-case",
        action="store_true",
        help="train repeatedly on one fixed case instead of random OD cases",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--critic-hidden-dim", type=int, default=None)
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
    parser.add_argument("--max-initial-release", type=float, default=5.0)
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
        "--max-vehicles-per-entrance",
        type=int,
        default=1,
        help=(
            "limit how many vehicles can share the same entrance in random generation; "
            "0 means no limit"
        ),
    )
    parser.add_argument(
        "--metrics-csv",
        default="output/ppo_n3/training_metrics.csv",
    )
    parser.add_argument(
        "--reward-cost-mode",
        choices=["delta_g", "pending_delay"],
        default="delta_g",
        help=(
            "cost used for one-step rewards: delta_g uses booked objective "
            "increments; pending_delay adds an exact potential that accrues "
            "waiting cost before task completion"
        ),
    )
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
        help=(
            "skip cases with no PPO branching decisions throughout the whole episode "
            "(no internal branching, i.e., single-path case)"
        ),
    )
    parser.add_argument(
        "--max-case-attempts",
        type=int,
        default=2000,
        help="max case-sampling attempts when --skip-trivial-cases is enabled",
    )
    parser.add_argument(
        "--plot-after-train",
        action="store_true",
        help="plot selected training metrics automatically when training finishes",
    )
    parser.add_argument(
        "--plots-dir",
        default="",
        help=(
            "directory for saved training plots; if empty, defaults to "
            "<metrics-dir>/plots; tokens {n_robots}/ {n} are supported"
        ),
    )
    parser.add_argument(
        "--run-root",
        default="output/ppo_runs",
        help=(
            "root directory for new training runs; when using default checkpoint/metrics "
            "paths, files are placed under <run-root>/<run-id>/n<robots>/..."
        ),
    )
    parser.add_argument(
        "--run-id",
        default="",
        help=(
            "run id used to create a fresh experiment folder; defaults to "
            "timestamp run_YYYYMMDD_HHMMSS_ffffff"
        ),
    )
    return parser.parse_args()


def _parse_float(value: str) -> float:
    if value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _load_metrics_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row: dict[str, float] = {}
            for key, value in raw.items():
                if key in {"update", "episodes", "rollout_steps"}:
                    try:
                        row[key] = int(value)
                    except (TypeError, ValueError):
                        row[key] = float("nan")
                else:
                    row[key] = _parse_float(value)
            rows.append(row)
    return rows


def _plot_training_curves(metrics_path: Path, plot_dir: Path) -> Path:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required to generate plots. Install it with "
            "`pip install matplotlib`."
        ) from exc

    rows = _load_metrics_rows(metrics_path)
    if not rows:
        raise RuntimeError(f"no rows found in metrics CSV: {metrics_path}")

    updates = [row.get("update", i + 1) for i, row in enumerate(rows)]
    plot_items: list[tuple[str, str, str]] = [
        ("mean_cost", "Mean rollout cost", "cost"),
        ("fixed_greedy_cost", "Fixed-case greedy cost", "cost"),
        ("mean_reward_raw", "Mean raw reward", "reward"),
        ("mean_reward", "Mean normalized reward", "reward"),
        ("mean_decisions", "Mean decisions", "count"),
        ("mean_branches", "Mean branching factor", "count"),
        ("actor_loss", "Actor loss", "loss"),
        ("critic_loss", "Critic loss", "loss"),
        ("entropy", "Policy entropy", "entropy"),
        ("scheduled_entropy", "Scheduled entropy", "entropy"),
        ("approx_kl", "Approx KL", "KL"),
        ("clip_fraction", "Clip fraction", "fraction"),
        ("learning_rate", "Learning rate", "LR"),
        ("mean_contention_records", "Mean contention records", "count"),
        ("mean_contention_pairs", "Mean contention pairs", "count"),
        ("contention_case_rate", "Contention case rate", "rate"),
    ]

    plot_items = [
        item for item in plot_items
        if any((row.get(item[0]) is not None and row.get(item[0]) == row.get(item[0])) for row in rows)
    ]
    if not plot_items:
        raise RuntimeError(f"no plottable numeric columns found in {metrics_path}")

    n_cols = 3
    n_rows = (len(plot_items) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.0 * n_cols, 3.0 * n_rows),
        sharex=True,
    )
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for axis in axes_flat[len(plot_items):]:
        axis.set_axis_off()

    for idx, (key, title, ylabel) in enumerate(plot_items):
        axis = axes_flat[idx]
        y = [float(row.get(key, float("nan"))) for row in rows]
        axis.plot(updates, y, marker="o", linewidth=1.2)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        if ylabel in {"cost", "reward", "loss", "count", "KL", "fraction", "rate", "entropy", "LR"}:
            axis.set_xlabel("update")
        else:
            axis.set_xlabel("update")

    fig.suptitle(f"PPO training curves ({metrics_path.stem})")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_path = plot_dir / f"{metrics_path.stem}_training_curves.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _format_resource_tuple(values: tuple[int, ...]) -> str:
    if not values:
        return ""
    return "|".join(str(v) for v in values)


def _parse_group_robots(spec: str) -> list[int]:
    if not spec:
        return []
    values: list[int] = []
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        if "-" in token:
            lo, hi = token.split("-", 1)
            lo_i = int(lo.strip())
            hi_i = int(hi.strip())
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
    if not values:
        return []
    # Stable unique ordering (preserve first appearance).
    unique: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _resolve_group_path(
    path_str: str,
    n_robots: int,
    *,
    run_id: str = "",
    run_root: str | Path = "",
) -> Path:
    if "{" in path_str and "}" in path_str:
        try:
            return Path(
                path_str.format(
                    n_robots=n_robots,
                    n=n_robots,
                    run_id=run_id,
                    run_root=str(run_root),
                )
            )
        except KeyError:
            pass
        except IndexError:
            pass
    if "{n_robots}" in path_str or "{n}" in path_str:
        return Path(path_str.format(n_robots=n_robots, n=n_robots, run_id=run_id, run_root=run_root))
    base = Path(path_str)
    parent = base.parent
    if parent.name:
        parent_name = re.sub(r"n\d+", f"n{n_robots}", parent.name, count=1)
        parent = parent.with_name(parent_name)
    return parent / base.name


def _default_run_id() -> str:
    return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


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


def _run_training_for_n(
    n_robots: int,
    *,
    args: argparse.Namespace,
    run_id: str,
    run_root: str | Path,
    device: torch.device,
    schedule: dict[str, float],
    entropy_schedule: str,
    lr_schedule: str,
    run_index: int = 0,
    total_runs: int = 1,
) -> None:
    run_seed = int(args.seed + run_index * 10007 + n_robots * 101)
    random.seed(run_seed)
    rng = random.Random(run_seed)
    torch.manual_seed(run_seed)

    encoding_config = EncodingConfig(n_robots=n_robots, n_resources=9, n_ports=12)
    case_factory = ThreeByThreeCaseFactory(
        seed=run_seed,
        randomize=not args.fixed_case,
        n_robots=n_robots,
        max_vehicles_per_entrance=args.max_vehicles_per_entrance,
        max_initial_release=args.max_initial_release,
        min_initial_release=args.min_initial_release,
        initial_release_step=args.initial_release_step,
        fix_shortest_paths=args.fix_shortest_paths,
    )
    reference_case_factory = ThreeByThreeCaseFactory(
        seed=run_seed,
        randomize=False,
        n_robots=n_robots,
        max_vehicles_per_entrance=args.max_vehicles_per_entrance,
        max_initial_release=args.max_initial_release,
        min_initial_release=args.min_initial_release,
        initial_release_step=args.initial_release_step,
        fix_shortest_paths=args.fix_shortest_paths,
    )
    shape_case = reference_case_factory()
    shape_encoder = BranchEncoder(shape_case.plans, encoding_config)

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
        reward_cost_mode=args.reward_cost_mode,
        reward_norm_mode=args.reward_norm_mode,
        reward_norm_minmax_eps=args.reward_norm_eps,
    )
    trainer = PPOTrainer(
        actor,
        critic,
        encoding_config=encoding_config,
        config=ppo_config,
        device=device,
        seed=run_seed,
    )

    checkpoint_path = _resolve_group_path(
        args.checkpoint, n_robots, run_id=run_id, run_root=run_root
    )
    metrics_path = _resolve_group_path(
        args.metrics_csv, n_robots, run_id=run_id, run_root=run_root
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"=== PPO training group {run_index + 1}/{total_runs} === "
        f"map=3x3 N={n_robots} "
        f"alpha0=[{args.min_initial_release},{args.max_initial_release}] "
        f"step={args.initial_release_step} "
        f"state_dim={shape_encoder.state_dim} action_dim={shape_encoder.action_dim}"
    )
    print(
        f"discount_factor={args.discount_factor} gae_lambda={trainer.config.gae_lambda}"
    )
    print(
        "reward: "
        f"cost_mode={args.reward_cost_mode} "
        f"normalization={args.reward_norm_mode} eps={args.reward_norm_eps}"
    )
    print(
        f"updates={args.updates} episodes/update={args.episodes_per_update} "
        f"seed={run_seed} "
        f"cases={'fixed' if args.fixed_case else 'random'}"
    )

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
    rows = []
    contention_rows_path = metrics_path.with_name(
        f"{metrics_path.stem}_case_contentions.csv"
    )
    contention_fields = [
        "n_robots",
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

    entropy_start, entropy_end = schedule["entropy_start"], schedule["entropy_end"]
    lr_start, lr_end = schedule["lr_start"], schedule["lr_end"]
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
                        "n_robots": n_robots,
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
            "n_robots": n_robots,
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
            f"[N={n_robots}] [{update_index:04d}] "
            f"episodes={row['episodes']} steps={row['rollout_steps']} "
            f"mean_J={row['mean_cost']:.3f} fixed_J={row['fixed_greedy_cost']:.3f} "
            f"entropy={row['entropy']:.3f} sched_ent={row['scheduled_entropy']:.3f} "
            f"lr={row['learning_rate']:.1e} kl={row['approx_kl']:.5f} "
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
    print(f"final fixed-case PPO cost (N={n_robots}): {final_eval.ppo_cost:.6f}")
    if final_eval.exact_cost is not None:
        print(
            f"final fixed-case exact cost: {final_eval.exact_cost:.6f}; "
            f"absolute gap={final_eval.absolute_gap:.6f}; "
            f"relative gap={final_eval.relative_gap:.3%}"
        )
    if args.plot_after_train:
        default_plots_dir = metrics_path.parent / "plots"
        resolved_plots_dir = (
            _resolve_group_path(args.plots_dir, n_robots, run_id=run_id, run_root=run_root)
            if args.plots_dir
            else default_plots_dir
        )
        try:
            plot_path = _plot_training_curves(
                metrics_path=metrics_path,
                plot_dir=resolved_plots_dir,
            )
            print(f"training curves: {plot_path}")
        except Exception as exc:
            print(f"warning: failed to generate training curves: {exc}")


def main() -> None:
    args = parse_args()
    if args.updates <= 0 or args.episodes_per_update <= 0:
        raise ValueError("updates and episodes-per-update must be positive")
    group_n_robots = _parse_group_robots(args.group_robots)
    if args.n_robots <= 0:
        raise ValueError("--n-robots must be positive")
    if group_n_robots and any(value <= 0 for value in group_n_robots):
        raise ValueError("--group-robots must specify positive integers")
    if not group_n_robots:
        group_n_robots = [args.n_robots]
    if args.hidden_dim <= 0:
        raise ValueError("--hidden-dim must be positive")
    if args.critic_hidden_dim is None:
        args.critic_hidden_dim = args.hidden_dim
    if args.critic_hidden_dim <= 0:
        raise ValueError("--critic-hidden-dim must be positive")
    if args.actor_hidden_layers <= 0:
        raise ValueError("--actor-hidden-layers must be positive")
    if args.critic_hidden_layers <= 0:
        raise ValueError("--critic-hidden-layers must be positive")
    if args.reward_norm_eps <= 0:
        raise ValueError("--reward-norm-eps must be positive")
    if args.max_vehicles_per_entrance < 0:
        raise ValueError("--max-vehicles-per-entrance must be non-negative")

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

    if args.group_robots:
        print(f"group training enabled: groups={group_n_robots}")
    else:
        print(f"single training group: N={args.n_robots}")

    torch.set_num_threads(max(1, int(args.torch_threads)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schedule = {
        "entropy_start": entropy_start,
        "entropy_end": entropy_end,
        "lr_start": lr_start,
        "lr_end": lr_end,
    }
    run_id = args.run_id.strip() or _default_run_id()
    run_root = Path(args.run_root)
    default_checkpoint = "output/ppo_n3/ppo_branch_actor.pt"
    default_metrics = "output/ppo_n3/training_metrics.csv"
    if args.checkpoint == default_checkpoint:
        args.checkpoint = str(run_root / "{run_id}" / "n{n_robots}" / "ppo_branch_actor.pt")
    if args.metrics_csv == default_metrics:
        args.metrics_csv = str(run_root / "{run_id}" / "n{n_robots}" / "training_metrics.csv")

    print(f"run id: {run_id}")
    print(f"run root: {run_root}")

    for run_index, n_robots in enumerate(group_n_robots):
        _run_training_for_n(
            n_robots,
            args=args,
            run_id=run_id,
            run_root=run_root,
            device=device,
            schedule=schedule,
            entropy_schedule=entropy_schedule,
            lr_schedule=lr_schedule,
            run_index=run_index,
            total_runs=len(group_n_robots),
        )


if __name__ == "__main__":
    main()
