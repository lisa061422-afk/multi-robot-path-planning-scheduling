"""Plot a compact A/B comparison of PPO reward formulations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _series(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    result = np.empty_like(values)
    result[0] = values[0]
    for index in range(1, len(values)):
        result[index] = alpha * values[index] + (1.0 - alpha) * result[index - 1]
    return result


def _plot_training_axis(
    axis: plt.Axes,
    update: np.ndarray,
    baseline: np.ndarray,
    shaped: np.ndarray,
    title: str,
    ylabel: str,
    span: int,
    log_scale: bool = False,
) -> None:
    colors = ("#4C78A8", "#E45756")
    labels = (r"Original reward: $-\Delta g$", "Pending-delay reward")
    for values, color, label in zip((baseline, shaped), colors, labels):
        axis.plot(update, values, color=color, alpha=0.16, linewidth=0.8)
        axis.plot(update, _ema(values, span), color=color, linewidth=2.2, label=label)
    axis.set_title(title)
    axis.set_xlabel("PPO update")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    if log_scale:
        axis.set_yscale("log")


def _summary_values(summary_path: Path) -> dict[str, float]:
    rows = _read_csv(summary_path)
    if len(rows) != 1:
        raise ValueError(f"expected one summary row in {summary_path}")
    return {key: float(value) for key, value in rows[0].items() if key != "n_robots"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--shaped-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ema-span", type=int, default=10)
    args = parser.parse_args()

    baseline_rows = _read_csv(args.baseline_run / "n5" / "training_metrics.csv")
    shaped_rows = _read_csv(args.shaped_run / "n5" / "training_metrics.csv")
    if len(baseline_rows) != len(shaped_rows):
        raise ValueError("A/B runs must contain the same number of PPO updates")

    update = _series(baseline_rows, "update")
    baseline_summary = _summary_values(
        args.baseline_run / "heldout_compare" / "comparison_summary.csv"
    )
    shaped_summary = _summary_values(
        args.shaped_run / "heldout_compare" / "comparison_summary.csv"
    )

    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    _plot_training_axis(
        axes[0, 0],
        update,
        _series(baseline_rows, "critic_loss"),
        _series(shaped_rows, "critic_loss"),
        "Critic loss (log scale)",
        "MSE loss",
        args.ema_span,
        log_scale=True,
    )
    _plot_training_axis(
        axes[0, 1],
        update,
        _series(baseline_rows, "actor_loss"),
        _series(shaped_rows, "actor_loss"),
        "Actor loss",
        "PPO surrogate loss",
        args.ema_span,
    )
    _plot_training_axis(
        axes[0, 2],
        update,
        _series(baseline_rows, "entropy"),
        _series(shaped_rows, "entropy"),
        "Policy entropy",
        "Entropy",
        args.ema_span,
    )
    _plot_training_axis(
        axes[1, 0],
        update,
        _series(baseline_rows, "mean_cost"),
        _series(shaped_rows, "mean_cost"),
        "Mean rollout cost",
        "Cost",
        args.ema_span,
    )

    labels = [r"Original $-\Delta g$", "Pending delay"]
    colors = ["#4C78A8", "#E45756"]
    gap_values = [
        100.0 * baseline_summary["ppo_mean_relative_gap"],
        100.0 * shaped_summary["ppo_mean_relative_gap"],
    ]
    axes[1, 1].bar(labels, gap_values, color=colors, width=0.62)
    axes[1, 1].set_title("Held-out mean gap to exact")
    axes[1, 1].set_ylabel("Optimality gap (%)")
    axes[1, 1].grid(axis="y", alpha=0.25)
    for index, value in enumerate(gap_values):
        axes[1, 1].text(index, value, f"{value:.3f}%", ha="center", va="bottom")

    categories = ["Better", "Tie", "Worse"]
    x = np.arange(len(categories))
    width = 0.36
    baseline_counts = [
        baseline_summary["ppo_better_than_fcfs"],
        baseline_summary["ppo_tied_with_fcfs"],
        baseline_summary["ppo_worse_than_fcfs"],
    ]
    shaped_counts = [
        shaped_summary["ppo_better_than_fcfs"],
        shaped_summary["ppo_tied_with_fcfs"],
        shaped_summary["ppo_worse_than_fcfs"],
    ]
    axes[1, 2].bar(
        x - width / 2, baseline_counts, width, color=colors[0], label=labels[0]
    )
    axes[1, 2].bar(
        x + width / 2, shaped_counts, width, color=colors[1], label=labels[1]
    )
    axes[1, 2].set_xticks(x, categories)
    axes[1, 2].set_title("Held-out PPO versus FCFS")
    axes[1, 2].set_ylabel("Nontrivial cases")
    axes[1, 2].grid(axis="y", alpha=0.25)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.97),
    )
    figure.suptitle(
        "N=5 reward-shaping pilot: 100 updates / 3,200 episodes per run",
        fontsize=15,
        y=0.995,
    )
    figure.text(
        0.5,
        0.012,
        "Faint lines: raw update values. Solid lines: "
        f"EMA (span={args.ema_span}). Held-out set: 94 nontrivial cases.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
