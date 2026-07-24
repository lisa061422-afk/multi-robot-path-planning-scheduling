"""Measure where FCFS and PPO costs lie between optimal and worst tree leaves."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics
import sys
import time

from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from coarse_scheduler import search_dynamic_codesign_dfs_bb
from PPO_model.cases import ThreeByThreeCaseFactory


TOLERANCE = 1e-8


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.mean(finite) if finite else math.nan


def _median(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.median(finite) if finite else math.nan


def _tree_quality(cost: float, optimal: float, worst: float) -> float:
    """Return 0 at worst and 1 at optimal."""
    width = worst - optimal
    if width <= TOLERANCE:
        return 1.0
    return max(0.0, min(1.0, (worst - cost) / width))


def _strictly_worse_fraction(cost: float, leaf_costs: list[float]) -> float:
    return sum(value > cost + TOLERANCE for value in leaf_costs) / len(leaf_costs)


def _font(size: int, bold: bool = False):
    filename = "arialbd.ttf" if bold else "arial.ttf"
    try:
        return ImageFont.truetype(
            str(Path("C:/Windows/Fonts") / filename), size=size
        )
    except OSError:
        return ImageFont.load_default()


def _plot_summary(summary: list[dict[str, object]], output_path: Path) -> None:
    width, height = 1500, 820
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(30, bold=True)
    body_font = _font(19)
    small_font = _font(16)
    colors = ["#9D9D9D", "#4C78A8", "#E45756", "#72B7B2", "#F2CF5B"]

    title = "Position of FCFS and PPO within the complete decision tree"
    box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(
        ((width - (box[2] - box[0])) / 2, 24),
        title,
        fill="#111111",
        font=title_font,
    )

    panels = [
        (
            "mean_tree_quality",
            "Mean normalized quality",
            "0% = worst leaf, 100% = optimal leaf",
        ),
        (
            "mean_strictly_worse_leaf_fraction",
            "Mean fraction of leaves strictly worse",
            "Higher means the method ranks better in the tree",
        ),
    ]
    for panel_index, (key, heading, note) in enumerate(panels):
        left = 100 + panel_index * 730
        top, right, bottom = 150, 680 + panel_index * 730, 650
        draw.text((left + 150, 91), heading, font=body_font, fill="#222222")
        draw.text((left + 105, 121), note, font=small_font, fill="#555555")
        for tick in range(6):
            ratio = tick / 5
            y = bottom - ratio * (bottom - top)
            draw.line((left, y, right, y), fill="#dddddd", width=1)
            draw.text(
                (left - 52, y - 9),
                f"{ratio:.0%}",
                font=small_font,
                fill="#444444",
            )
        slot = (right - left) / len(summary)
        bar_width = slot * 0.62
        for index, row in enumerate(summary):
            value = float(row[key])
            center = left + slot * (index + 0.5)
            height_px = value * (bottom - top)
            draw.rectangle(
                (
                    center - bar_width / 2,
                    bottom - height_px,
                    center + bar_width / 2,
                    bottom,
                ),
                fill=colors[index % len(colors)],
            )
            value_text = f"{value:.1%}"
            value_box = draw.textbbox((0, 0), value_text, font=small_font)
            draw.text(
                (
                    center - (value_box[2] - value_box[0]) / 2,
                    bottom - height_px - 23,
                ),
                value_text,
                font=small_font,
                fill="#222222",
            )
            label = str(row["method"])
            label_box = draw.textbbox((0, 0), label, font=small_font)
            draw.text(
                (
                    center - (label_box[2] - label_box[0]) / 2,
                    bottom + 14,
                ),
                label,
                font=small_font,
                fill="#222222",
            )
    draw.text(
        (100, 770),
        "Computed by exhaustive enumeration without branch-and-bound pruning.",
        font=small_font,
        fill="#555555",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard-cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-robots", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--deadline-per-case", type=float, default=30.0)
    parser.add_argument("--max-nodes-per-case", type=int, default=2_000_000)
    parser.add_argument("--min-initial-release", type=float, default=0.0)
    parser.add_argument("--max-initial-release", type=float, default=5.0)
    parser.add_argument("--max-vehicles-per-entrance", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_rows = _read_csv(args.hard_cases)
    if not source_rows:
        raise ValueError("hard-case CSV is empty")
    target_by_candidate = {
        int(row["candidate_index"]): row for row in source_rows
    }
    max_candidate = max(target_by_candidate)
    ppo_labels = sorted(
        key[: -len("_ppo_cost")]
        for key in source_rows[0]
        if key.endswith("_ppo_cost")
    )
    methods = ["uniform_leaf", "fcfs"] + ppo_labels

    factory = ThreeByThreeCaseFactory(
        seed=args.seed,
        n_robots=args.n_robots,
        randomize=True,
        min_initial_release=args.min_initial_release,
        max_initial_release=args.max_initial_release,
        fix_shortest_paths=True,
        max_vehicles_per_entrance=args.max_vehicles_per_entrance,
    )
    output_rows: list[dict[str, object]] = []
    partial_cases: list[int] = []
    started = time.perf_counter()
    for candidate_index in range(1, max_candidate + 1):
        case = factory()
        source = target_by_candidate.get(candidate_index)
        if source is None:
            continue

        search_started = time.perf_counter()
        result = search_dynamic_codesign_dfs_bb(
            case.plans,
            branch_and_bound=False,
            deadline=args.deadline_per_case,
            max_nodes=args.max_nodes_per_case,
            verbose=False,
        )
        exhaustive_seconds = time.perf_counter() - search_started
        limited = any(
            "deadline hit" in item or "max_nodes hit" in item for item in result.log
        )
        if limited or not result.leaves:
            partial_cases.append(candidate_index)
            continue

        leaf_costs = [float(result.nodes[index].g) for index in result.leaves]
        optimal = min(leaf_costs)
        worst = max(leaf_costs)
        expected_optimal = float(source["exact_cost"])
        if abs(optimal - expected_optimal) > TOLERANCE:
            raise AssertionError(
                f"case {candidate_index}: exhaustive minimum {optimal} "
                f"!= stored exact {expected_optimal}"
            )
        row: dict[str, object] = {
            "hard_case_id": source["hard_case_id"],
            "candidate_index": candidate_index,
            "requests": source["requests"],
            "terminal_leaves": len(leaf_costs),
            "unique_terminal_costs": len(
                {round(value, 10) for value in leaf_costs}
            ),
            "exhaustive_nodes": len(result.nodes),
            "exhaustive_seconds": exhaustive_seconds,
            "optimal_cost": optimal,
            "worst_cost": worst,
            "tree_cost_range": worst - optimal,
            "uniform_leaf_cost": _mean(leaf_costs),
        }
        for method in methods:
            if method == "uniform_leaf":
                cost = float(row["uniform_leaf_cost"])
            else:
                cost_key = (
                    "fcfs_cost" if method == "fcfs" else f"{method}_ppo_cost"
                )
                cost = float(source[cost_key])
            row[f"{method}_cost"] = cost
            row[f"{method}_tree_quality"] = _tree_quality(cost, optimal, worst)
            row[f"{method}_strictly_worse_leaf_fraction"] = (
                _strictly_worse_fraction(cost, leaf_costs)
            )
            row[f"{method}_relative_reduction_from_worst"] = (
                (worst - cost) / abs(worst) if abs(worst) > TOLERANCE else 0.0
            )
            row[f"{method}_is_worst"] = abs(cost - worst) <= TOLERANCE
            row[f"{method}_is_optimal"] = abs(cost - optimal) <= TOLERANCE
        output_rows.append(row)
        if len(output_rows) % 10 == 0:
            print(
                f"completed={len(output_rows)}/{len(source_rows)} "
                f"candidate={candidate_index} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    if partial_cases:
        raise RuntimeError(
            f"exhaustive enumeration incomplete for candidates: {partial_cases}"
        )
    if len(output_rows) != len(source_rows):
        raise RuntimeError(
            f"completed {len(output_rows)} of {len(source_rows)} cases"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "tree_position_rows.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    summary: list[dict[str, object]] = []
    for method in methods:
        qualities = [
            float(row[f"{method}_tree_quality"]) for row in output_rows
        ]
        worse_fractions = [
            float(row[f"{method}_strictly_worse_leaf_fraction"])
            for row in output_rows
        ]
        reductions = [
            float(row[f"{method}_relative_reduction_from_worst"])
            for row in output_rows
        ]
        summary.append(
            {
                "method": (
                    "UniformLeaf"
                    if method == "uniform_leaf"
                    else "FCFS"
                    if method == "fcfs"
                    else method
                ),
                "cases": len(output_rows),
                "mean_tree_quality": _mean(qualities),
                "median_tree_quality": _median(qualities),
                "mean_strictly_worse_leaf_fraction": _mean(worse_fractions),
                "median_strictly_worse_leaf_fraction": _median(worse_fractions),
                "mean_relative_reduction_from_worst": _mean(reductions),
                "median_relative_reduction_from_worst": _median(reductions),
                "worst_leaf_matches": sum(
                    bool(row[f"{method}_is_worst"]) for row in output_rows
                ),
                "optimal_leaf_matches": sum(
                    bool(row[f"{method}_is_optimal"]) for row in output_rows
                ),
                "mean_terminal_leaves": _mean(
                    [float(row["terminal_leaves"]) for row in output_rows]
                ),
                "median_terminal_leaves": _median(
                    [float(row["terminal_leaves"]) for row in output_rows]
                ),
                "max_terminal_leaves": max(
                    int(row["terminal_leaves"]) for row in output_rows
                ),
            }
        )
    summary_path = output_dir / "tree_position_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    plot_path = output_dir / "tree_position_comparison.png"
    _plot_summary(summary, plot_path)
    print("--------------------------------------------------")
    print(f"details: {detail_path}")
    print(f"summary: {summary_path}")
    print(f"plot: {plot_path}")
    for row in summary:
        print(
            f"{row['method']}: quality={float(row['mean_tree_quality']):.2%} "
            f"leaves-worse={float(row['mean_strictly_worse_leaf_fraction']):.2%} "
            f"worst-matches={row['worst_leaf_matches']}"
        )


if __name__ == "__main__":
    main()
