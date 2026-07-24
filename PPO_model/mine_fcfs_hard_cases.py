"""Mine fixed-path cases where proven-optimal scheduling clearly beats FCFS."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics
import sys
import time

from PIL import Image, ImageDraw, ImageFont
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from coarse_scheduler import search_dynamic_codesign_dfs_bb
from fcfs_baseline_experiments.independent_fcfs_shortest_path_scheduler import (
    search_fixed_shortest_fcfs_dfs_bb,
)
from PPO_model.cases import ThreeByThreeCaseFactory
from PPO_model.evaluate import greedy_policy_cost, load_actor_checkpoint


TOLERANCE = 1e-8


def _relative_gap(candidate: float, exact: float) -> float:
    if abs(exact) <= TOLERANCE:
        return 0.0 if abs(candidate - exact) <= TOLERANCE else math.inf
    return (candidate - exact) / abs(exact)


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=RUN_DIR")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("PPO run label must not be empty")
    return label, Path(raw_path)


def _mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.mean(finite) if finite else math.nan


def _median(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.median(finite) if finite else math.nan


def _font(size: int, bold: bool = False):
    filename = "arialbd.ttf" if bold else "arial.ttf"
    try:
        return ImageFont.truetype(
            str(Path("C:/Windows/Fonts") / filename), size=size
        )
    except OSError:
        return ImageFont.load_default()


def _plot_summary(
    summary_rows: list[dict[str, object]],
    output_path: Path,
    threshold: float,
) -> None:
    width, height = 1700, 850
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(30, bold=True)
    body_font = _font(19)
    small_font = _font(16)
    colors = ["#4C78A8", "#E45756", "#72B7B2", "#F2CF5B", "#B279A2"]

    title = (
        "PPO on fixed-path hard cases "
        f"(selected by FCFS gap >= {threshold:.0%})"
    )
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(
        ((width - (title_box[2] - title_box[0])) / 2, 24),
        title,
        fill="#111111",
        font=title_font,
    )

    # Left panel: mean optimality gap.
    left_bounds = (100, 125, 820, 700)
    left, top, right, bottom = left_bounds
    gap_values = [float(row["ppo_mean_relative_gap"]) for row in summary_rows]
    fcfs_gap = float(summary_rows[0]["fcfs_mean_relative_gap"])
    all_values = gap_values + [fcfs_gap]
    y_max = max(all_values + [0.1]) * 1.15
    for index in range(6):
        ratio = index / 5
        y = bottom - ratio * (bottom - top)
        draw.line((left, y, right, y), fill="#dddddd", width=1)
        draw.text(
            (left - 60, y - 9),
            f"{ratio * y_max:.0%}",
            fill="#444444",
            font=small_font,
        )
    labels = [str(row["model"]) for row in summary_rows] + ["FCFS"]
    bar_values = gap_values + [fcfs_gap]
    slot = (right - left) / len(labels)
    bar_width = slot * 0.62
    for index, (label, value) in enumerate(zip(labels, bar_values)):
        center = left + slot * (index + 0.5)
        bar_height = max(value, 0.0) / y_max * (bottom - top)
        color = colors[index % len(colors)] if label != "FCFS" else "#A0A0A0"
        draw.rectangle(
            (
                center - bar_width / 2,
                bottom - bar_height,
                center + bar_width / 2,
                bottom,
            ),
            fill=color,
        )
        value_text = f"{value:.1%}"
        value_box = draw.textbbox((0, 0), value_text, font=small_font)
        draw.text(
            (
                center - (value_box[2] - value_box[0]) / 2,
                bottom - bar_height - 24,
            ),
            value_text,
            fill="#222222",
            font=small_font,
        )
        label_box = draw.textbbox((0, 0), label, font=small_font)
        draw.text(
            (
                center - (label_box[2] - label_box[0]) / 2,
                bottom + 12,
            ),
            label,
            fill="#222222",
            font=small_font,
        )
    draw.text((left + 190, 82), "Mean gap to exact (lower is better)", font=body_font)

    # Right panel: PPO win/tie/loss against FCFS.
    panel_left = 930
    draw.text(
        (panel_left + 180, 82),
        "PPO result versus FCFS on selected cases",
        font=body_font,
    )
    row_height = 110
    max_cases = max(int(row["cases"]) for row in summary_rows)
    scale_width = 570
    for index, row in enumerate(summary_rows):
        y = 170 + index * row_height
        label = str(row["model"])
        better = int(row["ppo_better_than_fcfs"])
        tied = int(row["ppo_tied_with_fcfs"])
        worse = int(row["ppo_worse_than_fcfs"])
        draw.text((panel_left, y + 24), label, fill="#222222", font=small_font)
        x = panel_left + 145
        for count, color in (
            (better, "#54A24B"),
            (tied, "#BAB0AC"),
            (worse, "#E45756"),
        ):
            segment = scale_width * count / max(max_cases, 1)
            draw.rectangle((x, y, x + segment, y + 55), fill=color)
            if count:
                text = str(count)
                box = draw.textbbox((0, 0), text, font=small_font)
                if segment >= box[2] - box[0] + 8:
                    draw.text(
                        (
                            x + (segment - (box[2] - box[0])) / 2,
                            y + 17,
                        ),
                        text,
                        fill="#111111",
                        font=small_font,
                    )
            x += segment
    legend_y = 660
    for index, (label, color) in enumerate(
        (("PPO better", "#54A24B"), ("Tie", "#BAB0AC"), ("PPO worse", "#E45756"))
    ):
        x = panel_left + index * 190
        draw.rectangle((x, legend_y, x + 24, legend_y + 20), fill=color)
        draw.text((x + 32, legend_y - 1), label, font=small_font, fill="#333333")

    draw.text(
        (100, 785),
        "Case selection uses only exact and FCFS results; PPO is evaluated after selection.",
        fill="#555555",
        font=small_font,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ppo-run",
        action="append",
        type=_parse_run,
        required=True,
        metavar="LABEL=RUN_DIR",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-robots", type=int, default=5)
    parser.add_argument("--target-cases", type=int, default=100)
    parser.add_argument("--max-candidates", type=int, default=5000)
    parser.add_argument("--min-fcfs-relative-gap", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--exact-deadline", type=float, default=3.0)
    parser.add_argument("--exact-max-nodes", type=int, default=100_000)
    parser.add_argument("--min-initial-release", type=float, default=0.0)
    parser.add_argument("--max-initial-release", type=float, default=5.0)
    parser.add_argument("--max-vehicles-per-entrance", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_robots <= 0:
        raise ValueError("--n-robots must be positive")
    if args.target_cases <= 0 or args.max_candidates < args.target_cases:
        raise ValueError("invalid target/max candidate counts")
    if args.min_fcfs_relative_gap <= 0:
        raise ValueError("--min-fcfs-relative-gap must be positive")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = []
    used_labels: set[str] = set()
    for label, run_dir in args.ppo_run:
        if label in used_labels:
            raise ValueError(f"duplicate PPO label: {label}")
        used_labels.add(label)
        checkpoint = run_dir.resolve() / f"n{args.n_robots}" / "ppo_branch_actor.pt"
        actor, encoding_config, _ = load_actor_checkpoint(checkpoint, device=device)
        models.append((label, actor, encoding_config))

    factory = ThreeByThreeCaseFactory(
        seed=args.seed,
        n_robots=args.n_robots,
        randomize=True,
        min_initial_release=args.min_initial_release,
        max_initial_release=args.max_initial_release,
        fix_shortest_paths=True,
        max_vehicles_per_entrance=args.max_vehicles_per_entrance,
    )

    rows: list[dict[str, object]] = []
    exact_solved = 0
    exact_partial = 0
    started = time.perf_counter()
    for candidate_index in range(1, args.max_candidates + 1):
        case = factory()
        if any(len(plan.route_options) != 1 for plan in case.plans):
            raise AssertionError("case factory did not lock every plan")

        fcfs_start = time.perf_counter()
        fcfs_result = search_fixed_shortest_fcfs_dfs_bb(case.plans, verbose=False)
        fcfs_seconds = time.perf_counter() - fcfs_start
        fcfs_cost = float(fcfs_result.best_g)

        exact_start = time.perf_counter()
        exact_result = search_dynamic_codesign_dfs_bb(
            case.plans,
            branch_and_bound=True,
            verbose=False,
            deadline=args.exact_deadline,
            max_nodes=args.exact_max_nodes,
        )
        exact_seconds = time.perf_counter() - exact_start
        limited = any(
            "deadline hit" in item or "max_nodes hit" in item
            for item in exact_result.log
        )
        solved = (
            not limited
            and exact_result.best_idx >= 0
            and math.isfinite(exact_result.best_g)
        )
        if not solved:
            exact_partial += 1
            continue
        exact_solved += 1
        exact_cost = float(exact_result.best_g)
        fcfs_gap = _relative_gap(fcfs_cost, exact_cost)
        if fcfs_gap + TOLERANCE < args.min_fcfs_relative_gap:
            if candidate_index % 100 == 0:
                print(
                    f"candidates={candidate_index} accepted={len(rows)} "
                    f"solved={exact_solved} partial={exact_partial}",
                    flush=True,
                )
            continue

        requests = ";".join(
            f"{vehicle_id}:{entrance}->{exit_port}@{alpha0:.9f}"
            for vehicle_id, entrance, exit_port, alpha0 in case.requests
        )
        row: dict[str, object] = {
            "hard_case_id": len(rows) + 1,
            "candidate_index": candidate_index,
            "case_name": case.name,
            "seed": args.seed,
            "requests": requests,
            "exact_cost": exact_cost,
            "fcfs_cost": fcfs_cost,
            "fcfs_relative_gap": fcfs_gap,
            "fcfs_seconds": fcfs_seconds,
            "exact_seconds": exact_seconds,
            "exact_nodes": len(exact_result.nodes),
        }
        for label, actor, encoding_config in models:
            ppo_start = time.perf_counter()
            ppo_cost, decisions = greedy_policy_cost(
                actor,
                case.plans,
                encoding_config=encoding_config,
                device=device,
            )
            ppo_seconds = time.perf_counter() - ppo_start
            ppo_minus_fcfs = ppo_cost - fcfs_cost
            row[f"{label}_ppo_cost"] = ppo_cost
            row[f"{label}_relative_gap"] = _relative_gap(ppo_cost, exact_cost)
            row[f"{label}_minus_fcfs"] = ppo_minus_fcfs
            row[f"{label}_vs_fcfs"] = (
                "better"
                if ppo_minus_fcfs < -TOLERANCE
                else "worse"
                if ppo_minus_fcfs > TOLERANCE
                else "tie"
            )
            row[f"{label}_exact_match"] = abs(ppo_cost - exact_cost) <= TOLERANCE
            row[f"{label}_decisions"] = decisions
            row[f"{label}_seconds"] = ppo_seconds
        rows.append(row)
        if len(rows) % 10 == 0 or len(rows) == args.target_cases:
            print(
                f"candidates={candidate_index} accepted={len(rows)}/"
                f"{args.target_cases} elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
        if len(rows) >= args.target_cases:
            break

    if len(rows) < args.target_cases:
        raise RuntimeError(
            f"found only {len(rows)} hard cases in {args.max_candidates} candidates"
        )

    detail_path = output_dir / "hard_cases.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fcfs_gaps = [float(row["fcfs_relative_gap"]) for row in rows]
    summary_rows: list[dict[str, object]] = []
    for label, _, _ in models:
        ppo_gaps = [float(row[f"{label}_relative_gap"]) for row in rows]
        outcomes = [str(row[f"{label}_vs_fcfs"]) for row in rows]
        summary_rows.append(
            {
                "model": label,
                "cases": len(rows),
                "candidate_cases_checked": int(rows[-1]["candidate_index"]),
                "minimum_fcfs_relative_gap": args.min_fcfs_relative_gap,
                "fcfs_mean_relative_gap": _mean(fcfs_gaps),
                "fcfs_median_relative_gap": _median(fcfs_gaps),
                "ppo_mean_relative_gap": _mean(ppo_gaps),
                "ppo_median_relative_gap": _median(ppo_gaps),
                "ppo_exact_matches": sum(
                    bool(row[f"{label}_exact_match"]) for row in rows
                ),
                "ppo_better_than_fcfs": outcomes.count("better"),
                "ppo_tied_with_fcfs": outcomes.count("tie"),
                "ppo_worse_than_fcfs": outcomes.count("worse"),
                "mean_ppo_minus_fcfs": _mean(
                    [float(row[f"{label}_minus_fcfs"]) for row in rows]
                ),
                "mean_ppo_seconds": _mean(
                    [float(row[f"{label}_seconds"]) for row in rows]
                ),
                "mean_fcfs_seconds": _mean(
                    [float(row["fcfs_seconds"]) for row in rows]
                ),
                "mean_exact_seconds": _mean(
                    [float(row["exact_seconds"]) for row in rows]
                ),
            }
        )

    summary_path = output_dir / "hard_case_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    plot_path = output_dir / "hard_case_comparison.png"
    _plot_summary(summary_rows, plot_path, args.min_fcfs_relative_gap)
    print("--------------------------------------------------")
    print(f"hard cases: {detail_path}")
    print(f"summary: {summary_path}")
    print(f"plot: {plot_path}")
    for row in summary_rows:
        print(
            f"{row['model']}: gap={float(row['ppo_mean_relative_gap']):.2%} "
            f"vs FCFS={row['ppo_better_than_fcfs']}/"
            f"{row['ppo_tied_with_fcfs']}/{row['ppo_worse_than_fcfs']} "
            "(better/tie/worse)"
        )


if __name__ == "__main__":
    main()
