"""Compare trained PPO actors with fixed-path FCFS and exact scheduling.

All three methods receive the same plans after each vehicle has been locked to
the same shortest route by ``ThreeByThreeCaseFactory``.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Iterable

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
    gap = candidate - exact
    if abs(exact) <= TOLERANCE:
        return 0.0 if abs(gap) <= TOLERANCE else math.inf
    return gap / abs(exact)


def _finite_mean(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.mean(finite) if finite else math.nan


def _finite_median(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.median(finite) if finite else math.nan


def _font(size: int, bold: bool = False):
    filename = "arialbd.ttf" if bold else "arial.ttf"
    try:
        return ImageFont.truetype(str(Path("C:/Windows/Fonts") / filename), size=size)
    except OSError:
        return ImageFont.load_default()


def _plot_gap_bars(summary_rows: list[dict[str, object]], output_path: Path) -> None:
    width, height = 1500, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(28, bold=True)
    body_font = _font(18)
    small_font = _font(15)
    bounds = (110, 90, 1420, 790)
    left, top, right, bottom = bounds

    values = [
        float(row[key])
        for row in summary_rows
        for key in ("ppo_mean_relative_gap", "fcfs_mean_relative_gap")
        if math.isfinite(float(row[key]))
    ]
    y_max = max(values + [0.01]) * 1.15
    y_max = max(y_max, 0.05)
    for index in range(6):
        ratio = index / 5
        y = bottom - int(ratio * (bottom - top))
        value = ratio * y_max
        draw.line((left, y, right, y), fill="#dddddd", width=1)
        label = f"{value:.0%}"
        box = draw.textbbox((0, 0), label, font=small_font)
        draw.text((left - (box[2] - box[0]) - 12, y - 8), label, fill="#333333", font=small_font)
    draw.rectangle(bounds, outline="#333333", width=1)

    group_width = (right - left) / max(len(summary_rows), 1)
    bar_width = group_width * 0.28
    for index, row in enumerate(summary_rows):
        center = left + group_width * (index + 0.5)
        for offset, key, color in (
            (-bar_width, "ppo_mean_relative_gap", "#1f77b4"),
            (0.0, "fcfs_mean_relative_gap", "#ff7f0e"),
        ):
            value = float(row[key])
            if not math.isfinite(value):
                continue
            bar_height = max(0.0, value) / y_max * (bottom - top)
            x0 = center + offset
            y0 = bottom - bar_height
            draw.rectangle((x0, y0, x0 + bar_width, bottom), fill=color)
        label = f"N={row['n_robots']}"
        box = draw.textbbox((0, 0), label, font=small_font)
        draw.text((center - (box[2] - box[0]) / 2, bottom + 12), label, fill="#333333", font=small_font)

    title = "Mean optimality gap on non-trivial fixed-shortest-path cases"
    box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((width - (box[2] - box[0])) / 2, 25), title, fill="#111111", font=title_font)
    legend_y = 835
    draw.rectangle((560, legend_y, 585, legend_y + 18), fill="#1f77b4")
    draw.text((595, legend_y - 2), "PPO", fill="#222222", font=body_font)
    draw.rectangle((700, legend_y, 725, legend_y + 18), fill="#ff7f0e")
    draw.text((735, legend_y - 2), "FCFS", fill="#222222", font=body_font)
    draw.text((110, 835), "Lower is better; exact = 0% gap.", fill="#555555", font=small_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-min", type=int, default=2)
    parser.add_argument("--n-max", type=int, default=12)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--min-initial-release", type=float, default=0.0)
    parser.add_argument("--max-initial-release", type=float, default=5.0)
    parser.add_argument("--max-vehicles-per-entrance", type=int, default=1)
    parser.add_argument(
        "--exact-deadline",
        type=float,
        default=0.0,
        help=(
            "seconds allowed per exact case (0 or less means no deadline; "
            "partial cases are excluded from optimality-gap statistics)"
        ),
    )
    parser.add_argument(
        "--exact-max-nodes",
        type=int,
        default=0,
        help="node cap per exact case (0 or less means unlimited)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_min <= 0 or args.n_max < args.n_min:
        raise ValueError("invalid N range")
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.max_vehicles_per_entrance <= 0:
        raise ValueError("--max-vehicles-per-entrance must be positive")
    if args.exact_deadline < 0:
        raise ValueError("exact-deadline must be non-negative")
    if args.exact_max_nodes < 0:
        raise ValueError("exact-max-nodes must be non-negative")

    run_dir = args.run_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_dir / "comparison_ppo_fcfs_exact"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    detail_path = output_dir / "comparison_rows.csv"
    summary_path = output_dir / "comparison_summary.csv"
    detail_handle = detail_path.open("w", encoding="utf-8", newline="")
    detail_writer: csv.DictWriter | None = None

    exact_deadline = None if args.exact_deadline <= 0.0 else args.exact_deadline
    exact_max_nodes = None if args.exact_max_nodes <= 0 else args.exact_max_nodes

    try:
        for n_robots in range(args.n_min, args.n_max + 1):
            checkpoint = run_dir / f"n{n_robots}" / "ppo_branch_actor.pt"
            actor, encoding_config, _ = load_actor_checkpoint(checkpoint, device=device)
            factory = ThreeByThreeCaseFactory(
                seed=args.seed + 1009 * n_robots,
                n_robots=n_robots,
                randomize=True,
                min_initial_release=args.min_initial_release,
                max_initial_release=args.max_initial_release,
                fix_shortest_paths=True,
                max_vehicles_per_entrance=args.max_vehicles_per_entrance,
            )
            n_rows: list[dict[str, object]] = []
            print(f"[N={n_robots}] starting {args.episodes} cases", flush=True)
            for episode in range(1, args.episodes + 1):
                case = factory()
                if any(len(plan.route_options) != 1 for plan in case.plans):
                    raise AssertionError("case factory did not lock every plan to one shortest route")

                start = time.perf_counter()
                ppo_cost, decisions = greedy_policy_cost(
                    actor,
                    case.plans,
                    encoding_config=encoding_config,
                    device=device,
                )
                ppo_seconds = time.perf_counter() - start

                start = time.perf_counter()
                fcfs_result = search_fixed_shortest_fcfs_dfs_bb(case.plans, verbose=False)
                fcfs_seconds = time.perf_counter() - start
                fcfs_cost = float(fcfs_result.best_g)

                start = time.perf_counter()
                exact_result = search_dynamic_codesign_dfs_bb(
                    case.plans,
                    branch_and_bound=False,
                    verbose=False,
                    deadline=exact_deadline,
                    max_nodes=exact_max_nodes,
                )
                exact_seconds = time.perf_counter() - start
                limited = any(
                    "deadline hit" in item or "max_nodes hit" in item
                    for item in exact_result.log
                )
                has_incumbent = (
                    exact_result.best_idx >= 0 and math.isfinite(exact_result.best_g)
                )
                exact_status = "partial" if limited else "solved"
                if not has_incumbent:
                    exact_status = "unsolved"
                exact_cost = float(exact_result.best_g) if has_incumbent else math.nan

                is_exact = exact_status == "solved"
                ppo_gap = ppo_cost - exact_cost if is_exact else math.nan
                fcfs_gap = fcfs_cost - exact_cost if is_exact else math.nan
                ppo_vs_fcfs = ppo_cost - fcfs_cost
                ordering_violation = is_exact and (
                    ppo_gap < -TOLERANCE or fcfs_gap < -TOLERANCE
                )
                requests = ";".join(
                    f"{vehicle_id}:{entrance}->{exit_port}@{alpha0:.9f}"
                    for vehicle_id, entrance, exit_port, alpha0 in case.requests
                )
                row: dict[str, object] = {
                    "n_robots": n_robots,
                    "episode": episode,
                    "case_name": case.name,
                    "seed": args.seed + 1009 * n_robots,
                    "requests": requests,
                    "fixed_path_verified": True,
                    "decisions": decisions,
                    "nontrivial": decisions > 0,
                    "ppo_cost": ppo_cost,
                    "fcfs_cost": fcfs_cost,
                    "exact_cost": exact_cost,
                    "exact_status": exact_status,
                    "ppo_absolute_gap": ppo_gap,
                    "fcfs_absolute_gap": fcfs_gap,
                    "ppo_relative_gap": (
                        _relative_gap(ppo_cost, exact_cost) if is_exact else math.nan
                    ),
                    "fcfs_relative_gap": (
                        _relative_gap(fcfs_cost, exact_cost) if is_exact else math.nan
                    ),
                    "ppo_minus_fcfs": ppo_vs_fcfs,
                    "ppo_vs_fcfs": (
                        "better"
                        if ppo_vs_fcfs < -TOLERANCE
                        else "worse"
                        if ppo_vs_fcfs > TOLERANCE
                        else "tie"
                    ),
                    "ppo_exact_match": is_exact and abs(ppo_gap) <= TOLERANCE,
                    "fcfs_exact_match": is_exact and abs(fcfs_gap) <= TOLERANCE,
                    "ordering_violation": ordering_violation,
                    "ppo_seconds": ppo_seconds,
                    "fcfs_seconds": fcfs_seconds,
                    "exact_seconds": exact_seconds,
                    "exact_nodes": len(exact_result.nodes),
                }
                n_rows.append(row)
                detail_rows.append(row)
                if detail_writer is None:
                    detail_writer = csv.DictWriter(
                        detail_handle, fieldnames=list(row.keys())
                    )
                    detail_writer.writeheader()
                detail_writer.writerow(row)
                detail_handle.flush()
                exact_text = f"{exact_cost:.6f}" if math.isfinite(exact_cost) else "nan"
                print(
                    f"  case={episode:03d} decisions={decisions:2d} "
                    f"exact={exact_text}({exact_status}) ppo={ppo_cost:.6f} "
                    f"fcfs={fcfs_cost:.6f} ppo-vs-fcfs={row['ppo_vs_fcfs']}",
                    flush=True,
                )

            nontrivial = [row for row in n_rows if bool(row["nontrivial"])]
            exact_base = [
                row for row in nontrivial if row["exact_status"] == "solved"
            ]
            summary_rows.append(
                {
                    "n_robots": n_robots,
                    "cases": len(n_rows),
                    "nontrivial_cases": len(nontrivial),
                    "trivial_cases": len(n_rows) - len(nontrivial),
                    "exact_solved_cases": sum(
                        row["exact_status"] == "solved" for row in n_rows
                    ),
                    "exact_partial_cases": sum(
                        row["exact_status"] == "partial" for row in n_rows
                    ),
                    "exact_unsolved_cases": sum(
                        row["exact_status"] == "unsolved" for row in n_rows
                    ),
                    "ppo_mean_relative_gap": _finite_mean(
                        float(row["ppo_relative_gap"]) for row in exact_base
                    ),
                    "ppo_median_relative_gap": _finite_median(
                        float(row["ppo_relative_gap"]) for row in exact_base
                    ),
                    "fcfs_mean_relative_gap": _finite_mean(
                        float(row["fcfs_relative_gap"]) for row in exact_base
                    ),
                    "fcfs_median_relative_gap": _finite_median(
                        float(row["fcfs_relative_gap"]) for row in exact_base
                    ),
                    "ppo_exact_matches": sum(bool(row["ppo_exact_match"]) for row in exact_base),
                    "fcfs_exact_matches": sum(bool(row["fcfs_exact_match"]) for row in exact_base),
                    "ppo_better_than_fcfs": sum(
                        row["ppo_vs_fcfs"] == "better" for row in nontrivial
                    ),
                    "ppo_tied_with_fcfs": sum(
                        row["ppo_vs_fcfs"] == "tie" for row in nontrivial
                    ),
                    "ppo_worse_than_fcfs": sum(
                        row["ppo_vs_fcfs"] == "worse" for row in nontrivial
                    ),
                    "mean_ppo_minus_fcfs": _finite_mean(
                        float(row["ppo_minus_fcfs"]) for row in nontrivial
                    ),
                    "ordering_violations": sum(bool(row["ordering_violation"]) for row in n_rows),
                    "mean_ppo_seconds": _finite_mean(float(row["ppo_seconds"]) for row in n_rows),
                    "mean_fcfs_seconds": _finite_mean(float(row["fcfs_seconds"]) for row in n_rows),
                    "mean_exact_seconds": _finite_mean(float(row["exact_seconds"]) for row in n_rows),
                }
            )
            with summary_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(summary_rows)
    finally:
        detail_handle.close()

    if not summary_rows:
        raise RuntimeError("no comparison rows were generated")

    plot_path = output_dir / "mean_optimality_gap_ppo_vs_fcfs.png"
    _plot_gap_bars(summary_rows, plot_path)

    violations = sum(int(row["ordering_violations"]) for row in summary_rows)
    print("--------------------------------------------------")
    print(f"rows={detail_path}")
    print(f"summary={summary_path}")
    print(f"plot={plot_path}")
    print(f"exact-ordering violations={violations}")
    if violations:
        raise RuntimeError("found a candidate cost below exact cost; inspect comparison rows")


if __name__ == "__main__":
    main()
