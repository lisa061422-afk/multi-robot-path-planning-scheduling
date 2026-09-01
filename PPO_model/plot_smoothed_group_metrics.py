from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean

from PIL import Image, ImageDraw, ImageFont


METRICS = {
    "mean_cost": "Mean rollout cost",
    "actor_loss": "Actor loss",
    "critic_loss": "Critic loss",
    "entropy": "Policy entropy",
    "approx_kl": "Approximate KL",
    "clip_fraction": "Clip fraction",
    "fixed_greedy_cost": "Fixed-case greedy cost",
}

COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#111111",
]


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    filename = "arialbd.ttf" if bold else "arial.ttf"
    path = Path("C:/Windows/Fonts") / filename
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def _read_metric(path: Path, metric: str) -> tuple[list[float], list[float]]:
    updates: list[float] = []
    values: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                update = float(row["update"])
                value = float(row[metric])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(update) and math.isfinite(value):
                updates.append(update)
                values.append(value)
    return updates, values


def _moving_average(values: list[float], window: int) -> list[float]:
    result: list[float] = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        count = min(index + 1, window)
        result.append(running_sum / count)
    return result


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = fraction * (len(ordered) - 1)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return ordered[lo]
    ratio = position - lo
    return ordered[lo] * (1.0 - ratio) + ordered[hi] * ratio


def _range_with_padding(values: list[float], robust: bool = True) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    if robust and len(values) >= 20:
        lo = _percentile(values, 0.01)
        hi = _percentile(values, 0.99)
    else:
        lo, hi = min(values), max(values)
    if lo == hi:
        padding = max(abs(lo) * 0.05, 1e-6)
    else:
        padding = (hi - lo) * 0.07
    return lo - padding, hi + padding


def _fmt(value: float) -> str:
    magnitude = abs(value)
    if magnitude == 0:
        return "0"
    if magnitude < 0.001 or magnitude >= 10000:
        return f"{value:.2e}"
    if magnitude < 0.1:
        return f"{value:.4f}"
    if magnitude < 10:
        return f"{value:.3f}"
    return f"{value:.1f}"


def _line_points(
    xs: list[float],
    ys: list[float],
    bounds: tuple[int, int, int, int],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> list[tuple[int, int]]:
    left, top, right, bottom = bounds
    x_min, x_max = x_range
    y_min, y_max = y_range
    x_span = max(x_max - x_min, 1e-12)
    y_span = max(y_max - y_min, 1e-12)
    points: list[tuple[int, int]] = []
    for x, y in zip(xs, ys):
        clipped_y = min(max(y, y_min), y_max)
        px = left + int((x - x_min) / x_span * (right - left))
        py = bottom - int((clipped_y - y_min) / y_span * (bottom - top))
        points.append((px, py))
    return points


def _draw_axes(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    small_font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    x_ticks: int = 5,
    y_ticks: int = 5,
) -> None:
    left, top, right, bottom = bounds
    grid = "#dddddd"
    axis = "#333333"
    for index in range(x_ticks + 1):
        ratio = index / x_ticks
        x = left + int(ratio * (right - left))
        value = x_range[0] + ratio * (x_range[1] - x_range[0])
        draw.line((x, top, x, bottom), fill=grid, width=1)
        label = f"{value:.0f}"
        box = draw.textbbox((0, 0), label, font=small_font)
        draw.text((x - (box[2] - box[0]) / 2, bottom + 10), label, fill=axis, font=small_font)
    for index in range(y_ticks + 1):
        ratio = index / y_ticks
        y = bottom - int(ratio * (bottom - top))
        value = y_range[0] + ratio * (y_range[1] - y_range[0])
        draw.line((left, y, right, y), fill=grid, width=1)
        label = _fmt(value)
        box = draw.textbbox((0, 0), label, font=small_font)
        draw.text((left - (box[2] - box[0]) - 12, y - (box[3] - box[1]) / 2), label, fill=axis, font=small_font)
    draw.rectangle(bounds, outline=axis, width=1)


def _plot_overlay(
    series: dict[int, tuple[list[float], list[float], list[float]]],
    metric: str,
    output_path: Path,
    window: int,
) -> None:
    width, height = 1600, 950
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(26, bold=True)
    body_font = _font(18)
    small_font = _font(15)
    bounds = (100, 80, 1350, 850)

    all_x = [x for xs, _, _ in series.values() for x in xs]
    all_raw = [y for _, raw, _ in series.values() for y in raw]
    all_smooth = [y for _, _, smooth in series.values() for y in smooth]
    x_range = (min(all_x), max(all_x))
    y_range = _range_with_padding(all_raw + all_smooth, robust=True)
    _draw_axes(draw, bounds, x_range, y_range, small_font)

    raw_layer = Image.new("RGBA", image.size, (255, 255, 255, 0))
    raw_draw = ImageDraw.Draw(raw_layer)
    for index, n_robots in enumerate(sorted(series)):
        xs, raw, _ = series[n_robots]
        color = COLORS[index % len(COLORS)]
        rgba = tuple(int(color[pos : pos + 2], 16) for pos in (1, 3, 5)) + (35,)
        points = _line_points(xs, raw, bounds, x_range, y_range)
        if len(points) >= 2:
            raw_draw.line(points, fill=rgba, width=1)
    image = Image.alpha_composite(image.convert("RGBA"), raw_layer)
    draw = ImageDraw.Draw(image)

    for index, n_robots in enumerate(sorted(series)):
        xs, _, smooth = series[n_robots]
        color = COLORS[index % len(COLORS)]
        points = _line_points(xs, smooth, bounds, x_range, y_range)
        if len(points) >= 2:
            draw.line(points, fill=color, width=4)

    title = f"{METRICS[metric]}: faint raw data + {window}-update moving average"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((width - (title_box[2] - title_box[0])) / 2, 20), title, fill="#111111", font=title_font)
    x_label = "PPO update"
    x_label_box = draw.textbbox((0, 0), x_label, font=body_font)
    draw.text(
        ((bounds[0] + bounds[2] - (x_label_box[2] - x_label_box[0])) / 2, 875),
        x_label,
        fill="#333333",
        font=body_font,
    )

    legend_x, legend_y = 1390, 95
    for index, n_robots in enumerate(sorted(series)):
        y = legend_y + index * 34
        color = COLORS[index % len(COLORS)]
        draw.line((legend_x, y + 9, legend_x + 35, y + 9), fill=color, width=5)
        draw.text((legend_x + 48, y), f"N={n_robots}", fill="#222222", font=body_font)

    draw.text(
        (100, 923),
        "Y-axis uses the 1st-99th percentile range; extreme raw spikes are clipped at the plot boundary.",
        fill="#666666",
        font=small_font,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path, quality=95)


def _plot_entropy_panels(
    series: dict[int, tuple[list[float], list[float], list[float]]],
    output_path: Path,
    window: int,
) -> None:
    width, height = 1800, 1250
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(28, bold=True)
    panel_title_font = _font(18, bold=True)
    small_font = _font(13)
    cols, rows = 3, 4
    margin_x, margin_top, margin_bottom = 55, 85, 55
    gap_x, gap_y = 35, 35
    panel_w = (width - 2 * margin_x - (cols - 1) * gap_x) // cols
    panel_h = (height - margin_top - margin_bottom - (rows - 1) * gap_y) // rows

    title = f"Policy entropy by N: raw data + {window}-update moving average"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((width - (title_box[2] - title_box[0])) / 2, 20), title, fill="#111111", font=title_font)

    for panel_index, n_robots in enumerate(sorted(series)):
        row, col = divmod(panel_index, cols)
        panel_left = margin_x + col * (panel_w + gap_x)
        panel_top = margin_top + row * (panel_h + gap_y)
        bounds = (panel_left + 65, panel_top + 35, panel_left + panel_w - 15, panel_top + panel_h - 45)
        xs, raw, smooth = series[n_robots]
        x_range = (min(xs), max(xs))
        y_range = _range_with_padding(raw + smooth, robust=True)
        _draw_axes(draw, bounds, x_range, y_range, small_font, x_ticks=3, y_ticks=3)

        raw_points = _line_points(xs, raw, bounds, x_range, y_range)
        if len(raw_points) >= 2:
            draw.line(raw_points, fill="#d7d7d7", width=1)
        color = COLORS[panel_index % len(COLORS)]
        smooth_points = _line_points(xs, smooth, bounds, x_range, y_range)
        if len(smooth_points) >= 2:
            draw.line(smooth_points, fill=color, width=4)

        first_count = min(window, len(raw))
        start_mean = mean(raw[:first_count])
        end_mean = mean(raw[-first_count:])
        label = f"N={n_robots}    {_fmt(start_mean)} -> {_fmt(end_mean)}"
        draw.text((panel_left + 65, panel_top + 5), label, fill=color, font=panel_title_font)

    draw.text(
        (margin_x, height - 33),
        "Each panel has its own y-axis. Header values compare the first and last moving-average windows.",
        fill="#555555",
        font=small_font,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot smoothed PPO metrics for grouped N runs.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.window <= 0:
        raise ValueError("--window must be positive")

    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or (run_dir / "plots")).resolve()
    loaded: dict[str, dict[int, tuple[list[float], list[float], list[float]]]] = {}
    for metric in METRICS:
        metric_series: dict[int, tuple[list[float], list[float], list[float]]] = {}
        for metrics_path in sorted(run_dir.glob("n*/training_metrics.csv")):
            folder = metrics_path.parent.name
            if not folder.startswith("n") or not folder[1:].isdigit():
                continue
            n_robots = int(folder[1:])
            updates, values = _read_metric(metrics_path, metric)
            if values:
                metric_series[n_robots] = (updates, values, _moving_average(values, args.window))
        if metric_series:
            loaded[metric] = metric_series
            output_path = output_dir / f"smoothed_{metric}_w{args.window}.png"
            _plot_overlay(metric_series, metric, output_path, args.window)
            print(output_path)

    entropy_series = loaded.get("entropy")
    if entropy_series:
        output_path = output_dir / f"smoothed_entropy_panels_w{args.window}.png"
        _plot_entropy_panels(entropy_series, output_path, args.window)
        print(output_path)


if __name__ == "__main__":
    main()
