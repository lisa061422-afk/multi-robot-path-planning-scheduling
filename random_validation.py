"""Random validation runner for the current path-selection scheduler.

The generated cases are intentionally capped before solving.  In this model,
finite route sets are schedulable, but exact DFS can still explode
combinatorially.  The caps below keep the question as "is this small exact
case solved correctly?" instead of "how long can the tree grow?"
"""

from __future__ import annotations

import argparse
import html
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from coarse_scheduler import (
    apply_relaxed_entrance_headway,
    build_relaxed_vehicle_plan,
    search_dynamic_codesign_parallel_dfs_bb,
    search_relaxed_parallel_dfs_bb,
    write_interactive_solution_html,
)
from scheduler_models import RelaxedSearchResult, RelaxedVehiclePlan
from traffic_map import TrafficMap


@dataclass(frozen=True)
class CaseSpec:
    case_id: int
    map_name: str
    vehicle_requests: tuple[tuple[int, int, int, float], ...]
    t_headway: float
    max_hops: int
    max_paths: int


@dataclass(frozen=True)
class CaseStats:
    option_product: int
    max_potential_vehicles_per_resource: int
    max_option_visits_per_resource: int


@dataclass(frozen=True)
class RunStats:
    elapsed: float
    nodes: int
    leaves: int
    best_g: float
    delay: float
    path_extra: float
    html_path: str


def traffic_map_by_name(name: str) -> TrafficMap:
    if name == "paper_2x2":
        return TrafficMap.paper_2x2()
    if name == "grid_3x3":
        return TrafficMap.rectangular_grid(3, 3, name="grid_3x3")
    raise ValueError(f"unknown map name: {name}")


def make_relaxed_plans(
    tmap: TrafficMap,
    spec: CaseSpec,
    *,
    road_time: float,
) -> list[RelaxedVehiclePlan]:
    plans = [
        build_relaxed_vehicle_plan(
            tmap,
            vehicle_id=vehicle_id,
            entrance=entrance,
            exit=exit_,
            alpha0=alpha0,
            road_time=road_time,
            max_hops=spec.max_hops,
            max_paths=spec.max_paths,
        )
        for vehicle_id, entrance, exit_, alpha0 in spec.vehicle_requests
    ]
    return apply_relaxed_entrance_headway(plans, headway=spec.t_headway)


def option_product(plans: Sequence[RelaxedVehiclePlan]) -> int:
    product = 1
    for plan in plans:
        product *= len(plan.route_options)
    return product


def load_stats(plans: Sequence[RelaxedVehiclePlan]) -> CaseStats:
    potential_vehicles: dict[int, set[int]] = {}
    option_visits: dict[int, int] = {}
    for plan in plans:
        vehicle_resources = set()
        for option in plan.route_options:
            for resource in option.resource_sequence:
                vehicle_resources.add(resource)
                option_visits[resource] = option_visits.get(resource, 0) + 1
        for resource in vehicle_resources:
            potential_vehicles.setdefault(resource, set()).add(plan.vehicle_id)

    return CaseStats(
        option_product=option_product(plans),
        max_potential_vehicles_per_resource=max(
            (len(items) for items in potential_vehicles.values()),
            default=0,
        ),
        max_option_visits_per_resource=max(option_visits.values(), default=0),
    )


def case_is_safe(
    plans: Sequence[RelaxedVehiclePlan],
    *,
    max_option_product: int,
    max_resource_vehicles: int,
    max_resource_option_visits: int,
) -> bool:
    if any(len(plan.route_options) == 0 for plan in plans):
        return False
    stats = load_stats(plans)
    if max_resource_vehicles <= 0:
        max_resource_vehicles = float("inf")  # disable resource-vehicle cap
    return (
        stats.option_product <= max_option_product
        and stats.max_potential_vehicles_per_resource <= max_resource_vehicles
        and stats.max_option_visits_per_resource <= max_resource_option_visits
    )


def random_case(
    rng: random.Random,
    case_id: int,
    *,
    map_names: Sequence[str],
    min_robots: int,
    max_robots: int,
    road_time: float,
    max_option_product: int,
    max_resource_vehicles: int,
    max_resource_option_visits: int,
    max_vehicles_per_entrance: int,
    max_attempts: int = 500,
) -> tuple[CaseSpec, list[RelaxedVehiclePlan], CaseStats]:
    for _attempt in range(max_attempts):
        map_name = rng.choice(tuple(map_names))
        tmap = traffic_map_by_name(map_name)
        if map_name == "paper_2x2":
            n_vehicles = rng.randint(min_robots, max_robots)
            max_hops = 4
            max_paths = 3
        else:
            n_vehicles = rng.randint(min_robots, max_robots)
            max_hops = rng.choice((3, 4))
            max_paths = 3

        ports = list(tmap.port_ids)
        if (
            max_vehicles_per_entrance > 0
            and n_vehicles > len(ports) * max_vehicles_per_entrance
        ):
            raise RuntimeError(
                "max_vehicles_per_entrance is too small for sampled robot count in random_case"
            )
        entrance_counts: dict[int, int] = {port: 0 for port in ports}
        requests = []
        for vehicle_id in range(1, n_vehicles + 1):
            if max_vehicles_per_entrance > 0:
                available_entrances = [
                    port
                    for port, count in entrance_counts.items()
                    if count < max_vehicles_per_entrance
                ]
                if not available_entrances:
                    raise RuntimeError(
                        "could not sample entrance assignments under the max vehicles per "
                        "entrance cap"
                    )
                entrance = rng.choice(available_entrances)
            else:
                entrance = rng.choice(ports)
            exit_ = rng.choice(tuple(port for port in ports if port != entrance))
            alpha0 = rng.choice((0.0, 0.5, 1.0, 1.5))
            requests.append((vehicle_id, entrance, exit_, alpha0))
            entrance_counts[entrance] += 1

        spec = CaseSpec(
            case_id=case_id,
            map_name=map_name,
            vehicle_requests=tuple(requests),
            t_headway=rng.choice((0.0, 1.0, 2.0)),
            max_hops=max_hops,
            max_paths=max_paths,
        )
        plans = make_relaxed_plans(tmap, spec, road_time=road_time)
        if case_is_safe(
            plans,
            max_option_product=max_option_product,
            max_resource_vehicles=max_resource_vehicles,
            max_resource_option_visits=max_resource_option_visits,
        ):
            return spec, plans, load_stats(plans)

    raise RuntimeError(
        f"could not generate safe case {case_id} after {max_attempts} attempts"
    )


def assert_schedule_valid(
    result: RelaxedSearchResult,
    plans: Sequence[RelaxedVehiclePlan],
) -> None:
    if result.best_idx < 0 or not math.isfinite(result.best_g):
        raise AssertionError("solver did not find a finite complete solution")

    node = result.best_node
    if any(len(candidates) != 1 for candidates in node.route_candidates):
        raise AssertionError(f"best leaf has non-singleton route candidates: {node.route_candidates}")

    expected_by_vehicle: dict[int, list[tuple[int, float]]] = {}
    for plan, candidates in zip(plans, node.route_candidates):
        option = plan.route_options[candidates[0]]
        expected_by_vehicle[plan.vehicle_id] = list(
            zip(option.resource_sequence, option.execution_times)
        )

    actual_by_vehicle: dict[int, list] = {}
    for seg in sorted(result.best_schedule, key=lambda s: (s.vehicle_id, s.task_index)):
        actual_by_vehicle.setdefault(seg.vehicle_id, []).append(seg)
        if seg.start_time + 1e-9 < seg.requested_time:
            raise AssertionError(f"N{seg.vehicle_id} starts before request")
        if seg.end_time + 1e-9 < seg.start_time:
            raise AssertionError(f"N{seg.vehicle_id} has negative duration")

    for vehicle_id, expected in expected_by_vehicle.items():
        actual = actual_by_vehicle.get(vehicle_id, [])
        if len(actual) != len(expected):
            raise AssertionError(
                f"N{vehicle_id} scheduled {len(actual)} tasks; expected {len(expected)}"
            )
        for task_index, (seg, (resource, duration)) in enumerate(
            zip(actual, expected),
            start=1,
        ):
            if seg.task_index != task_index or seg.resource != resource:
                raise AssertionError(
                    f"N{vehicle_id} task {task_index}: got I{seg.resource}, "
                    f"expected I{resource}"
                )
            if abs((seg.end_time - seg.start_time) - duration) > 1e-7:
                raise AssertionError(
                    f"N{vehicle_id} task {task_index}: wrong duration "
                    f"{seg.end_time - seg.start_time} vs {duration}"
                )

    by_resource: dict[int, list] = {}
    for seg in result.best_schedule:
        by_resource.setdefault(seg.resource, []).append(seg)
    for resource, segs in by_resource.items():
        ordered = sorted(segs, key=lambda s: (s.start_time, s.end_time))
        for left, right in zip(ordered, ordered[1:]):
            if left.end_time > right.start_time + 1e-7:
                raise AssertionError(
                    f"I{resource} overlap: N{left.vehicle_id} and N{right.vehicle_id}"
                )

    delay = sum(seg.delay for seg in result.best_schedule)
    if abs(delay - node.g_delay) > 1e-7:
        raise AssertionError(f"delay mismatch: schedule={delay}, node={node.g_delay}")
    expected_g = node.g_delay + node.g_path
    if abs(expected_g - node.g) > 1e-7 or abs(expected_g - result.best_g) > 1e-7:
        raise AssertionError("objective mismatch")


def has_cutoff(log_lines: Iterable[str]) -> bool:
    text = "\n".join(log_lines).lower()
    return "deadline hit" in text or "max_nodes hit" in text


def write_compact_schedule_html(
    result: RelaxedSearchResult,
    path: Path,
    *,
    plans: Sequence[RelaxedVehiclePlan],
    tmap: TrafficMap,
    elapsed: float,
) -> None:
    """Write a lightweight optimal-schedule viewer.

    The full decision-tree HTML can become tens of MB for 5-6 robots.  This
    compact view keeps only the selected co-design solution.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    node = result.best_node
    segments = sorted(result.best_schedule, key=lambda s: (s.resource, s.start_time))
    attempts = sorted(node.attempts, key=lambda a: (a.resource, a.start_time))
    max_t = max(
        [1.0]
        + [seg.end_time for seg in segments]
        + [seg.requested_time for seg in segments]
        + [attempt.end_time for attempt in attempts]
    )
    resources = list(tmap.intersection_ids)
    vehicle_ids = [plan.vehicle_id for plan in plans]
    palette = [
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#9333ea",
        "#ea580c",
        "#0891b2",
        "#be123c",
        "#4f46e5",
        "#65a30d",
        "#c2410c",
    ]
    color_by_vehicle = {
        vehicle_id: palette[index % len(palette)]
        for index, vehicle_id in enumerate(vehicle_ids)
    }

    route_rows = []
    for index, plan in enumerate(plans):
        candidates = node.route_candidates[index] if index < len(node.route_candidates) else ()
        option_index = candidates[0] if candidates else 0
        option = plan.route_options[option_index]
        route_text = " -> ".join(f"I{i}" for i in option.intersections)
        turns = ", ".join(item.turn for item in option.traversals)
        route_rows.append(
            "<tr>"
            f"<td>N{plan.vehicle_id}</td>"
            f"<td>P{plan.entrance} -> P{plan.exit}</td>"
            f"<td>option {option_index + 1}</td>"
            f"<td>{html.escape(route_text)}</td>"
            f"<td>{html.escape(turns)}</td>"
            "</tr>"
        )

    x0 = 76
    x1 = 980
    row_h = 26
    height = 54 + len(resources) * row_h + 36

    def x_of(t: float) -> float:
        return x0 + (t / max_t) * (x1 - x0)

    svg_parts = [
        f'<svg viewBox="0 0 1040 {height}" width="100%" role="img">',
        '<rect width="1040" height="100%" fill="#ffffff"/>',
    ]
    for row, resource in enumerate(resources):
        y = 28 + row * row_h
        svg_parts.append(
            f'<text x="16" y="{y + 15}" font-size="12" font-weight="700">I{resource}</text>'
        )
        svg_parts.append(
            f'<rect x="{x0}" y="{y}" width="{x1 - x0}" height="18" '
            'fill="#f8fafc" stroke="#dbe3ef"/>'
        )
        for seg in [item for item in segments if item.resource == resource]:
            xs = x_of(seg.start_time)
            xe = x_of(seg.end_time)
            xa = x_of(seg.requested_time)
            if seg.start_time > seg.requested_time + 1e-9:
                svg_parts.append(
                    f'<rect x="{xa:.2f}" y="{y}" width="{max(1.0, xs - xa):.2f}" '
                    'height="18" fill="#e5e7eb" stroke="#94a3b8"/>'
                )
            color = color_by_vehicle[seg.vehicle_id]
            svg_parts.append(
                f'<rect x="{xs:.2f}" y="{y}" width="{max(2.0, xe - xs):.2f}" '
                f'height="18" fill="{color}" opacity="0.82"/>'
            )
            svg_parts.append(
                f'<text x="{(xs + xe) / 2:.2f}" y="{y + 12}" text-anchor="middle" '
                'font-size="10" font-weight="700" fill="white">'
                f'N{seg.vehicle_id} K{seg.task_index}</text>'
            )
        for attempt in [item for item in attempts if item.resource == resource]:
            xs = x_of(attempt.start_time)
            xe = x_of(attempt.end_time)
            svg_parts.append(
                f'<rect x="{xs:.2f}" y="{y}" width="{max(2.0, xe - xs):.2f}" '
                'height="18" fill="none" stroke="#f97316" stroke-width="2" '
                'stroke-dasharray="4 2"/>'
            )

    axis_y = 36 + len(resources) * row_h
    svg_parts.append(
        f'<line x1="{x0}" y1="{axis_y}" x2="{x1}" y2="{axis_y}" stroke="#111827"/>'
    )
    tick_count = 8
    for tick in range(tick_count + 1):
        t = max_t * tick / tick_count
        x = x_of(t)
        svg_parts.append(
            f'<line x1="{x:.2f}" y1="{axis_y}" x2="{x:.2f}" y2="{axis_y + 5}" '
            'stroke="#111827"/>'
        )
        svg_parts.append(
            f'<text x="{x:.2f}" y="{axis_y + 19}" text-anchor="middle" '
            'font-size="10">{t:.1f}</text>'
        )
    svg_parts.append("</svg>")

    schedule_rows = []
    for seg in sorted(result.best_schedule, key=lambda s: (s.start_time, s.resource)):
        schedule_rows.append(
            "<tr>"
            f"<td>N{seg.vehicle_id}</td>"
            f"<td>K{seg.task_index}</td>"
            f"<td>I{seg.resource}</td>"
            f"<td>{seg.requested_time:.3f}</td>"
            f"<td>{seg.start_time:.3f}</td>"
            f"<td>{seg.end_time:.3f}</td>"
            f"<td>{seg.delay:.3f}</td>"
            "</tr>"
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Compact Co-Design Schedule</title>
  <style>
    body {{ margin:24px; font-family:Arial, sans-serif; color:#111827; background:#f8fafc; }}
    h1 {{ font-size:22px; margin:0 0 8px; }}
    h2 {{ font-size:16px; margin:22px 0 8px; }}
    .meta {{ display:flex; flex-wrap:wrap; gap:10px; margin:12px 0; }}
    .pill {{ background:white; border:1px solid #dbe3ef; padding:7px 10px; border-radius:6px; font-size:13px; }}
    .panel {{ background:white; border:1px solid #dbe3ef; padding:12px; margin-top:12px; }}
    table {{ width:100%; border-collapse:collapse; background:white; }}
    th, td {{ border-bottom:1px solid #e5e7eb; padding:7px 9px; text-align:left; font-size:12px; }}
    th {{ background:#e5edf7; font-weight:700; }}
  </style>
</head>
<body>
  <h1>Compact Co-Design Schedule</h1>
  <div class="meta">
    <div class="pill">J* = {result.best_g:.6f}</div>
    <div class="pill">delay = {node.g_delay:.6f}</div>
    <div class="pill">path_extra = {node.g_path:.6f}</div>
    <div class="pill">compute time = {elapsed:.6f}s</div>
    <div class="pill">nodes = {len(result.nodes)}</div>
    <div class="pill">leaves = {len(result.leaves)}</div>
  </div>
  <div class="panel">
    {''.join(svg_parts)}
  </div>
  <h2>Selected Paths</h2>
  <table>
    <thead><tr><th>Vehicle</th><th>OD</th><th>Option</th><th>Path</th><th>Turns</th></tr></thead>
    <tbody>{''.join(route_rows)}</tbody>
  </table>
  <h2>Scheduled Tasks</h2>
  <table>
    <thead><tr><th>Vehicle</th><th>Task</th><th>Resource</th><th>Request</th><th>Start</th><th>End</th><th>Delay</th></tr></thead>
    <tbody>{''.join(schedule_rows)}</tbody>
  </table>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def run_case(
    spec: CaseSpec,
    plans: Sequence[RelaxedVehiclePlan],
    *,
    tmap: TrafficMap,
    html_path: Path,
    compare_full_path_baseline: bool,
    deadline: float | None,
    max_nodes: int | None,
    frontier_depth: int,
    max_workers: int,
    full_tree_html: bool,
    max_terminal_paths: int | None,
) -> RunStats:
    start = time.perf_counter()
    result = search_dynamic_codesign_parallel_dfs_bb(
        plans,
        frontier_depth=frontier_depth,
        max_workers=max_workers,
        deadline=deadline,
        max_nodes=max_nodes,
        branch_and_bound=True,
        verbose=False,
    )
    elapsed = time.perf_counter() - start
    if has_cutoff(result.log):
        raise AssertionError("dynamic solver hit deadline/max_nodes; result is best-so-far")
    assert_schedule_valid(result, plans)

    if compare_full_path_baseline:
        baseline = search_relaxed_parallel_dfs_bb(
            plans,
            deadline=deadline,
            max_nodes=max_nodes,
            branch_and_bound=True,
            max_workers=max_workers,
            verbose=False,
        )
        if has_cutoff(baseline.log):
            raise AssertionError("baseline solver hit deadline/max_nodes; result is best-so-far")
        if abs(result.best_g - baseline.best_g) > 1e-7:
            raise AssertionError(
                f"dynamic best_g={result.best_g:.9f}, "
                f"full-path baseline best_g={baseline.best_g:.9f}"
            )

    if full_tree_html:
        write_interactive_solution_html(
            result,
            html_path,
            plans=plans,
            tmap=tmap,
            max_terminal_paths=max_terminal_paths,
        )
    else:
        write_compact_schedule_html(
            result,
            html_path,
            plans=plans,
            tmap=tmap,
            elapsed=elapsed,
        )

    return RunStats(
        elapsed=elapsed,
        nodes=len(result.nodes),
        leaves=len(result.leaves),
        best_g=result.best_g,
        delay=result.best_node.g_delay,
        path_extra=result.best_node.g_path,
        html_path=str(html_path),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--min-robots", type=int, default=2)
    parser.add_argument("--max-robots", type=int, default=12)
    parser.add_argument("--road-time", type=float, default=3.0)
    parser.add_argument("--frontier-depth", type=int, default=2)
    parser.add_argument("--max-workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument(
        "--maps",
        default="paper_2x2",
        help="comma-separated map names; default is paper_2x2",
    )
    parser.add_argument("--max-option-product", type=int, default=81)
    parser.add_argument(
        "--max-resource-vehicles",
        type=int,
        default=0,
        help="max potential vehicles per resource in generated candidates; <=0 disables this cap",
    )
    parser.add_argument("--max-resource-option-visits", type=int, default=8)
    parser.add_argument(
        "--max-vehicles-per-entrance",
        type=int,
        default=0,
        help="max number of vehicles sharing one entrance in random case generation; 0 means no limit",
    )
    parser.add_argument("--deadline", type=float, default=None)
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--timing-csv", default="output/random_validation_times.csv")
    parser.add_argument("--html-dir", default="output/random_validation_html")
    parser.add_argument("--summary-html", default="output/random_validation_summary.html")
    parser.add_argument(
        "--full-tree-html",
        dest="full_tree_html",
        action="store_true",
        default=True,
        help="save interactive decision-tree HTML; this is the default",
    )
    parser.add_argument(
        "--compact-html",
        dest="full_tree_html",
        action="store_false",
        help="save compact best-schedule HTML instead of the interactive decision tree",
    )
    parser.add_argument(
        "--max-terminal-paths",
        type=int,
        default=50,
        help="maximum terminal paths to keep in full-tree HTML; optimal is always included",
    )
    parser.add_argument(
        "--no-baseline-compare",
        action="store_true",
        help="skip comparison with exhaustive full-path-choice baseline",
    )
    return parser.parse_args()


def write_summary_html(
    rows: Sequence[dict[str, str]],
    path: Path,
    *,
    title: str = "Random Co-Design Validation Results",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def cell(text: str) -> str:
        return html.escape(text)

    table_rows = []
    for row in rows:
        html_path = row["html_href"]
        table_rows.append(
            "<tr>"
            f"<td>{cell(row['case_id'])}</td>"
            f"<td>{cell(row['map'])}</td>"
            f"<td>{cell(row['robots'])}</td>"
            f"<td>{cell(row['t_headway'])}</td>"
            f"<td>{cell(row['elapsed_seconds'])}</td>"
            f"<td>{cell(row['optimal_cost'])}</td>"
            f"<td>{cell(row['delay'])}</td>"
            f"<td>{cell(row['path_extra'])}</td>"
            f"<td>{cell(row['nodes'])}</td>"
            f"<td>{cell(row['leaves'])}</td>"
            f"<td><a href=\"{cell(html_path)}\">open schedule</a></td>"
            f"<td>{cell(row['requests'])}</td>"
            "</tr>"
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{cell(title)}</title>
  <style>
    body {{
      margin: 24px;
      font-family: Arial, sans-serif;
      color: #111827;
      background: #f8fafc;
    }}
    h1 {{
      margin: 0 0 14px;
      font-size: 22px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: white;
      border: 1px solid #dbe3ef;
    }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid #e5e7eb;
      text-align: left;
      font-size: 12px;
      vertical-align: top;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #e5edf7;
      font-weight: 700;
    }}
    a {{
      color: #1d4ed8;
      font-weight: 700;
      text-decoration: none;
    }}
  </style>
</head>
<body>
  <h1>{cell(title)}</h1>
  <table>
    <thead>
      <tr>
        <th>Case</th>
        <th>Map</th>
        <th>Robots</th>
        <th>T_headway</th>
        <th>Compute Time</th>
        <th>Optimal Cost</th>
        <th>Delay</th>
        <th>Path Extra</th>
        <th>Nodes</th>
        <th>Leaves</th>
        <th>Schedule</th>
        <th>Requests</th>
      </tr>
    </thead>
    <tbody>
      {''.join(table_rows)}
    </tbody>
  </table>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    total_elapsed = 0.0
    summary_rows: list[dict[str, str]] = []
    map_names = tuple(name.strip() for name in args.maps.split(",") if name.strip())
    if not map_names:
        raise ValueError("--maps must include at least one map name")
    if args.min_robots < 1 or args.max_robots < args.min_robots:
        raise ValueError("--min-robots/--max-robots must define a valid positive range")
    if args.max_vehicles_per_entrance < 0:
        raise ValueError("--max-vehicles-per-entrance must be non-negative")
    timing_path = Path(args.timing_csv)
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    html_dir = Path(args.html_dir)
    html_dir.mkdir(parents=True, exist_ok=True)
    timing_path.write_text(
        "case_id,map,robots,t_headway,option_product,max_resource_vehicles,"
        "nodes,leaves,elapsed_seconds,optimal_cost,delay,path_extra,html_path,requests\n",
        encoding="utf-8",
    )

    print(
        "Random scheduler validation: "
        f"cases={args.cases}, seed={args.seed}, workers={args.max_workers}, "
        f"frontier_depth={args.frontier_depth}, maps={','.join(map_names)}, "
        f"robots={args.min_robots}..{args.max_robots}",
        flush=True,
    )
    print(
        "Caps: "
        f"option_product<={args.max_option_product}, "
        f"resource_vehicles<={args.max_resource_vehicles if args.max_resource_vehicles > 0 else 'unlimited'}, "
        f"resource_option_visits<={args.max_resource_option_visits}, "
        f"max_vehicles_per_entrance={args.max_vehicles_per_entrance if args.max_vehicles_per_entrance > 0 else 'unlimited'}",
        flush=True,
    )
    print(f"Timing log: {timing_path}", flush=True)
    print(f"HTML dir: {html_dir}", flush=True)

    for case_id in range(1, args.cases + 1):
        spec, plans, stats = random_case(
            rng,
            case_id,
            map_names=map_names,
            min_robots=args.min_robots,
            max_robots=args.max_robots,
            road_time=args.road_time,
            max_option_product=args.max_option_product,
            max_resource_vehicles=args.max_resource_vehicles,
            max_resource_option_visits=args.max_resource_option_visits,
            max_vehicles_per_entrance=args.max_vehicles_per_entrance,
        )
        tmap = traffic_map_by_name(spec.map_name)
        html_path = html_dir / f"case_{case_id:02d}.html"
        print(
            f"[{case_id:02d}] starting {spec.map_name} robots={len(plans)} "
            f"T_headway={spec.t_headway:g} options={stats.option_product} "
            f"maxResVeh={stats.max_potential_vehicles_per_resource}",
            flush=True,
        )
        run_stats = run_case(
            spec,
            plans,
            tmap=tmap,
            html_path=html_path,
            compare_full_path_baseline=not args.no_baseline_compare,
            deadline=args.deadline,
            max_nodes=args.max_nodes,
            frontier_depth=args.frontier_depth,
            max_workers=args.max_workers,
            full_tree_html=args.full_tree_html,
            max_terminal_paths=args.max_terminal_paths,
        )
        total_elapsed += run_stats.elapsed
        requests = " ".join(
            f"N{vid}:P{ent}->P{ext}@{alpha:g}"
            for vid, ent, ext, alpha in spec.vehicle_requests
        )
        with timing_path.open("a", encoding="utf-8") as fh:
            fh.write(
                f"{case_id},{spec.map_name},{len(plans)},{spec.t_headway:g},"
                f"{stats.option_product},"
                f"{stats.max_potential_vehicles_per_resource},"
                f"{run_stats.nodes},{run_stats.leaves},{run_stats.elapsed:.6f},"
                f"{run_stats.best_g:.9f},{run_stats.delay:.9f},"
                f"{run_stats.path_extra:.9f},\"{run_stats.html_path}\",\"{requests}\"\n"
            )
        summary_rows.append(
            {
                "case_id": str(case_id),
                "map": spec.map_name,
                "robots": str(len(plans)),
                "t_headway": f"{spec.t_headway:g}",
                "option_product": str(stats.option_product),
                "max_resource_vehicles": str(stats.max_potential_vehicles_per_resource),
                "nodes": str(run_stats.nodes),
                "leaves": str(run_stats.leaves),
                "elapsed_seconds": f"{run_stats.elapsed:.6f}",
                "optimal_cost": f"{run_stats.best_g:.9f}",
                "delay": f"{run_stats.delay:.9f}",
                "path_extra": f"{run_stats.path_extra:.9f}",
                "html_path": run_stats.html_path,
                "html_href": os.path.relpath(
                    run_stats.html_path,
                    start=Path(args.summary_html).parent,
                ).replace(os.sep, "/"),
                "requests": requests,
            }
        )
        print(
            f"[{case_id:02d}] completed {spec.map_name} robots={len(plans)} "
            f"T_headway={spec.t_headway:g} options={stats.option_product} "
            f"maxResVeh={stats.max_potential_vehicles_per_resource} "
            f"nodes={run_stats.nodes} leaves={run_stats.leaves} "
            f"time={run_stats.elapsed:.3f}s "
            f"J*={run_stats.best_g:.3f} "
            f"delay={run_stats.delay:.3f} "
            f"path_extra={run_stats.path_extra:.3f} "
            f"html={run_stats.html_path} | {requests}",
            flush=True,
        )

    print(
        f"All {args.cases} random cases completed in "
        f"{total_elapsed:.3f}s solver time.",
        flush=True,
    )
    summary_path = Path(args.summary_html)
    write_summary_html(summary_rows, summary_path)
    print(f"Summary HTML: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
