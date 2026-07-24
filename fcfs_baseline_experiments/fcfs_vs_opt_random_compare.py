"""Compare fixed-shortest-path FCFS against optimal dynamic co-design on random 3x3 cases."""

from __future__ import annotations

import argparse
import html
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from coarse_scheduler import (
    build_relaxed_vehicle_plan,
    apply_relaxed_entrance_headway,
    search_dynamic_codesign_dfs_bb,
    write_interactive_solution_html,
)
from fcfs_baseline_experiments.independent_fcfs_shortest_path_scheduler import (
    search_fixed_shortest_fcfs_dfs_bb,
)
from scheduler_models import RelaxedVehiclePlan
from traffic_map import TrafficMap


def make_plans(rng: random.Random, case_id: int, *, n_vehicles: int, road_time: float) -> Tuple[RelaxedVehiclePlan, str]:
    tmap = TrafficMap.rectangular_grid(3, 3, name="grid_3x3_case_compare")
    ports = list(tmap.port_ids)
    requests = []
    for vehicle_id in range(1, n_vehicles + 1):
        entrance = rng.choice(ports)
        exit_ = rng.choice([item for item in ports if item != entrance])
        alpha0 = rng.choice((0.0, 0.5, 1.0, 1.5, 2.0))
        requests.append((vehicle_id, entrance, exit_, alpha0))

    plans = [
        build_relaxed_vehicle_plan(
            tmap,
            vehicle_id=vehicle_id,
            entrance=entrance,
            exit=exit_,
            alpha0=alpha0,
            road_time=road_time,
            max_hops=5,
            max_paths=6,
        )
        for vehicle_id, entrance, exit_, alpha0 in requests
    ]
    plans = apply_relaxed_entrance_headway(plans, headway=0.0)
    requests_text = " ".join(
        f"N{vehicle_id}:P{entrance}->P{exit_}@{alpha0:g}"
        for vehicle_id, entrance, exit_, alpha0 in requests
    )
    return plans, requests_text


def safe_run_optimal(plans: list[RelaxedVehiclePlan], *, deadline: float | None, max_nodes: int | None):
    result = search_dynamic_codesign_dfs_bb(
        plans,
        deadline=deadline,
        max_nodes=max_nodes,
        branch_and_bound=True,
        verbose=False,
    )
    if result.best_idx < 0:
        raise RuntimeError("optimal solver did not finish")
    if not any("deadline hit" in line.lower() or "max_nodes hit" in line.lower() for line in result.log):
        return result
    raise RuntimeError("optimal solver cut off by deadline/max_nodes")


def write_compare_html(
    case_dir: Path,
    case_id: int,
    *,
    fcfs_cost: float,
    optimal_cost: float,
    fcfs_routes: str,
    optimal_routes: str,
    fcfs_path: Path,
    optimal_path: Path,
    requests: str,
):
    better = "FCFS" if fcfs_cost <= optimal_cost else "Optimal"
    gap = optimal_cost - fcfs_cost
    case_dir.mkdir(parents=True, exist_ok=True)
    path = case_dir / "compare.html"
    body = """<!doctype html>
<html>
<head><meta charset=\"utf-8\"><title>FCFS vs Optimal Case %(case_id)s</title></head>
<body>
  <h1>FCFS vs Optimal Case %(case_id)s</h1>
  <p><strong>Requests:</strong> %(requests)s</p>
  <h2>Numerics</h2>
  <table border=\"1\" cellpadding=\"6\" cellspacing=\"0\">
    <tr><th>FCFS cost</th><td>%(fcfs_cost).6f</td></tr>
    <tr><th>Optimal cost</th><td>%(optimal_cost).6f</td></tr>
    <tr><th>Gap (optimal - FCFS)</th><td>%(gap).6f</td></tr>
    <tr><th>Better</th><td>%(better)s</td></tr>
  </table>
  <h2>Selected path indices</h2>
  <p><strong>FCFS (fixed shortest):</strong> %(fcfs_routes)s</p>
  <p><strong>Optimal:</strong> %(optimal_routes)s</p>
  <h2>Open full solutions</h2>
  <p><a href=\"%(fcfs_link)s\">Open FCFS schedule</a> | <a href=\"%(opt_link)s\">Open Optimal schedule</a></p>
  <h2>Gantt (resource-time)</h2>
  <p><a href=\"%(fcfs_schedule_link)s\">Open FCFS Gantt</a> | <a href=\"%(optimal_schedule_link)s\">Open Optimal Gantt</a></p>
  <div style=\"display:flex; gap:12px; flex-wrap:wrap;\">
    <div style=\"flex:1; min-width:420px;\">
      <h3>FCFS</h3>
      <iframe id=\"fcfsGanttFrame\" src=\"%(fcfs_schedule_link)s\" width=\"100%%\" height=\"560\" style=\"border:1px solid #dbe3ef;\"></iframe>
    </div>
    <div style=\"flex:1; min-width:420px;\">
      <h3>Optimal</h3>
      <iframe id=\"optimalGanttFrame\" src=\"%(optimal_schedule_link)s\" width=\"100%%\" height=\"560\" style=\"border:1px solid #dbe3ef;\"></iframe>
    </div>
  </div>
  <script>
    function jumpToSchedule(frameId) {
      var frame = document.getElementById(frameId);
      if (!frame) {
        return;
      }
      function ensureHash() {
        var src = frame.getAttribute("src") || "";
        if (src.indexOf("#schedulePanel") === -1) {
          frame.setAttribute("src", src + "#schedulePanel");
        }
      }
      function attemptScroll() {
        try {
          var doc = frame.contentWindow && frame.contentWindow.document;
          var target = doc ? doc.getElementById("schedulePanel") : null;
          if (!target) {
            return;
          }
          var pane = target.closest ? target.closest("section.pane") : null;
          if (pane && pane.scrollTo) {
            pane.scrollTop = Math.max(0, target.offsetTop - 8);
          }
          if (doc.scrollingElement) {
            doc.scrollingElement.scrollTop = Math.max(0, target.offsetTop);
          }
          if (target.scrollIntoView) {
            target.scrollIntoView({ block: "start", inline: "nearest" });
          }
        } catch (_) {
          return;
        }
      }
      function scheduleScroll() {
        attemptScroll();
        setTimeout(attemptScroll, 80);
        setTimeout(attemptScroll, 220);
        setTimeout(attemptScroll, 420);
      }
      frame.addEventListener("load", function onLoad() {
        scheduleScroll();
      }, { once: true });
      ensureHash();
      try {
        frame.src = frame.src;
      } catch (_) {
      }
      var doc = frame.contentWindow && frame.contentWindow.document;
      if (doc && doc.readyState === "complete") {
        scheduleScroll();
      } else {
        setTimeout(scheduleScroll, 40);
      }
    }
    jumpToSchedule("fcfsGanttFrame");
    jumpToSchedule("optimalGanttFrame");
  </script>
</body>
</html>
""" % {
        "case_id": case_id,
        "requests": html.escape(requests),
        "fcfs_cost": fcfs_cost,
        "optimal_cost": optimal_cost,
        "gap": gap,
        "better": better,
        "fcfs_routes": html.escape(fcfs_routes),
        "optimal_routes": html.escape(optimal_routes),
        "fcfs_link": fcfs_path.name,
        "opt_link": optimal_path.name,
        "fcfs_schedule_link": f"{fcfs_path.name}#schedulePanel",
        "optimal_schedule_link": f"{optimal_path.name}#schedulePanel",
        "optimal_link": optimal_path.name,
    }
    path.write_text(body, encoding="utf-8")
    return path

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cases", type=int, default=6)
    parser.add_argument("--vehicles-min", type=int, default=6)
    parser.add_argument("--vehicles-max", type=int, default=12)
    parser.add_argument("--road-time", type=float, default=3.0)
    parser.add_argument("--deadline", type=float, default=None)
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument(
        "--output-root",
        default="C:/Users/rwang26/Documents/Codex/2026-07-23/continue-the-previous-task-please-first/outputs/fcfs_vs_opt_standalone_3x3",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "results.csv"
    html_lines: List[str] = [
        "case_id,vehicles,fcfs_cost,optimal_cost,better,gap,fcfs_routes,optimal_routes,fcfs_html,optimal_html,compare_html,requests",
    ]

    written = 0
    attempts = 0
    case_id = 1
    while written < args.cases and attempts < args.cases * 8:
        attempts += 1
        n_vehicles = rng.randint(args.vehicles_min, args.vehicles_max)
        plans, requests = make_plans(
            rng,
            case_id=case_id,
            n_vehicles=n_vehicles,
            road_time=args.road_time,
        )
        if any(len(plan.route_options) == 0 for plan in plans):
            continue

        try:
            t0 = time.perf_counter()
            optimal = safe_run_optimal(
                plans,
                deadline=args.deadline,
                max_nodes=args.max_nodes,
            )
            t_opt = time.perf_counter() - t0

            t0 = time.perf_counter()
            fcfs = search_fixed_shortest_fcfs_dfs_bb(plans, verbose=False)
            t_fcfs = time.perf_counter() - t0
        except Exception as exc:
            print(f"[SKIP] case {case_id:02d} due runtime error: {exc}")
            case_id += 1
            continue

        case_dir = output_root / f"case_{case_id:02d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        fcfs_html = case_dir / "fcfs.html"
        opt_html = case_dir / "optimal.html"
        write_interactive_solution_html(
            fcfs,
            fcfs_html,
            plans=plans,
            max_terminal_paths=1,
        )
        write_interactive_solution_html(
            optimal,
            opt_html,
            plans=plans,
            max_terminal_paths=1,
        )
        compare = write_compare_html(
            case_dir,
            case_id=case_id,
            fcfs_cost=fcfs.best_g,
            optimal_cost=optimal.best_g,
            fcfs_routes=", ".join(
                f"N{plans[i].vehicle_id}:P{fcfs.best_node.route_choices[i] + 1}"
                for i in range(len(plans))
            ),
            optimal_routes=", ".join(
                f"N{plans[i].vehicle_id}:P{(optimal.best_node.route_candidates[i][0] + 1)}"
                if optimal.best_node.route_candidates[i]
                else "N%d:-" % plans[i].vehicle_id
                for i in range(len(plans))
            ),
            fcfs_path=fcfs_html,
            optimal_path=opt_html,
            requests=requests,
        )
        better = "optimal" if optimal.best_g < fcfs.best_g else "fcfs"
        gap = optimal.best_g - fcfs.best_g
        fcfs_routes = ", ".join(
            f"N{plans[i].vehicle_id}:P{fcfs.best_node.route_choices[i] + 1}"
            for i in range(len(plans))
        )
        optimal_routes = ", ".join(
            f"N{plans[i].vehicle_id}:P{optimal.best_node.route_candidates[i][0] + 1}"
            if optimal.best_node.route_candidates[i]
            else f"N{plans[i].vehicle_id}:-"
            for i in range(len(plans))
        )
        row = (
            f"{case_id},{n_vehicles},{fcfs.best_g:.9f},{optimal.best_g:.9f},{better},{gap:.9f},"
            f"\"{fcfs_routes}\",\"{optimal_routes}\","
            f"{fcfs_html},{opt_html},{compare},\"{requests}\""
        )
        html_lines.append(row)
        print(
            f"[CASE {case_id:02d}] veh={n_vehicles} fcfs={fcfs.best_g:.4f} "
            f"opt={optimal.best_g:.4f} better={better} "
            f"t_fcfs={t_fcfs:.3f}s t_opt={t_opt:.3f}s"
        )
        written += 1
        case_id += 1

    csv_path.write_text("\n".join(html_lines) + "\n", encoding="utf-8")
    print(f"results: {written} cases generated, saved to {csv_path}")


if __name__ == "__main__":
    main()
