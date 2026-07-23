"""Greedy PPO evaluation and optional exact DFS comparison."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
import statistics
from pathlib import Path
from typing import Sequence

import torch

from coarse_scheduler import search_dynamic_codesign_dfs_bb
from scheduler_models import RelaxedVehiclePlan

from .encoding import BranchEncoder, EncodingConfig
from .environment import DecisionTreeEnv
from .networks import BranchScoringActor


@dataclass(frozen=True)
class EvaluationResult:
    ppo_cost: float
    exact_cost: float | None
    absolute_gap: float | None
    relative_gap: float | None
    decisions: int
    exact_status: str


def load_actor_checkpoint(
    path: str,
    *,
    device: torch.device | str = "cpu",
) -> tuple[BranchScoringActor, EncodingConfig, dict]:
    device = torch.device(device)
    payload = torch.load(path, map_location=device, weights_only=True)
    actor = BranchScoringActor(
        int(payload["state_dim"]),
        int(payload["action_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        hidden_layers=int(payload.get("actor_hidden_layers", 2)),
    )
    actor.load_state_dict(payload["actor_state_dict"])
    actor.to(device)
    actor.eval()
    encoding_config = EncodingConfig(**payload["encoding_config"])
    return actor, encoding_config, payload


def greedy_policy_cost(
    actor: BranchScoringActor,
    plans: Sequence[RelaxedVehiclePlan],
    *,
    encoding_config: EncodingConfig = EncodingConfig(),
    device: torch.device | str = "cpu",
) -> tuple[float, int]:
    env = DecisionTreeEnv(plans)
    encoder = BranchEncoder(plans, encoding_config)
    device = torch.device(device)
    actor.eval()
    node, branches, terminated = env.reset()
    with torch.no_grad():
        while not terminated:
            state = encoder.encode_state(node).to(device)
            actions = encoder.encode_actions(node, branches).to(device)
            action_index = int(torch.argmax(actor(state, actions)).item())
            result = env.step(action_index)
            node, branches, terminated = (
                result.node,
                result.branches,
                result.terminated,
            )
    return float(node.g), env.decision_count


def evaluate_against_exact(
    actor: BranchScoringActor,
    plans: Sequence[RelaxedVehiclePlan],
    *,
    encoding_config: EncodingConfig = EncodingConfig(),
    device: torch.device | str = "cpu",
    run_exact: bool = True,
) -> EvaluationResult:
    ppo_cost, decisions = greedy_policy_cost(
        actor,
        plans,
        encoding_config=encoding_config,
        device=device,
    )
    if not run_exact:
        return EvaluationResult(
            ppo_cost=ppo_cost,
            exact_cost=None,
            absolute_gap=None,
            relative_gap=None,
            decisions=decisions,
            exact_status="skipped",
        )

    exact = search_dynamic_codesign_dfs_bb(
        plans,
        branch_and_bound=True,
        verbose=False,
    )
    exact_cost = float(exact.best_g)
    absolute_gap = ppo_cost - exact_cost
    if abs(exact_cost) < 1e-9:
        relative_gap = 0.0 if abs(ppo_cost) < 1e-9 else math.inf
    else:
        relative_gap = absolute_gap / abs(exact_cost)
    status = "solved" if math.isfinite(exact.best_g) and exact.best_idx >= 0 else "unsolved"
    return EvaluationResult(
        ppo_cost=ppo_cost,
        exact_cost=float(exact.best_g),
        absolute_gap=float(absolute_gap),
        relative_gap=float(relative_gap),
        decisions=decisions,
        exact_status=status,
    )


def evaluate_against_exact_safely(
    actor: BranchScoringActor,
    plans: Sequence[RelaxedVehiclePlan],
    *,
    encoding_config: EncodingConfig = EncodingConfig(),
    device: torch.device | str = "cpu",
    exact: bool = True,
    exact_deadline: float | None = None,
    exact_max_nodes: int | None = None,
) -> EvaluationResult:
    ppo_cost, decisions = greedy_policy_cost(
        actor,
        plans,
        encoding_config=encoding_config,
        device=device,
    )
    if not exact:
        return EvaluationResult(
            ppo_cost=ppo_cost,
            exact_cost=None,
            absolute_gap=None,
            relative_gap=None,
            decisions=decisions,
            exact_status="skipped",
        )

    exact_result = search_dynamic_codesign_dfs_bb(
        plans,
        branch_and_bound=True,
        verbose=False,
        deadline=exact_deadline,
        max_nodes=exact_max_nodes,
    )
    solved = (
        exact_result.best_idx >= 0
        and math.isfinite(exact_result.best_g)
        and len(exact_result.nodes) > 0
    )
    status = "solved"
    for log_item in exact_result.log:
        if "deadline hit" in log_item or "max_nodes hit" in log_item:
            status = "partial"
            break
    if not solved:
        return EvaluationResult(
            ppo_cost=ppo_cost,
            exact_cost=None,
            absolute_gap=None,
            relative_gap=None,
            decisions=decisions,
            exact_status="unsolved",
        )

    exact_cost = float(exact_result.best_g)
    absolute_gap = ppo_cost - exact_cost
    if abs(exact_cost) < 1e-9:
        relative_gap = 0.0 if abs(ppo_cost) < 1e-9 else math.inf
    else:
        relative_gap = absolute_gap / abs(exact_cost)
    return EvaluationResult(
        ppo_cost=float(ppo_cost),
        exact_cost=float(exact_cost),
        absolute_gap=float(absolute_gap),
        relative_gap=float(relative_gap),
        decisions=int(decisions),
        exact_status=status,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--n-robots", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--min-initial-release", type=float, default=0.0)
    parser.add_argument("--max-initial-release", type=float, default=2.0)
    parser.add_argument("--initial-release-step", type=float, default=0.5)
    parser.add_argument(
        "--fix-shortest-paths",
        action="store_true",
        help="evaluate with each vehicle locked to shortest route; scheduling-only baseline",
    )
    parser.add_argument(
        "--exact-deadline",
        type=float,
        default=None,
        help="time budget in seconds for exact solver; no limit by default",
    )
    parser.add_argument(
        "--exact-max-nodes",
        type=int,
        default=None,
        help="node cap for exact solver; no cap by default",
    )
    parser.add_argument(
        "--metrics-csv",
        default="",
        help="optional CSV output path for batch evaluation rows",
    )
    return parser.parse_args()


def main() -> None:
    from .cases import ThreeByThreeCaseFactory

    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    if args.n_robots <= 0:
        raise ValueError("--n-robots must be positive")
    if args.initial_release_step <= 0:
        raise ValueError("--initial-release-step must be positive")
    if args.min_initial_release > args.max_initial_release:
        raise ValueError("--min-initial-release must not exceed --max-initial-release")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor, encoding_config, payload = load_actor_checkpoint(
        args.checkpoint,
        device=device,
    )
    case_factory = ThreeByThreeCaseFactory(
        seed=args.seed,
        randomize=args.randomize,
        n_robots=args.n_robots,
        min_initial_release=args.min_initial_release,
        max_initial_release=args.max_initial_release,
        initial_release_step=args.initial_release_step,
        fix_shortest_paths=args.fix_shortest_paths,
    )

    rows = []
    exact_gaps: list[float] = []
    finite_exact = 0
    exact_status_counts = {"skipped": 0, "solved": 0, "partial": 0, "unsolved": 0}

    for episode in range(1, args.episodes + 1):
        case = case_factory()
        result = evaluate_against_exact_safely(
            actor,
            case.plans,
            encoding_config=encoding_config,
            device=device,
            exact=args.exact,
            exact_deadline=args.exact_deadline,
            exact_max_nodes=args.exact_max_nodes,
        )
        row = {
            "episode": episode,
            "n_robots": args.n_robots,
            "ppo_cost": result.ppo_cost,
            "exact_cost": result.exact_cost,
            "absolute_gap": result.absolute_gap,
            "relative_gap": result.relative_gap,
            "decisions": result.decisions,
            "exact_status": result.exact_status,
        }
        rows.append(row)
        status = result.exact_status
        exact_status_counts[status] = exact_status_counts.get(status, 0) + 1
        if result.relative_gap is not None and math.isfinite(result.relative_gap):
            exact_gaps.append(result.relative_gap)
            finite_exact += 1
        exact_cost_text = (
            float("nan")
            if result.exact_cost is None or not math.isfinite(result.exact_cost)
            else result.exact_cost
        )
        rel_gap_text = (
            float("nan")
            if result.relative_gap is None or not math.isfinite(result.relative_gap)
            else result.relative_gap
        )
        print(
            f"episode={episode:03d} "
            f"n={args.n_robots} "
            f"ppo_cost={result.ppo_cost:.6f} "
            f"exact_cost={exact_cost_text:.6f} "
            f"rel_gap={rel_gap_text:.3%} "
            f"status={status} "
            f"decisions={result.decisions}"
        )

    mean_gap = statistics.mean(exact_gaps) if exact_gaps else float("nan")
    median_gap = statistics.median(exact_gaps) if exact_gaps else float("nan")

    print("--------------------------------------------------")
    print(f"checkpoint={args.checkpoint}")
    print(f"trained={payload.get('extra', {})}")
    print(f"episodes={args.episodes}")
    print(
        f"exact_solved={finite_exact}/{args.episodes} "
        f"(solved={exact_status_counts['solved']} "
        f"partial={exact_status_counts['partial']} "
        f"unsolved={exact_status_counts['unsolved']} "
        f"skipped={exact_status_counts['skipped']})"
    )
    if args.exact:
        finite_abs_gaps = [
            row["absolute_gap"]
            for row in rows
            if row["absolute_gap"] is not None and math.isfinite(row["absolute_gap"])
        ]
        mean_abs_gap = statistics.mean(finite_abs_gaps) if finite_abs_gaps else float("nan")
        print(
            f"mean_relative_gap={mean_gap:.3%}; "
            f"median_relative_gap={median_gap:.3%}; "
            f"mean_abs_gap={mean_abs_gap:.6f}"
        )
    else:
        print("exact evaluation skipped")

    if args.metrics_csv:
        path = Path(args.metrics_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "episode",
                    "n_robots",
                    "ppo_cost",
                    "exact_cost",
                    "absolute_gap",
                    "relative_gap",
                    "decisions",
                    "exact_status",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"metrics: {path}")


if __name__ == "__main__":
    main()
