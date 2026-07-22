"""Greedy PPO evaluation and optional exact DFS comparison."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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


def load_actor_checkpoint(
    path: str,
    *,
    device: torch.device | str = "cpu",
) -> tuple[BranchScoringActor, EncodingConfig, dict]:
    device = torch.device(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    actor = BranchScoringActor(
        int(payload["state_dim"]),
        int(payload["action_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
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
        return EvaluationResult(ppo_cost, None, None, None, decisions)

    exact = search_dynamic_codesign_dfs_bb(
        plans,
        branch_and_bound=True,
        verbose=False,
    )
    absolute_gap = ppo_cost - exact.best_g
    relative_gap = absolute_gap / max(abs(exact.best_g), 1e-9)
    return EvaluationResult(
        ppo_cost=ppo_cost,
        exact_cost=float(exact.best_g),
        absolute_gap=float(absolute_gap),
        relative_gap=float(relative_gap),
        decisions=decisions,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--exact", action="store_true")
    return parser.parse_args()


def main() -> None:
    from .cases import ThreeByThreeCaseFactory

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor, encoding_config, payload = load_actor_checkpoint(
        args.checkpoint,
        device=device,
    )
    case = ThreeByThreeCaseFactory(randomize=False)()
    result = evaluate_against_exact(
        actor,
        case.plans,
        encoding_config=encoding_config,
        device=device,
        run_exact=args.exact,
    )
    print(f"checkpoint={args.checkpoint}")
    print(f"trained={payload.get('extra', {})}")
    print(f"PPO cost={result.ppo_cost:.6f}; decisions={result.decisions}")
    if result.exact_cost is not None:
        print(
            f"exact cost={result.exact_cost:.6f}; "
            f"absolute gap={result.absolute_gap:.6f}; "
            f"relative gap={result.relative_gap:.3%}"
        )


if __name__ == "__main__":
    main()
