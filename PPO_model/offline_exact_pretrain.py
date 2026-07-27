"""Offline exact-data supervised pretraining for PPO and optional online PPO fine-tune.

Workflow:
1) sample random cases
2) solve each case by exact DFS
3) collect supervised transitions (state, branch set, expert action, value target)
4) train actor/critic on the dataset
5) optionally continue with PPO updates
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
import random
from pathlib import Path
from statistics import mean
from typing import Sequence

import torch
from torch import nn
from torch.distributions import Categorical

from .cases import ThreeByThreeCaseFactory
from .encoding import BranchEncoder, EncodingConfig
from .environment import DecisionTreeEnv
from .evaluate import evaluate_against_exact
from .networks import BranchScoringActor, StateValueCritic
from .trainer import PPOConfig, PPOTrainer, finite_mean
from coarse_scheduler import search_dynamic_codesign_dfs_bb


@dataclass(frozen=True)
class ExpertTransition:
    state: torch.Tensor
    branch_actions: torch.Tensor
    action_index: int
    critic_value_target: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain PPO policies from offline exact solutions."
    )
    parser.add_argument("--n-robots", type=int, default=5)
    parser.add_argument("--exact-cases", type=int, default=300)
    parser.add_argument("--skip-trivial-cases", action="store_true")
    parser.add_argument("--max-case-attempts", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261226)
    parser.add_argument("--run-root", default="output/ppo_pretrain")
    parser.add_argument("--run-id", default="n5_offline_exact")
    parser.add_argument(
        "--fix-shortest-paths",
        action="store_true",
        help="lock each vehicle to one shortest route (scheduling-only).",
    )
    parser.add_argument("--min-initial-release", type=float, default=0.0)
    parser.add_argument("--max-initial-release", type=float, default=5.0)
    parser.add_argument("--initial-release-step", type=float, default=0.5)
    parser.add_argument("--max-vehicles-per-entrance", type=int, default=0)
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--actor-hidden-dim", type=int, default=16)
    parser.add_argument("--critic-hidden-dim", type=int, default=16)
    parser.add_argument("--actor-hidden-layers", type=int, default=2)
    parser.add_argument("--critic-hidden-layers", type=int, default=2)
    parser.add_argument("--pretrain-entropy", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--run-ppo-updates", type=int, default=0)
    parser.add_argument("--run-ppo-episodes-per-update", type=int, default=16)
    parser.add_argument(
        "--reward-cost-mode",
        choices=["delta_g", "pending_delay"],
        default="delta_g",
    )
    parser.add_argument(
        "--reward-norm-mode",
        choices=["none", "absmax"],
        default="none",
    )
    parser.add_argument("--reward-norm-eps", type=float, default=1e-12)
    parser.add_argument("--ppo-learning-rate", type=float, default=3e-4)
    parser.add_argument("--ppo-update-epochs", type=int, default=4)
    parser.add_argument("--ppo-minibatch-size", type=int, default=64)
    parser.add_argument("--ppo-entropy", type=float, default=0.01)
    parser.add_argument("--ppo-max-decisions", type=int, default=200)
    parser.add_argument("--no-final-eval", action="store_true")
    return parser.parse_args()


def _canon_scalar(value: float) -> int | float | str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return round(float(value), 9)


def _node_signature(node) -> tuple:
    return (
        round(float(node.tw), 9),
        round(float(node.g), 9),
        node.U_temp,
        node.ni,
        tuple(_canon_scalar(v) for v in node.d),
        tuple(_canon_scalar(v) for v in node.r),
        tuple(_canon_scalar(v) for v in node.o),
        tuple(tuple(_canon_scalar(v) for v in row) for row in node.alpha),
        tuple(tuple(_canon_scalar(v) for v in row) for row in node.gamma),
        node.route_candidates,
    )


def _best_path(result: object) -> tuple[int, ...]:
    if not getattr(result, "nodes", None):
        return ()
    if result.best_idx < 0:
        return ()
    path: list[int] = []
    idx = result.best_idx
    while idx >= 0:
        path.append(int(idx))
        idx = int(result.nodes[idx].parent)
    return tuple(reversed(path))


def _best_match_exact_index(
    signature: tuple,
    by_signature: dict[tuple, list[int]],
    nodes: Sequence,
    node_g: float,
) -> int | None:
    candidates = by_signature.get(signature)
    if not candidates:
        return None
    if len(candidates) == 1:
        return int(candidates[0])
    return min(candidates, key=lambda idx: abs(float(nodes[idx].g) - float(node_g)))


def _build_exact_maps(nodes: Sequence, result) -> tuple[dict[tuple, list[int]], dict[int, int], float]:
    by_signature: dict[tuple, list[int]] = {}
    for index, node in enumerate(nodes):
        by_signature.setdefault(_node_signature(node), []).append(index)
    best_path = _best_path(result)
    next_map: dict[int, int] = {}
    for i in range(len(best_path) - 1):
        next_map[best_path[i]] = best_path[i + 1]
    return by_signature, next_map, float(result.best_g)


def _collect_case_transitions(
    plans,
    encoder: BranchEncoder,
    *,
    skip_trivial: bool,
) -> list[ExpertTransition]:
    transitions: list[ExpertTransition] = []
    env = DecisionTreeEnv(plans)
    try:
        result = search_dynamic_codesign_dfs_bb(
            plans,
            branch_and_bound=False,
            verbose=False,
        )
    except Exception:
        return []
    if not result.nodes or result.best_idx < 0:
        return []

    best_path = _best_path(result)
    if len(best_path) < 2:
        if skip_trivial:
            return []

    by_signature, next_map, exact_total_cost = _build_exact_maps(result.nodes, result)

    node, branches, terminated = env.reset()
    if skip_trivial and (terminated or len(branches) == 0):
        return []

    while not terminated:
        if not branches:
            break

        signature = _node_signature(node)
        exact_idx = _best_match_exact_index(
            signature,
            by_signature,
            result.nodes,
            float(node.g),
        )
        if exact_idx is None or exact_idx not in next_map:
            break
        target_idx = next_map[exact_idx]
        target_signature = _node_signature(result.nodes[target_idx])
        if target_idx >= len(result.nodes):
            break

        expert_action = None
        for branch_index, branch in enumerate(branches):
            if _node_signature(branch) == target_signature:
                expert_action = branch_index
                break
        if expert_action is None:
            for branch_index, branch in enumerate(branches):
                matched = _best_match_exact_index(
                    _node_signature(branch),
                    by_signature,
                    result.nodes,
                    float(branch.g),
                )
                if matched is not None and matched == target_idx:
                    expert_action = branch_index
                    break
        if expert_action is None:
            break

        state = encoder.encode_state(node).detach()
        branch_actions = encoder.encode_actions(node, branches).detach()
        transitions.append(
            ExpertTransition(
                state=state,
                branch_actions=branch_actions,
                action_index=int(expert_action),
                critic_value_target=-(exact_total_cost - float(node.g)),
            )
        )

        step = env.step(int(expert_action))
        node = step.node
        branches = step.branches
        terminated = step.terminated

    return transitions


def collect_offline_dataset(
    case_factory: ThreeByThreeCaseFactory,
    encoding_config: EncodingConfig,
    *,
    exact_cases: int,
    skip_trivial: bool,
    max_attempts: int,
) -> tuple[list[ExpertTransition], int, int, int]:
    transitions: list[ExpertTransition] = []
    case_with_data = 0
    attempts = 0
    total_steps = 0

    while case_with_data < exact_cases and attempts < max_attempts:
        attempts += 1
        case = case_factory()
        plans = case.plans
        encoder = BranchEncoder(plans, encoding_config)
        case_transitions = _collect_case_transitions(
            plans,
            encoder,
            skip_trivial=skip_trivial,
        )
        if case_transitions:
            transitions.extend(case_transitions)
            case_with_data += 1
            total_steps += len(case_transitions)

    return transitions, attempts, case_with_data, total_steps


def _run_supervised_pretrain(
    transitions: list[ExpertTransition],
    actor: BranchScoringActor,
    critic: StateValueCritic,
    *,
    device: torch.device,
    epochs: int,
    actor_lr: float,
    critic_lr: float,
    grad_clip: float,
    entropy_coef: float,
    seed: int,
) -> tuple[float, float, float, torch.optim.Adam, torch.optim.Adam]:
    if not transitions:
        raise ValueError("no supervised transitions collected")

    rng = random.Random(seed)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    ce = nn.CrossEntropyLoss()
    actor_losses_by_epoch: list[float] = []
    critic_losses_by_epoch: list[float] = []
    entropies_by_epoch: list[float] = []

    for epoch in range(1, epochs + 1):
        order = list(range(len(transitions)))
        rng.shuffle(order)

        step_actor_losses: list[float] = []
        step_critic_losses: list[float] = []
        step_entropies: list[float] = []

        for index in order:
            sample = transitions[index]
            if sample.branch_actions.numel() == 0:
                continue

            state = sample.state.to(device)
            branch_actions = sample.branch_actions.to(device)
            logits = actor(state, branch_actions)
            if logits.numel() == 0:
                continue

            action_target = torch.tensor([sample.action_index], device=device, dtype=torch.long)
            actor_loss = ce(logits.unsqueeze(0), action_target)
            entropy = Categorical(logits=logits).entropy().mean()
            actor_objective = actor_loss - entropy_coef * entropy

            actor_optimizer.zero_grad(set_to_none=True)
            actor_objective.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)
            actor_optimizer.step()

            value_pred = critic(state).squeeze(-1)
            value_target = torch.tensor(sample.critic_value_target, device=device)
            critic_loss = nn.functional.mse_loss(value_pred, value_target)

            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
            critic_optimizer.step()

            step_actor_losses.append(float(actor_loss.item()))
            step_critic_losses.append(float(critic_loss.item()))
            step_entropies.append(float(entropy.item()))

        actor_loss = mean(step_actor_losses) if step_actor_losses else float("nan")
        critic_loss = mean(step_critic_losses) if step_critic_losses else float("nan")
        entropy = mean(step_entropies) if step_entropies else float("nan")
        actor_losses_by_epoch.append(actor_loss)
        critic_losses_by_epoch.append(critic_loss)
        entropies_by_epoch.append(entropy)

        print(
            f"[pretrain epoch {epoch:03d}] "
            f"actor_loss={actor_loss:.3f} critic_loss={critic_loss:.3f} "
            f"entropy={entropy:.3f}"
        )

    return (
        actor_losses_by_epoch[-1],
        critic_losses_by_epoch[-1],
        entropies_by_epoch[-1],
        actor_optimizer,
        critic_optimizer,
    )


def _encoder_for_case(
    plans,
    n_robots: int,
) -> tuple[BranchEncoder, EncodingConfig]:
    encoding_config = EncodingConfig(n_robots=n_robots)
    return BranchEncoder(plans, encoding_config), encoding_config


def _save_checkpoint(
    path: Path,
    actor: BranchScoringActor,
    critic: StateValueCritic,
    actor_optimizer: torch.optim.Adam,
    critic_optimizer: torch.optim.Adam,
    encoding_config: EncodingConfig,
    ppo_config: PPOConfig,
) -> None:
    payload = {
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": critic_optimizer.state_dict(),
        "state_dim": actor.state_dim,
        "action_dim": actor.action_dim,
        "hidden_dim": actor.hidden_dim,
        "actor_hidden_layers": actor.hidden_layers,
        "actor_type": actor.__class__.__name__,
        "actor_gnn_message_layers": getattr(actor, "message_layers", 0),
        "actor_n_robots": getattr(actor, "n_robots", None),
        "critic_hidden_dim": critic.hidden_dim,
        "critic_hidden_layers": critic.hidden_layers,
        "encoding_config": encoding_config.__dict__,
        "ppo_config": ppo_config.__dict__,
        "extra": {
            "offline_pretrain_version": "1.0",
        },
    }
    torch.save(payload, str(path))


def main() -> None:
    args = parse_args()
    if args.n_robots <= 0:
        raise ValueError("--n-robots must be positive")
    if args.exact_cases <= 0:
        raise ValueError("--exact-cases must be positive")
    if args.pretrain_epochs <= 0:
        raise ValueError("--pretrain-epochs must be positive")
    if args.actor_hidden_dim <= 0 or args.critic_hidden_dim <= 0:
        raise ValueError("--actor-hidden-dim and --critic-hidden-dim must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    run_root = Path(args.run_root)
    run_dir = run_root / args.run_id / f"n{args.n_robots}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "ppo_branch_actor.pt"
    metrics_pretrain_path = run_dir / "pretrain_metrics.csv"
    metrics_dataset_path = run_dir / "pretrain_dataset_cases.csv"
    metrics_training_path = run_dir / "training_metrics.csv"
    contention_path = run_dir / "training_metrics_case_contentions.csv"

    prototype_factory = ThreeByThreeCaseFactory(
        seed=args.seed,
        randomize=False,
        n_robots=args.n_robots,
        max_vehicles_per_entrance=args.max_vehicles_per_entrance,
        max_initial_release=args.max_initial_release,
        min_initial_release=args.min_initial_release,
        initial_release_step=args.initial_release_step,
        fix_shortest_paths=args.fix_shortest_paths,
    )
    prototype_plans = prototype_factory().plans
    encoder, encoding_config = _encoder_for_case(prototype_plans, args.n_robots)

    actor = BranchScoringActor(
        state_dim=encoder.state_dim,
        action_dim=encoder.action_dim,
        hidden_dim=args.actor_hidden_dim,
        hidden_layers=args.actor_hidden_layers,
    )
    critic = StateValueCritic(
        state_dim=encoder.state_dim,
        hidden_dim=args.critic_hidden_dim,
        hidden_layers=args.critic_hidden_layers,
    )
    actor.to(device)
    critic.to(device)

    case_factory = ThreeByThreeCaseFactory(
        seed=args.seed,
        randomize=True,
        n_robots=args.n_robots,
        max_vehicles_per_entrance=args.max_vehicles_per_entrance,
        max_initial_release=args.max_initial_release,
        min_initial_release=args.min_initial_release,
        initial_release_step=args.initial_release_step,
        fix_shortest_paths=args.fix_shortest_paths,
    )

    transitions, attempts, cases_with_data, total_steps = collect_offline_dataset(
        case_factory,
        encoding_config,
        exact_cases=args.exact_cases,
        skip_trivial=args.skip_trivial_cases,
        max_attempts=args.max_case_attempts,
    )

    if not transitions:
        raise RuntimeError(
            "no supervised transitions collected; increase attempts or reduce filtering."
        )

    actor_loss, critic_loss, entropy, actor_optimizer, critic_optimizer = (
        _run_supervised_pretrain(
            transitions,
            actor,
            critic,
            device=device,
            epochs=args.pretrain_epochs,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            grad_clip=args.grad_clip,
            entropy_coef=args.pretrain_entropy,
            seed=args.seed + 1,
        )
    )

    with metrics_pretrain_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "transitions",
                "actor_loss",
                "critic_loss",
                "entropy",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "epoch": args.pretrain_epochs,
                "transitions": len(transitions),
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "entropy": entropy,
            }
        )

    with metrics_dataset_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["attempts", "requested_cases", "collected_cases", "total_transitions"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "attempts": attempts,
                "requested_cases": args.exact_cases,
                "collected_cases": cases_with_data,
                "total_transitions": total_steps,
            }
        )

    ppo_config = PPOConfig(
        discount_factor=1.0,
        gae_lambda=0.95,
        entropy_coef=args.ppo_entropy,
        value_loss_coef=0.5,
        learning_rate=args.ppo_learning_rate,
        update_epochs=args.ppo_update_epochs,
        minibatch_size=args.ppo_minibatch_size,
        max_grad_norm=args.grad_clip,
        max_decisions_per_episode=args.ppo_max_decisions,
        reward_cost_mode=args.reward_cost_mode,
        reward_norm_mode=args.reward_norm_mode,
        reward_norm_minmax_eps=args.reward_norm_eps,
        critic_supervise_weight=0.0,
    )

    _save_checkpoint(
        checkpoint_path,
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        encoding_config,
        ppo_config,
    )

    print(f"pretrain checkpoint: {checkpoint_path}")
    print(f"pretrain metrics: {metrics_pretrain_path}")
    print(f"dataset stats: {metrics_dataset_path}")

    if args.no_final_eval:
        return

    reference_case_factory = ThreeByThreeCaseFactory(
        seed=args.seed + 2,
        randomize=False,
        n_robots=args.n_robots,
        max_vehicles_per_entrance=args.max_vehicles_per_entrance,
        max_initial_release=args.max_initial_release,
        min_initial_release=args.min_initial_release,
        initial_release_step=args.initial_release_step,
        fix_shortest_paths=args.fix_shortest_paths,
    )
    reference_case = reference_case_factory()
    initial_eval = evaluate_against_exact(
        actor,
        reference_case.plans,
        encoding_config=encoding_config,
        device=device,
        run_exact=True,
    )
    print(
        f"pretrain final fixed-case PPO cost={initial_eval.ppo_cost:.6f}, "
        f"exact_cost={initial_eval.exact_cost:.6f}, gap={initial_eval.absolute_gap:.6f}"
    )

    if args.run_ppo_updates <= 0:
        return

    trainer = PPOTrainer(
        actor,
        critic,
        encoding_config=encoding_config,
        config=ppo_config,
        device=device,
        seed=args.seed + 10,
    )
    ppo_case_factory = ThreeByThreeCaseFactory(
        seed=args.seed + 3,
        randomize=True,
        n_robots=args.n_robots,
        max_vehicles_per_entrance=args.max_vehicles_per_entrance,
        max_initial_release=args.max_initial_release,
        min_initial_release=args.min_initial_release,
        initial_release_step=args.initial_release_step,
        fix_shortest_paths=args.fix_shortest_paths,
    )
    with contention_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "n_robots",
                "update",
                "episode_in_update",
                "case_name",
                "contention_records",
                "contention_pairs",
                "contention_record_count",
            ]
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
        "mean_contention_records",
        "mean_contention_pairs",
        "contention_case_rate",
        "actor_loss",
        "critic_loss",
        "critic_supervise_loss",
        "entropy",
        "scheduled_entropy",
        "learning_rate",
        "approx_kl",
        "clip_fraction",
        "fixed_greedy_cost",
        "elapsed_seconds",
    ]
    rows: list[dict[str, float]] = []

    for update_index in range(1, args.run_ppo_updates + 1):
        rollout_stats = trainer.collect_rollouts(
            ppo_case_factory,
            args.run_ppo_episodes_per_update,
            skip_trivial=args.skip_trivial_cases,
            max_attempts=args.max_case_attempts,
        )
        with contention_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            for local_ep, stat in enumerate(rollout_stats, start=1):
                writer.writerow(
                    [
                        args.n_robots,
                        update_index,
                        local_ep,
                        stat.case_name,
                        ";".join(map(str, stat.contention_records)),
                        stat.contention_pair_count,
                        len(stat.contention_records),
                    ]
                )

        update_stats = trainer.update_with_entropy(args.ppo_entropy)
        fixed_eval = evaluate_against_exact(
            actor,
            reference_case.plans,
            encoding_config=encoding_config,
            device=device,
            run_exact=False,
        )
        row = {
            "n_robots": args.n_robots,
            "update": update_index,
            "episodes": update_index * args.run_ppo_episodes_per_update,
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
            "critic_supervise_loss": update_stats.critic_supervise_loss,
            "entropy": update_stats.entropy,
            "scheduled_entropy": args.ppo_entropy,
            "learning_rate": args.ppo_learning_rate,
            "approx_kl": update_stats.approx_kl,
            "clip_fraction": update_stats.clip_fraction,
            "fixed_greedy_cost": fixed_eval.ppo_cost,
            "elapsed_seconds": 0.0,
        }
        rows.append(row)
        print(
            f"[N={args.n_robots}] [PPO {update_index:04d}] "
            f"episodes={row['episodes']} steps={row['rollout_steps']} "
            f"mean_J={row['mean_cost']:.3f} fixed_J={row['fixed_greedy_cost']:.3f} "
            f"entropy={row['entropy']:.3f}"
        )
        trainer.save_checkpoint(
            str(checkpoint_path),
            extra={
                "completed_updates": update_index,
                "completed_episodes": row["episodes"],
                "fixed_greedy_cost": row["fixed_greedy_cost"],
            },
        )
        with metrics_training_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    final_eval = evaluate_against_exact(
        actor,
        reference_case.plans,
        encoding_config=encoding_config,
        device=device,
        run_exact=True,
    )
    print(f"final fixed-case PPO cost={final_eval.ppo_cost:.6f}")
    if final_eval.exact_cost is not None:
        print(
            f"final fixed-case exact cost={final_eval.exact_cost:.6f}, "
            f"gap={final_eval.absolute_gap:.6f}"
        )
    print(f"checkpoint: {checkpoint_path}")
    print(f"training metrics: {metrics_training_path}")


if __name__ == "__main__":
    main()
