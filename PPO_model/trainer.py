"""On-policy PPO trainer for variable legal branch sets."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import itertools
from statistics import mean
from typing import Callable, Sequence

import torch
from torch import nn
from torch.distributions import Categorical

from scheduler_models import RelaxedNode, RelaxedVehiclePlan

from .encoding import BranchEncoder, EncodingConfig
from .environment import DecisionTreeEnv
from .networks import BranchScoringActor, StateValueCritic
from .rollout_buffer import RolloutBuffer, RolloutStep
from trajectory_conflicts import route_ids_conflict
from coarse_scheduler import search_dynamic_codesign_dfs_bb


@dataclass(frozen=True)
class PPOConfig:
    discount_factor: float = 1.0
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    learning_rate: float = 3e-4
    update_epochs: int = 4
    minibatch_size: int = 64
    max_grad_norm: float = 0.5
    max_decisions_per_episode: int = 200
    reward_cost_mode: str = "delta_g"
    reward_norm_mode: str = "none"
    reward_norm_minmax_eps: float = 1e-12
    critic_supervise_weight: float = 0.0


@dataclass(frozen=True)
class EpisodeStats:
    total_cost: float
    total_reward: float
    total_reward_normalized: float
    case_name: str
    contention_records: tuple[int, ...]
    contention_pair_count: int
    decisions: int
    tree_edges: int
    mean_branches: float
    terminated: bool


@dataclass(frozen=True)
class UpdateStats:
    actor_loss: float
    critic_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    rollout_steps: int
    critic_supervise_loss: float


class PPOTrainer:
    def __init__(
        self,
        actor: BranchScoringActor,
        critic: StateValueCritic,
        *,
        encoding_config: EncodingConfig = EncodingConfig(),
        config: PPOConfig = PPOConfig(),
        device: torch.device | str = "cpu",
        seed: int = 20260721,
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.encoding_config = encoding_config
        self.config = config
        self.device = torch.device(device)
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.learning_rate
        )
        self.buffer = RolloutBuffer()
        self.rng = random.Random(seed)
        torch.manual_seed(seed)

    @staticmethod
    def _canon_scalar(value: float) -> int | float | str:
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return round(float(value), 9)

    @staticmethod
    def _node_signature(node: RelaxedNode) -> tuple:
        return (
            round(float(node.tw), 9),
            round(float(node.g), 9),
            node.U_temp,
            node.ni,
            tuple(PPOTrainer._canon_scalar(v) for v in node.d),
            tuple(PPOTrainer._canon_scalar(v) for v in node.r),
            tuple(PPOTrainer._canon_scalar(v) for v in node.o),
            tuple(tuple(PPOTrainer._canon_scalar(v) for v in row) for row in node.alpha),
            tuple(tuple(PPOTrainer._canon_scalar(v) for v in row) for row in node.gamma),
            node.route_candidates,
        )

    @staticmethod
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

    def _exact_teacher(
        self,
        plans: Sequence[RelaxedVehiclePlan],
    ) -> dict[str, object] | None:
        try:
            result = search_dynamic_codesign_dfs_bb(
                plans,
                branch_and_bound=False,
                verbose=False,
            )
        except Exception:
            return None

        if not result.nodes or result.best_idx < 0:
            return None
        path = self._best_path(result)
        if len(path) < 2:
            return None

        by_signature: dict[tuple, list[int]] = {}
        for index, node in enumerate(result.nodes):
            by_signature.setdefault(self._node_signature(node), []).append(index)

        return {
            "nodes": result.nodes,
            "by_signature": by_signature,
            "exact_total_cost": float(result.best_g),
        }

    @staticmethod
    def _best_match_exact_index(
        signature: tuple,
        by_signature: dict[tuple, list[int]],
        nodes: Sequence[RelaxedNode],
        node_g: float,
    ) -> int | None:
        candidates = by_signature.get(signature)
        if not candidates:
            return None
        if len(candidates) == 1:
            return int(candidates[0])
        return min(
            candidates,
            key=lambda idx: abs(float(nodes[idx].g) - float(node_g)),
        )

    @staticmethod
    def _pad_plans(
        plans: Sequence[RelaxedVehiclePlan],
        target_n: int,
    ) -> tuple[RelaxedVehiclePlan, ...]:
        if target_n < 0:
            raise ValueError("target_n must be non-negative")
        if len(plans) > target_n:
            raise ValueError(
                f"case has {len(plans)} plans, which exceeds target_n={target_n}"
            )
        if len(plans) == target_n:
            return tuple(plans)
        if not plans:
            raise ValueError("case plans must be non-empty before padding")
        if target_n == 0:
            return ()
        base_road_time = float(plans[0].road_time)
        max_vehicle_id = max(plan.vehicle_id for plan in plans)
        padded = list(plans)
        for _idx in range(target_n - len(plans)):
            padded.append(
                RelaxedVehiclePlan(
                    vehicle_id=max_vehicle_id + 1 + _idx,
                    entrance=plans[0].entrance,
                    exit=plans[0].exit,
                    route_options=(),
                    alpha0=0.0,
                    road_time=base_road_time,
                )
            )
            max_vehicle_id += 1
        return tuple(padded)


    @staticmethod
    def _case_contention_profile(
        plans: Sequence[RelaxedVehiclePlan],
    ) -> tuple[tuple[int, ...], int]:
        """Count contention intersections and returning conflicting resources.

        This metric uses each vehicle's current fixed plan route (the first option)
        and is intended as a cheap case-level supervision signal while training
        with fixed or low-branching roadmaps.
        """

        by_resource: dict[int, list[tuple[int, int]]] = {}
        for n, plan in enumerate(plans):
            if not plan.route_options:
                continue
            route = plan.route_options[0]
            for ti, resource in enumerate(route.intersections):
                by_resource.setdefault(int(resource), []).append((n, int(route.traversals[ti].route_id)))

        contention_records: list[int] = []
        contention_pair_count = 0
        for resource, assignments in by_resource.items():
            if len(assignments) < 2:
                continue
            has_conflict = False
            for (_n_i, route_id_i), (_n_j, route_id_j) in itertools.combinations(assignments, 2):
                if route_ids_conflict(route_id_i, route_id_j):
                    contention_pair_count += 1
                    has_conflict = True
            if has_conflict:
                contention_records.append(resource)

        return tuple(sorted(contention_records)), contention_pair_count

    def collect_episode(
        self,
        plans: Sequence[RelaxedVehiclePlan],
        *,
        deterministic: bool = False,
        store: bool = True,
        skip_trivial: bool = False,
        case_name: str = "unknown",
    ) -> EpisodeStats:
        plans = self._pad_plans(plans, self.encoding_config.n_robots)

        if skip_trivial and self._is_single_path_case(
            plans,
            max_decisions=self.config.max_decisions_per_episode,
        ):
            return EpisodeStats(
                total_cost=0.0,
                total_reward=0.0,
                total_reward_normalized=0.0,
                case_name=case_name,
                contention_records=(),
                contention_pair_count=0,
                decisions=0,
                tree_edges=0,
                mean_branches=0.0,
                terminated=True,
            )

        env = DecisionTreeEnv(
            plans,
            reward_cost_mode=self.config.reward_cost_mode,
        )
        encoder = BranchEncoder(plans, self.encoding_config)
        if encoder.state_dim != self.actor.state_dim:
            raise ValueError("case state dimension does not match Actor")
        if encoder.action_dim != self.actor.action_dim:
            raise ValueError("case action dimension does not match Actor")

        case_contentions, case_contention_count = self._case_contention_profile(plans)
        node, branches, terminated = env.reset()
        if skip_trivial and (terminated or len(branches) == 0):
            return EpisodeStats(
                total_cost=float(node.g),
                total_reward=0.0,
                total_reward_normalized=0.0,
                case_name=case_name,
                contention_records=case_contentions,
                contention_pair_count=case_contention_count,
                decisions=0,
                tree_edges=env.edge_count,
                mean_branches=0.0,
                terminated=terminated,
            )
        episode_steps: list[RolloutStep] = []
        branch_counts: list[int] = []
        rewards: list[float] = []
        total_reward = 0.0
        exact_teacher = (
            self._exact_teacher(plans)
            if self.config.critic_supervise_weight > 0.0
            else None
        )

        while not terminated:
            if len(episode_steps) >= self.config.max_decisions_per_episode:
                raise RuntimeError(
                    "episode exceeded max_decisions_per_episode; increase the "
                    "limit or inspect the transition model"
                )

            state_cpu = encoder.encode_state(node)
            actions_cpu = encoder.encode_actions(node, branches)
            state = state_cpu.to(self.device)
            branch_actions = actions_cpu.to(self.device)
            expert_value_target = float("nan")
            if exact_teacher is not None:
                teacher_nodes = exact_teacher["nodes"]  # type: ignore[assignment]
                by_signature = exact_teacher["by_signature"]  # type: ignore[assignment]
                exact_total_cost = exact_teacher["exact_total_cost"]  # type: ignore[assignment]
                signature = self._node_signature(node)
                exact_idx = self._best_match_exact_index(
                    signature,
                    by_signature,  # type: ignore[arg-type]
                    teacher_nodes,  # type: ignore[arg-type]
                    float(node.g),
                )
                if exact_idx is not None:
                    expert_value_target = -(
                        float(exact_total_cost) - float(teacher_nodes[exact_idx].g)
                    )

            with torch.no_grad():
                logits = self.actor(state, branch_actions)
                distribution = Categorical(logits=logits)
                if deterministic:
                    action = torch.argmax(logits)
                else:
                    action = distribution.sample()
                log_prob = distribution.log_prob(action)
                value = self.critic(state)

            result = env.step(int(action.item()))
            rewards.append(float(result.reward))
            total_reward += float(result.reward)
            branch_counts.append(len(branches))
            episode_steps.append(
                RolloutStep(
                    state=state_cpu,
                    branch_actions=actions_cpu,
                    action_index=int(action.item()),
                    old_log_prob=float(log_prob.item()),
                    reward=float(result.reward),
                    value=float(value.item()),
                    expert_value_target=expert_value_target,
                )
            )
            node = result.node
            branches = result.branches
            terminated = result.terminated

        normalized_rewards = self._normalize_rewards(rewards)
        if store:
            for step, normalized_reward in zip(episode_steps, normalized_rewards):
                step.reward = normalized_reward
            self.buffer.extend_episode(
                episode_steps,
                discount_factor=self.config.discount_factor,
                gae_lambda=self.config.gae_lambda,
                bootstrap_value=0.0,
            )

        total_reward_normalized = sum(normalized_rewards)

        return EpisodeStats(
            total_cost=float(node.g),
            total_reward=total_reward,
            total_reward_normalized=total_reward_normalized,
            case_name=case_name,
            contention_records=case_contentions,
            contention_pair_count=case_contention_count,
            decisions=len(episode_steps),
            tree_edges=env.edge_count,
            mean_branches=mean(branch_counts) if branch_counts else 0.0,
            terminated=terminated,
        )

    def collect_rollouts(
        self,
        case_factory: Callable[[], object],
        episodes: int,
        *,
        skip_trivial: bool = False,
        max_attempts: int = 2000,
    ) -> list[EpisodeStats]:
        if episodes <= 0:
            raise ValueError("episodes must be positive")
        self.buffer.clear()
        stats = []
        self.actor.eval()
        self.critic.eval()
        attempts = 0
        max_attempts = max(int(max_attempts), episodes)
        while len(stats) < episodes:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    "too many attempts when collecting non-trivial episodes; "
                    "adjust --skip-trivial-cases or reduce filter strictness"
                )
            case = case_factory()
            plans = getattr(case, "plans", case)
            episode_stats = self.collect_episode(
                plans,
                store=True,
                skip_trivial=skip_trivial,
                case_name=getattr(case, "name", "unknown"),
            )
            if skip_trivial and episode_stats.decisions == 0:
                continue
            stats.append(episode_stats)
        return stats

    @staticmethod
    def _is_single_path_case(
        plans: Sequence[RelaxedVehiclePlan],
        max_decisions: int,
    ) -> bool:
        """Return True when an episode has no branching decisions at any step.

        This walks the deterministic chain induced by unique-branch nodes and
        reports True only if every visited decision node exposes one or fewer
        actions (including already-terminated roots).
        """
        env = DecisionTreeEnv(plans)
        node, branches, terminated = env.reset()
        if terminated:
            return True
        if not branches:
            return True
        step_limit = max(1, int(max_decisions))
        steps = 0
        while not terminated:
            if len(branches) > 1:
                return False
            if not branches:
                return True
            if steps >= step_limit * 3:
                raise RuntimeError(
                    "single-path check exceeded safety limit; "
                    "please increase --max-decisions or inspect transition logic"
                )
            result = env.step(0)
            branches = result.branches
            terminated = result.terminated
            steps += 1
        return True

    def update(self) -> UpdateStats:
        if not self.buffer.steps:
            raise RuntimeError("cannot update PPO with an empty rollout buffer")

        return self.update_with_entropy(self.config.entropy_coef)

    def set_learning_rates(self, learning_rate: float) -> None:
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        for group in self.actor_optimizer.param_groups:
            group["lr"] = learning_rate
        for group in self.critic_optimizer.param_groups:
            group["lr"] = learning_rate

    def update_with_entropy(self, entropy_coef: float) -> UpdateStats:
        if not self.buffer.steps:
            raise RuntimeError("cannot update PPO with an empty rollout buffer")

        if not math.isfinite(entropy_coef):
            raise ValueError("entropy_coef must be finite")

        self.actor.train()
        self.critic.train()
        cfg = self.config
        advantages = self.buffer.normalized_advantages(self.device)
        value_targets = torch.tensor(
            [step.value_target for step in self.buffer.steps],
            dtype=torch.float32,
            device=self.device,
        )
        old_log_probs = torch.tensor(
            [step.old_log_prob for step in self.buffer.steps],
            dtype=torch.float32,
            device=self.device,
        )

        actor_losses: list[float] = []
        critic_losses: list[float] = []
        entropies: list[float] = []
        kls: list[float] = []
        clip_fractions: list[float] = []
        critic_supervise_losses: list[float] = []
        count = len(self.buffer.steps)

        for _epoch in range(cfg.update_epochs):
            permutation = torch.randperm(count).tolist()
            for offset in range(0, count, cfg.minibatch_size):
                indices = permutation[offset : offset + cfg.minibatch_size]
                new_log_prob_items = []
                entropy_items = []
                state_items = []

                for index in indices:
                    step = self.buffer.steps[index]
                    state = step.state.to(self.device)
                    actions = step.branch_actions.to(self.device)
                    distribution = Categorical(logits=self.actor(state, actions))
                    action = torch.tensor(step.action_index, device=self.device)
                    new_log_prob_items.append(distribution.log_prob(action))
                    entropy_items.append(distribution.entropy())
                    state_items.append(state)

                new_log_probs = torch.stack(new_log_prob_items)
                entropy = torch.stack(entropy_items).mean()
                batch_indices = torch.tensor(indices, device=self.device)
                log_ratio = new_log_probs - old_log_probs[batch_indices]
                ratio = log_ratio.exp()
                batch_advantages = advantages[batch_indices]
                unclipped = ratio * batch_advantages
                clipped = torch.clamp(
                    ratio,
                    1.0 - cfg.clip_epsilon,
                    1.0 + cfg.clip_epsilon,
                ) * batch_advantages
                actor_loss = -torch.minimum(unclipped, clipped).mean()
                actor_objective = actor_loss - entropy_coef * entropy

                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_objective.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
                self.actor_optimizer.step()

                states = torch.stack(state_items)
                predictions = self.critic(states)
                critic_loss = torch.mean(
                    (predictions - value_targets[batch_indices]) ** 2
                )
                expert_value_targets = torch.tensor(
                    [
                        self.buffer.steps[idx].expert_value_target
                        for idx in indices
                    ],
                    device=self.device,
                    dtype=torch.float32,
                )
                expert_mask = torch.isfinite(expert_value_targets)
                if expert_mask.any() and cfg.critic_supervise_weight > 0.0:
                    critic_supervise_loss = torch.mean(
                        (predictions[expert_mask] - expert_value_targets[expert_mask]) ** 2
                    )
                else:
                    critic_supervise_loss = torch.tensor(0.0, device=self.device)
                critic_objective = (
                    cfg.value_loss_coef * critic_loss
                    + cfg.critic_supervise_weight * critic_supervise_loss
                )
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_objective.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.critic_optimizer.step()

                with torch.no_grad():
                    approx_kl = torch.mean((ratio - 1.0) - log_ratio)
                    clip_fraction = torch.mean(
                        (torch.abs(ratio - 1.0) > cfg.clip_epsilon).float()
                    )
                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(critic_loss.item()))
                entropies.append(float(entropy.item()))
                kls.append(float(approx_kl.item()))
                clip_fractions.append(float(clip_fraction.item()))
                critic_supervise_losses.append(float(critic_supervise_loss.item()))

        return UpdateStats(
            actor_loss=mean(actor_losses),
            critic_loss=mean(critic_losses),
            entropy=mean(entropies),
            approx_kl=mean(kls),
            clip_fraction=mean(clip_fractions),
            rollout_steps=count,
            critic_supervise_loss=mean(critic_supervise_losses),
        )

    def _normalize_rewards(self, rewards: list[float]) -> list[float]:
        if not rewards or self.config.reward_norm_mode == "none":
            return rewards

        mode = self.config.reward_norm_mode.lower()
        if mode in {"absmax", "abs_max", "maxabs"}:
            denom = max(abs(value) for value in rewards)
            if denom <= self.config.reward_norm_minmax_eps:
                return [0.0 for _ in rewards]
            return [value / denom for value in rewards]

        if mode == "none":
            return rewards
        raise ValueError(
            f"invalid reward_norm_mode='{self.config.reward_norm_mode}', "
            "expected one of: none, absmax"
        )

    def save_checkpoint(self, path: str, *, extra: dict | None = None) -> None:
        payload = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "state_dim": self.actor.state_dim,
            "action_dim": self.actor.action_dim,
            "hidden_dim": self.actor.hidden_dim,
            "actor_hidden_layers": self.actor.hidden_layers,
            "actor_type": self.actor.__class__.__name__,
            "actor_gnn_message_layers": getattr(self.actor, "message_layers", 0),
            "actor_n_robots": getattr(self.actor, "n_robots", None),
            "critic_hidden_dim": self.critic.hidden_dim,
            "critic_hidden_layers": self.critic.hidden_layers,
            "encoding_config": self.encoding_config.__dict__,
            "ppo_config": self.config.__dict__,
            "extra": extra or {},
        }
        torch.save(payload, path)


def finite_mean(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return mean(finite) if finite else math.nan
