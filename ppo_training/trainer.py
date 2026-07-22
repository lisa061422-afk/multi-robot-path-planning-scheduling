"""On-policy PPO trainer for variable legal branch sets."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from statistics import mean
from typing import Callable, Sequence

import torch
from torch import nn
from torch.distributions import Categorical

from scheduler_models import RelaxedVehiclePlan

from .encoding import BranchEncoder, EncodingConfig
from .environment import DecisionTreeEnv
from .networks import BranchScoringActor, StateValueCritic
from .rollout_buffer import RolloutBuffer, RolloutStep


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


@dataclass(frozen=True)
class EpisodeStats:
    total_cost: float
    total_reward: float
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

    def collect_episode(
        self,
        plans: Sequence[RelaxedVehiclePlan],
        *,
        deterministic: bool = False,
        store: bool = True,
    ) -> EpisodeStats:
        env = DecisionTreeEnv(plans)
        encoder = BranchEncoder(plans, self.encoding_config)
        if encoder.state_dim != self.actor.state_dim:
            raise ValueError("case state dimension does not match Actor")
        if encoder.action_dim != self.actor.action_dim:
            raise ValueError("case action dimension does not match Actor")

        node, branches, terminated = env.reset()
        episode_steps: list[RolloutStep] = []
        branch_counts: list[int] = []
        total_reward = 0.0

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
            total_reward += result.reward
            branch_counts.append(len(branches))
            episode_steps.append(
                RolloutStep(
                    state=state_cpu,
                    branch_actions=actions_cpu,
                    action_index=int(action.item()),
                    old_log_prob=float(log_prob.item()),
                    reward=float(result.reward),
                    value=float(value.item()),
                )
            )
            node = result.node
            branches = result.branches
            terminated = result.terminated

        if store:
            self.buffer.extend_episode(
                episode_steps,
                discount_factor=self.config.discount_factor,
                gae_lambda=self.config.gae_lambda,
                bootstrap_value=0.0,
            )

        return EpisodeStats(
            total_cost=float(node.g),
            total_reward=total_reward,
            decisions=len(episode_steps),
            tree_edges=env.edge_count,
            mean_branches=mean(branch_counts) if branch_counts else 0.0,
            terminated=terminated,
        )

    def collect_rollouts(
        self,
        case_factory: Callable[[], object],
        episodes: int,
    ) -> list[EpisodeStats]:
        if episodes <= 0:
            raise ValueError("episodes must be positive")
        self.buffer.clear()
        stats = []
        self.actor.eval()
        self.critic.eval()
        for _ in range(episodes):
            case = case_factory()
            plans = getattr(case, "plans", case)
            stats.append(self.collect_episode(plans, store=True))
        return stats

    def update(self) -> UpdateStats:
        if not self.buffer.steps:
            raise RuntimeError("cannot update PPO with an empty rollout buffer")

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
                actor_objective = actor_loss - cfg.entropy_coef * entropy

                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_objective.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
                self.actor_optimizer.step()

                states = torch.stack(state_items)
                predictions = self.critic(states)
                critic_loss = torch.mean(
                    (predictions - value_targets[batch_indices]) ** 2
                )
                critic_objective = cfg.value_loss_coef * critic_loss
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

        return UpdateStats(
            actor_loss=mean(actor_losses),
            critic_loss=mean(critic_losses),
            entropy=mean(entropies),
            approx_kl=mean(kls),
            clip_fraction=mean(clip_fractions),
            rollout_steps=count,
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
            "encoding_config": self.encoding_config.__dict__,
            "ppo_config": self.config.__dict__,
            "extra": extra or {},
        }
        torch.save(payload, path)


def finite_mean(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return mean(finite) if finite else math.nan
