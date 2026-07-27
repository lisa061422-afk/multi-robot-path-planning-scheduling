"""Trajectory storage and GAE for variable-branch PPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class RolloutStep:
    state: torch.Tensor
    branch_actions: torch.Tensor
    action_index: int
    old_log_prob: float
    reward: float
    value: float
    expert_value_target: float = float("nan")
    advantage: float = 0.0
    value_target: float = 0.0


class RolloutBuffer:
    def __init__(self) -> None:
        self.steps: list[RolloutStep] = []

    def __len__(self) -> int:
        return len(self.steps)

    def clear(self) -> None:
        self.steps.clear()

    def extend_episode(
        self,
        episode_steps: Iterable[RolloutStep],
        *,
        discount_factor: float,
        gae_lambda: float,
        bootstrap_value: float = 0.0,
    ) -> None:
        steps = list(episode_steps)
        if not steps:
            return

        gae = 0.0
        next_value = float(bootstrap_value)
        for step in reversed(steps):
            delta = step.reward + discount_factor * next_value - step.value
            gae = delta + discount_factor * gae_lambda * gae
            step.advantage = gae
            step.value_target = gae + step.value
            next_value = step.value
        self.steps.extend(steps)

    def normalized_advantages(self, device: torch.device) -> torch.Tensor:
        values = torch.tensor(
            [step.advantage for step in self.steps],
            dtype=torch.float32,
            device=device,
        )
        if values.numel() > 1:
            values = (values - values.mean()) / (values.std(unbiased=False) + 1e-8)
        return values
