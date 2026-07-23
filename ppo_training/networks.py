"""Neural networks for variable-branch PPO."""

from __future__ import annotations

import torch
from torch import nn


def _orthogonal_init(module: nn.Module, gain: float = 1.0) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)


class BranchScoringActor(nn.Module):
    """Score each currently legal branch with one shared MLP."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = max(1, int(hidden_layers))
        layers: list[nn.Module] = []
        input_dim = self.state_dim + self.action_dim
        for layer_index in range(self.hidden_layers):
            input_size = input_dim if layer_index == 0 else self.hidden_dim
            layers.append(nn.Linear(input_size, self.hidden_dim))
            _orthogonal_init(layers[-1], gain=2.0**0.5)
            layers.append(nn.LayerNorm(self.hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(self.hidden_dim, 1))
        self.network = nn.Sequential(
            *layers,
        )
        _orthogonal_init(self.network[-1], gain=0.01)

    def forward(
        self,
        state: torch.Tensor,
        branch_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Return one logit per legal branch.

        ``state`` may be shape ``[state_dim]`` or ``[1, state_dim]``.
        ``branch_actions`` must be ``[n_branches, action_dim]``.
        """

        if state.ndim == 2:
            if state.shape[0] != 1:
                raise ValueError("BranchScoringActor expects one state at a time")
            state = state.squeeze(0)
        if state.ndim != 1 or state.numel() != self.state_dim:
            raise ValueError(f"bad state shape {tuple(state.shape)}")
        if branch_actions.ndim != 2 or branch_actions.shape[1] != self.action_dim:
            raise ValueError(f"bad branch action shape {tuple(branch_actions.shape)}")
        if branch_actions.shape[0] == 0:
            return torch.empty(0, device=state.device, dtype=state.dtype)

        repeated_state = state.unsqueeze(0).expand(branch_actions.shape[0], -1)
        inputs = torch.cat((repeated_state, branch_actions), dim=-1)
        return self.network(inputs).squeeze(-1)


class StateValueCritic(nn.Module):
    """Estimate negative expected remaining cost from the current node state."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = max(1, int(hidden_layers))
        layers: list[nn.Module] = []
        input_size = self.state_dim
        for _ in range(self.hidden_layers):
            layers.append(nn.Linear(input_size, self.hidden_dim))
            _orthogonal_init(layers[-1], gain=2.0**0.5)
            layers.append(nn.LayerNorm(self.hidden_dim))
            layers.append(nn.SiLU())
            input_size = self.hidden_dim
        layers.append(nn.Linear(self.hidden_dim, 1))
        self.network = nn.Sequential(*layers)
        _orthogonal_init(self.network[-1], gain=1.0)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states).squeeze(-1)
