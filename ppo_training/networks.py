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


class BranchScoringActorGNN(nn.Module):
    """Lightweight graph-style branch scorer.

    The model builds a small message-passing network over candidate branches.
    Each branch is a node in a fully/partly connected graph built from shared
    resource-level structure (requested/enter/exit intersection IDs).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_robots: int,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
        message_layers: int = 2,
    ) -> None:
        super().__init__()
        if n_robots <= 0:
            raise ValueError("n_robots must be positive")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = max(1, int(hidden_layers))
        self.message_layers = max(1, int(message_layers))
        self.n_robots = int(n_robots)

        if self.action_dim % self.n_robots != 0:
            raise ValueError(
                "action_dim must be divisible by n_robots for graph-actor "
                "to recover per-robot action blocks"
            )
        self.per_robot_action_dim = self.action_dim // self.n_robots
        if self.per_robot_action_dim < 7:
            raise ValueError("per-robot action dimension is unexpectedly small")

        if (self.per_robot_action_dim - 4) % 3 != 0:
            raise ValueError("per-robot action format is not [R+1, R+1, R+1, 4]")
        self.resource_dim = (self.per_robot_action_dim - 4) // 3
        if self.resource_dim <= 0:
            raise ValueError("resource one-hot dim must be positive")

        self.state_embed = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
        )
        _orthogonal_init(self.state_embed[0], gain=2.0**0.5)

        layers: list[nn.Module] = []
        input_dim = self.action_dim
        for layer_index in range(self.hidden_layers):
            input_size = input_dim if layer_index == 0 else self.hidden_dim
            layers.append(nn.Linear(input_size, self.hidden_dim))
            _orthogonal_init(layers[-1], gain=2.0**0.5)
            layers.append(nn.LayerNorm(self.hidden_dim))
            layers.append(nn.SiLU())
        self.action_encoder = nn.Sequential(*layers)

        # Message passing block: aggregate neighbors per action node.
        self.message_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                    nn.SiLU(),
                )
                for _ in range(self.message_layers)
            ]
        )

        self.score_head = nn.Sequential(
            nn.Linear(self.hidden_dim + self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        _orthogonal_init(self.score_head[-1], gain=0.01)

    def _resource_id(self, one_hot: torch.Tensor) -> int:
        # one_hot includes index 0 for None (no requested resource).
        idx = int(torch.argmax(one_hot).item())
        return idx - 1

    def _action_adjacency(self, branch_actions: torch.Tensor) -> torch.Tensor:
        # Build adjacency by shared selected / from / next resource indices.
        # Positive resource id means real intersection; 0 means None.
        if branch_actions.numel() == 0:
            return branch_actions.new_empty((0, 0))

        n_branches = branch_actions.shape[0]
        selected_ids = torch.full((self.n_robots, n_branches), -1, dtype=torch.long)
        from_ids = torch.full((self.n_robots, n_branches), -1, dtype=torch.long)
        next_ids = torch.full((self.n_robots, n_branches), -1, dtype=torch.long)

        for n in range(self.n_robots):
            start = n * self.per_robot_action_dim
            selected = branch_actions[:, start : start + self.resource_dim + 1]
            from_r = branch_actions[
                :, start + (self.resource_dim + 1) : start + 2 * (self.resource_dim + 1)
            ]
            next_r = branch_actions[
                :,
                start + 2 * (self.resource_dim + 1) : start + 3 * (self.resource_dim + 1)
            ]
            for b in range(n_branches):
                selected_ids[n, b] = self._resource_id(selected[b])
                from_ids[n, b] = self._resource_id(from_r[b])
                next_ids[n, b] = self._resource_id(next_r[b])

        adj = torch.eye(
            n_branches, dtype=torch.float32, device=branch_actions.device
        )
        for i in range(n_branches):
            for j in range(i + 1, n_branches):
                conflict = False
                for n in range(self.n_robots):
                    if (
                        selected_ids[n, i] >= 0
                        and selected_ids[n, i] == selected_ids[n, j]
                    ):
                        conflict = True
                        break
                    if from_ids[n, i] >= 0 and from_ids[n, i] == from_ids[n, j]:
                        conflict = True
                        break
                    if next_ids[n, i] >= 0 and next_ids[n, i] == next_ids[n, j]:
                        conflict = True
                        break
                if conflict:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0

        return adj

    def forward(
        self,
        state: torch.Tensor,
        branch_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Return one logit per legal branch."""

        if state.ndim != 1 or state.numel() != self.state_dim:
            raise ValueError(f"bad state shape {tuple(state.shape)}")
        if branch_actions.ndim != 2 or branch_actions.shape[1] != self.action_dim:
            raise ValueError(f"bad branch action shape {tuple(branch_actions.shape)}")
        if branch_actions.shape[0] == 0:
            return torch.empty(0, device=state.device, dtype=state.dtype)

        state_embedding = self.state_embed(state)
        h = self.action_encoder(branch_actions)

        adjacency = self._action_adjacency(branch_actions)
        if adjacency.numel() > 0:
            row_sum = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
            normalized = adjacency / row_sum
            for layer in self.message_mlps:
                m = normalized @ h
                h = layer(torch.cat([h, m], dim=-1))

        repeated_state = state_embedding.unsqueeze(0).expand(h.shape[0], -1)
        logits = self.score_head(torch.cat([h, repeated_state], dim=-1)).squeeze(-1)
        return logits


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
