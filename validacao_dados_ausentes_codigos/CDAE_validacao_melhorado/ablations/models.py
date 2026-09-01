"""Capacity-matched auxiliary-only MLP (g(x) -> t_hat)."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


def tcdr_hidden_widths(n_features: int, latent_dim: int = 16) -> Tuple[int, int, int]:
    """
    Match TCDR widths: TCDR uses in_dim = n_features + 2 for sizing.
    The aux MLP keeps those widths but takes only x as input.
    """
    capacity_in_dim = int(n_features) + 2
    h1 = max(64, 8 * capacity_in_dim)
    h2 = max(32, 4 * capacity_in_dim)
    return h1, h2, int(latent_dim)


class AuxiliaryOnlyMLP(nn.Module):
    """MLP soft sensor: input x only, output scalar target."""

    def __init__(
        self,
        n_features: int,
        latent_dim: int = 16,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        h1, h2, latent = tcdr_hidden_widths(n_features, latent_dim)
        self.n_features = int(n_features)
        self.capacity_basis_in_dim = int(n_features) + 2
        self.h1 = h1
        self.h2 = h2
        self.latent_dim = latent
        self.net = nn.Sequential(
            nn.Linear(self.n_features, h1),
            nn.BatchNorm1d(h1),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout / 2.0),
            nn.Linear(h2, latent),
            nn.BatchNorm1d(latent),
            nn.LeakyReLU(0.1),
            nn.Linear(latent, h2),
            nn.LeakyReLU(0.1),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def config_dict(self) -> Dict[str, object]:
        return {
            "model_type": "aux_mlp",
            "n_features": self.n_features,
            "capacity_basis_in_dim": self.capacity_basis_in_dim,
            "h1": self.h1,
            "h2": self.h2,
            "latent_dim": self.latent_dim,
            "parameter_count": self.parameter_count(),
        }
