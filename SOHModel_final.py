"""SOH regression model.

Neural network for battery State of Health (SOH) prediction.

Architecture:
- Physics-temporal decoupled encoder:
  - Physics encoder processes physical features (all input dims except time)
  - Temporal encoder processes time feature (last dim)
  - Learnable gating fuses physics and temporal embeddings
- Prediction head:
  - MLP head followed by sigmoid to map outputs into (0, 1)
- Optional feature regularization:
  - Smoothness across adjacent time steps
  - Energy non-increase across time steps
  - Light L2 constraint
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class Mish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class PhysicsEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, layers_num: int, dropout: float):
        super().__init__()
        if layers_num < 2:
            raise ValueError('layers_num must be >= 2')

        self.input_norm = nn.BatchNorm1d(input_dim)

        layers = []
        for i in range(layers_num):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == layers_num - 1 else hidden_dim

            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(Mish())
            if i != layers_num - 1:
                layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        return self.net(x)


class TemporalEncoder(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, max(4, hidden_dim // 2)),
            Mish(),
            nn.Linear(max(4, hidden_dim // 2), output_dim),
            Mish(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


class FeatureGating(nn.Module):
    """Scalar learnable gate for physics/time fusion."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.0))  # sigmoid(alpha) -> 0.5 init

        self.refine = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            Mish(),
            nn.Dropout(0.1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, h_physics: torch.Tensor, h_time: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        a = torch.sigmoid(self.alpha)
        h = a * h_physics + (1.0 - a) * h_time
        h = self.refine(h)
        return h, {'physics_weight': float(a.detach().cpu().item()), 'temporal_weight': float((1.0 - a).detach().cpu().item())}


class PhysicsTemporalEncoder(nn.Module):
    """Split input into physics features and time, then fuse."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, layers_num: int, dropout: float):
        super().__init__()
        if input_dim < 2:
            raise ValueError('input_dim must be >= 2 (physics + time)')

        self.physics_encoder = PhysicsEncoder(
            input_dim=input_dim - 1,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            layers_num=layers_num,
            dropout=dropout,
        )
        self.temporal_encoder = TemporalEncoder(hidden_dim=hidden_dim, output_dim=output_dim)
        self.gating = FeatureGating(feature_dim=output_dim)

    def forward(self, xt: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        x_physics = xt[:, :-1]
        x_time = xt[:, -1:]
        h_physics = self.physics_encoder(x_physics)
        h_time = self.temporal_encoder(x_time)
        h, gate = self.gating(h_physics, h_time)
        return h, gate


class Predictor(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(input_dim, 32),
            Mish(),
            nn.Linear(32, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureRegularizer(nn.Module):
    """Feature-space regularization (fixed weights)."""

    def __init__(
        self,
        lambda_smooth: float,
        lambda_energy: float,
        lambda_causal: float,
    ):
        super().__init__()
        self.lambda_smooth = float(lambda_smooth)
        self.lambda_energy = float(lambda_energy)
        self.lambda_causal = float(lambda_causal)

    def smoothness(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        return torch.mean((h2 - h1) ** 2)

    def energy_non_increase(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        e1 = torch.norm(h1, p=2, dim=1)
        e2 = torch.norm(h2, p=2, dim=1)
        inc = torch.relu(e2 - e1)
        return torch.mean(inc ** 2)

    def causal_l2(self, h: torch.Tensor) -> torch.Tensor:
        # Lightweight constraint; keep consistent with your current implementation spirit.
        return 0.01 * torch.mean(h ** 2)

    def forward(self, h1: torch.Tensor, h2: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor) -> Dict[str, torch.Tensor]:
        # t1, t2 are kept for API consistency; current terms don't explicitly use them.
        loss_smooth = self.smoothness(h1, h2)
        loss_energy = self.energy_non_increase(h1, h2)
        loss_causal = 0.5 * (self.causal_l2(h1) + self.causal_l2(h2))

        weighted_smooth = self.lambda_smooth * loss_smooth
        weighted_energy = self.lambda_energy * loss_energy
        weighted_causal = self.lambda_causal * loss_causal
        total = weighted_smooth + weighted_energy + weighted_causal

        return {
            'smooth': loss_smooth,
            'energy': loss_energy,
            'causal': loss_causal,
            'weighted_smooth': weighted_smooth,
            'weighted_energy': weighted_energy,
            'weighted_causal': weighted_causal,
            'total_reg': total,
        }


@dataclass
class ModelConfig:
    input_dim: int = 17
    enc_layers: int = 3
    enc_hidden_dim: int = 128
    emb_dim: int = 64
    enc_dropout: float = 0.2
    head_dropout: float = 0.2

    # feature regularization
    feature_reg: bool = True
    lambda_smooth: float = 0.01
    lambda_energy: float = 0.005
    lambda_causal: float = 0.005



class SOHRegressor(nn.Module):
    """Final SOH regression model.

    Forward returns:
        soh: [B,1]
        embedding: [B, emb_dim]
        gate_info: dict (fusion weights)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder = PhysicsTemporalEncoder(
            input_dim=cfg.input_dim,
            hidden_dim=cfg.enc_hidden_dim,
            output_dim=cfg.emb_dim,
            layers_num=cfg.enc_layers,
            dropout=cfg.enc_dropout,
        )
        self.head = Predictor(input_dim=cfg.emb_dim, dropout=cfg.head_dropout)

        self.feature_regularizer: Optional[FeatureRegularizer]
        if cfg.feature_reg:
            self.feature_regularizer = FeatureRegularizer(
                lambda_smooth=cfg.lambda_smooth,
                lambda_energy=cfg.lambda_energy,
                lambda_causal=cfg.lambda_causal,
            )
        else:
            self.feature_regularizer = None

    def get_embedding(self, xt: torch.Tensor) -> torch.Tensor:
        h, _ = self.encoder(xt)
        return h

    def forward(self, xt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        h, gate = self.encoder(xt)
        soh = torch.sigmoid(self.head(h))
        return soh, h, gate

