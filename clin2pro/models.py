import math
from typing import Dict

import torch
from torch import nn


class NumericalFeatureTokenizer(nn.Module):
    def __init__(self, n_features: int, hidden: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, hidden))
        self.bias = nn.Parameter(torch.empty(n_features, hidden))
        nn.init.normal_(self.weight, std=0.02)
        nn.init.normal_(self.bias, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class ClinicalEncoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden: int,
        layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden = hidden
        self.tokenizer = NumericalFeatureTokenizer(n_features, hidden)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
        self.feature_embedding = nn.Parameter(torch.empty(1, n_features, hidden))
        nn.init.normal_(self.feature_embedding, std=0.02)
        block = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(block, num_layers=layers)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x) + self.feature_embedding
        cls = self.cls.expand(x.shape[0], -1, -1)
        encoded = self.transformer(torch.cat([cls, tokens], dim=1))
        return self.norm(encoded[:, 0])


class ProteinPredictionHead(nn.Module):
    def __init__(self, hidden: int, n_proteins: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, n_proteins),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.network(latent)


class CrossSectionalModel(nn.Module):
    def __init__(self, encoder: ClinicalEncoder, head: ProteinPredictionHead) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int, output_dim: int, dropout: float):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden),
                    nn.Linear(hidden, hidden * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden * 2, hidden),
                    nn.Dropout(dropout),
                )
                for _ in range(3)
            ]
        )
        self.norm = nn.LayerNorm(hidden)
        self.output = nn.Linear(hidden, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        for block in self.blocks:
            x = x + block(x)
        return self.output(self.norm(x))


class TemporalPerturbationEncoder(nn.Module):
    def __init__(
        self, clinical_hidden: int, n_features: int, hidden: int, dropout: float
    ) -> None:
        super().__init__()
        input_dim = clinical_hidden * 3 + n_features * 2 + 2
        self.network = ResidualMLP(input_dim, hidden, hidden, dropout)

    def forward(
        self,
        z_pre: torch.Tensor,
        z_post: torch.Tensor,
        x_pre: torch.Tensor,
        x_post: torch.Tensor,
        interval_years: torch.Tensor,
    ) -> torch.Tensor:
        delta = x_post - x_pre
        rate = (delta / interval_years.clamp_min(1.0 / 365.25)).clamp(-20.0, 20.0)
        log_days = torch.log1p(interval_years.clamp_min(0.0) * 365.25)
        time_features = torch.cat([interval_years, log_days], dim=1)
        inputs = torch.cat(
            [z_pre, z_post, z_post - z_pre, delta, rate, time_features], dim=1
        )
        return self.network(inputs)


class DrugPretrainingModel(nn.Module):
    def __init__(
        self,
        clinical_encoder: ClinicalEncoder,
        temporal_encoder: TemporalPerturbationEncoder,
        temporal_hidden: int,
        n_targets: int,
        n_pathways: int,
    ) -> None:
        super().__init__()
        self.clinical_encoder = clinical_encoder
        self.temporal_encoder = temporal_encoder
        self.affinity_head = nn.Linear(temporal_hidden, n_targets)
        self.binary_head = nn.Linear(temporal_hidden, n_targets)
        self.pathway_head = nn.Linear(temporal_hidden, n_pathways)

    def forward(
        self, x_pre: torch.Tensor, x_post: torch.Tensor, interval_years: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        z_pre = self.clinical_encoder(x_pre)
        z_post = self.clinical_encoder(x_post)
        temporal = self.temporal_encoder(
            z_pre, z_post, x_pre, x_post, interval_years
        )
        return {
            "temporal": temporal,
            "affinity": self.affinity_head(temporal),
            "binary_logits": self.binary_head(temporal),
            "pathway": self.pathway_head(temporal),
        }


class LowRankProteinTargetBridge(nn.Module):
    def __init__(self, n_proteins: int, n_targets: int, rank: int) -> None:
        super().__init__()
        scale = 1.0 / math.sqrt(rank)
        self.U = nn.Parameter(torch.randn(n_proteins, rank) * scale * 0.05)
        self.V = nn.Parameter(torch.randn(n_targets, rank) * scale * 0.05)

    def forward(self, delta_protein: torch.Tensor) -> torch.Tensor:
        return delta_protein.matmul(self.U).matmul(self.V.transpose(0, 1))

    def regularization(self, l1_weight: float = 0.0) -> torch.Tensor:
        penalty = self.U.pow(2).mean() + self.V.pow(2).mean()
        if l1_weight > 0:
            penalty = penalty + l1_weight * (
                self.U.abs().mean() + self.V.abs().mean()
            )
        return penalty

    def matrix(self) -> torch.Tensor:
        return self.U.matmul(self.V.transpose(0, 1))


class LongitudinalModel(nn.Module):
    def __init__(
        self,
        clinical_encoder: ClinicalEncoder,
        protein_head: ProteinPredictionHead,
        temporal_encoder: TemporalPerturbationEncoder,
        temporal_hidden: int,
        clinical_hidden: int,
        n_proteins: int,
        n_targets: int,
        bridge_rank: int,
        protein_std: torch.Tensor,
        dropout: float,
    ) -> None:
        super().__init__()
        self.clinical_encoder = clinical_encoder
        self.protein_head = protein_head
        self.temporal_encoder = temporal_encoder
        self.delta_head = ResidualMLP(
            temporal_hidden + clinical_hidden,
            temporal_hidden,
            n_proteins,
            dropout,
        )
        self.bridge = LowRankProteinTargetBridge(
            n_proteins, n_targets, bridge_rank
        )
        self.register_buffer("protein_std", protein_std.reshape(1, -1))

    def forward(
        self, x_pre: torch.Tensor, x_post: torch.Tensor, interval_years: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        z_pre = self.clinical_encoder(x_pre)
        z_post = self.clinical_encoder(x_post)
        temporal = self.temporal_encoder(
            z_pre, z_post, x_pre, x_post, interval_years
        )
        baseline_standardized = self.protein_head(z_post) - self.protein_head(z_pre)
        baseline_delta = baseline_standardized * self.protein_std
        residual = self.delta_head(torch.cat([temporal, z_post - z_pre], dim=1))
        predicted_delta = baseline_delta + residual
        target_like = self.bridge(predicted_delta)
        return {
            "predicted_delta": predicted_delta,
            "target_like": target_like,
            "temporal": temporal,
            "baseline_delta": baseline_delta,
        }
