from __future__ import annotations

import torch
import torch.nn as nn


class Projection(nn.Module):
    """Linear projection followed by ReLU, as described in the manuscript."""

    def __init__(self, in_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MLPClassifier(nn.Module):
    """Two-layer MLP classifier for 256-dimensional fused representations."""

    def __init__(self, in_dim: int, num_classes: int = 2, dropout: float = 0.5) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ConcatenationFusionNet(nn.Module):
    """Patient-level concatenation fusion baseline."""

    def __init__(
        self,
        wsi_dim: int = 512,
        ct_dim: int = 512,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.wsi_proj = Projection(wsi_dim, hidden_dim)
        self.ct_proj = Projection(ct_dim, hidden_dim)
        self.classifier = MLPClassifier(hidden_dim * 2, num_classes, dropout)

    def forward(self, wsi_feat: torch.Tensor, ct_feat: torch.Tensor) -> torch.Tensor:
        wsi_latent = self.wsi_proj(wsi_feat)
        ct_latent = self.ct_proj(ct_feat)
        fused = torch.cat([wsi_latent, ct_latent], dim=1)
        return self.classifier(fused)


class GatedFusionNet(nn.Module):
    """Feature-wise gated fusion baseline."""

    def __init__(
        self,
        wsi_dim: int = 512,
        ct_dim: int = 512,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.wsi_proj = Projection(wsi_dim, hidden_dim)
        self.ct_proj = Projection(ct_dim, hidden_dim)
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.Sigmoid())
        self.classifier = MLPClassifier(hidden_dim * 2, num_classes, dropout)

    def forward(self, wsi_feat: torch.Tensor, ct_feat: torch.Tensor) -> torch.Tensor:
        wsi_latent = self.wsi_proj(wsi_feat)
        ct_latent = self.ct_proj(ct_feat)
        fused = torch.cat([wsi_latent, ct_latent], dim=1)
        gated = fused * self.gate(fused)
        return self.classifier(gated)


class WSIGuidedAttentionFusionNet(nn.Module):
    """M2M-ALK: WSI-guided representation-level attention fusion.

    The projected WSI representation is the query. The projected CT representation
    is both key and value. The attention output is added back to the projected WSI
    representation through a residual connection and layer normalization.
    """

    def __init__(
        self,
        wsi_dim: int = 512,
        ct_dim: int = 512,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_classes: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.wsi_proj = Projection(wsi_dim, hidden_dim)
        self.ct_proj = Projection(ct_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = MLPClassifier(hidden_dim * 2, num_classes, dropout)

    def forward(self, wsi_feat: torch.Tensor, ct_feat: torch.Tensor) -> torch.Tensor:
        wsi_latent = self.wsi_proj(wsi_feat)
        ct_latent = self.ct_proj(ct_feat)
        query = wsi_latent.unsqueeze(1)
        key = ct_latent.unsqueeze(1)
        value = ct_latent.unsqueeze(1)
        attn_out, _ = self.attention(query, key, value, need_weights=False)
        ct_informed_wsi = self.norm(attn_out.squeeze(1) + wsi_latent)
        fused = torch.cat([ct_informed_wsi, ct_latent], dim=1)
        return self.classifier(fused)


def build_fusion_model(
    model_name: str,
    wsi_dim: int = 512,
    ct_dim: int = 512,
    hidden_dim: int = 128,
    num_heads: int = 4,
    dropout: float = 0.5,
    num_classes: int = 2,
) -> nn.Module:
    name = model_name.lower().replace("-", "_")
    if name in {"concat", "concatenation", "simple_concat"}:
        return ConcatenationFusionNet(wsi_dim, ct_dim, hidden_dim, num_classes, dropout)
    if name in {"gated", "gated_fusion"}:
        return GatedFusionNet(wsi_dim, ct_dim, hidden_dim, num_classes, dropout)
    if name in {"attention", "cross_attention", "wsi_guided_attention", "m2m_alk"}:
        return WSIGuidedAttentionFusionNet(
            wsi_dim, ct_dim, hidden_dim, num_heads, num_classes, dropout
        )
    raise ValueError(
        "Unknown fusion model. Choose from: concat, gated, attention/m2m_alk."
    )
