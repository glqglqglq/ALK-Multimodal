from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from monai.networks.nets import ResNet

from .utils import load_checkpoint_state_dict


class CTResNet18(nn.Module):
    """3D ResNet18 CT model with a 512-dimensional penultimate feature.

    The model uses a MONAI 3D ResNet18 backbone, adaptive average pooling and
    adaptive max pooling, a 1024 -> 512 feature head, and a two-class classifier.
    ``extract_features`` returns the 512-dimensional radiological representation
    used by the multimodal fusion stage.
    """

    def __init__(
        self,
        n_input_channels: int = 1,
        num_classes: int = 2,
        feature_dim: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        base = ResNet(
            block="basic",
            layers=[2, 2, 2, 2],
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=3,
            n_input_channels=n_input_channels,
            num_classes=num_classes,
        )
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.maxpool = nn.AdaptiveMaxPool3d(1)
        self.feature_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, feature_dim),
            nn.ReLU(inplace=True),
        )
        classifier_layers = []
        if dropout > 0:
            classifier_layers.append(nn.Dropout(dropout))
        classifier_layers.append(nn.Linear(feature_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def pooled_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = torch.cat([self.avgpool(x), self.maxpool(x)], dim=1)
        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_head(self.pooled_backbone_features(x))

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.extract_features(x)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


def load_med3d_pretrained_weights(
    model: nn.Module,
    weights_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> None:
    """Load Med3D/ResNet18-style weights while ignoring task-specific heads."""
    state_dict = load_checkpoint_state_dict(weights_path, map_location=map_location)
    filtered = {
        k: v
        for k, v in state_dict.items()
        if not any(
            head_key in k
            for head_key in (
                "fc",
                "classifier",
                "feature_head",
                "last_linear",
                "head",
            )
        )
    }
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    # Avoid printing huge lists in normal use; leave details inspectable by caller.
    return None
