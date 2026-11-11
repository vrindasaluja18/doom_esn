"""Frozen pretrained perception: pixels -> a small latent, no from-scratch training.

The program's plan (docs/RESEARCH_PROGRAM.md) spends its effort on *memory*, not on
training a perception model, so perception is a frozen pretrained CNN used as a feature
extractor. Weights are downloaded once and never updated; only the memory and readout are
learned. This keeps the agent CPU-only and honest about where the bytes go: the encoder is
reported separately from the budgeted memory state.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from ._types import F32

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


class MobileNetEncoder:
    """Frozen ImageNet MobileNetV3-Small trunk; outputs a pooled ``dim``-d latent."""

    def __init__(self, size: int = 96, pretrained: bool = True) -> None:
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        net = mobilenet_v3_small(weights=weights)
        self.features = net.features
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.features.eval()
        for p in self.features.parameters():
            p.requires_grad_(False)
        self.size = size
        self.dim = 576  # mobilenet_v3_small last-channel width
        self.name = "mobilenet_v3_small_imagenet" if pretrained else "mobilenet_v3_small_random"
        self.params = sum(p.numel() for p in self.features.parameters())

    def _prep(self, frame: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.ascontiguousarray(frame)).float().permute(2, 0, 1) / 255.0
        x = nn.functional.interpolate(
            x[None], size=(self.size, self.size), mode="bilinear", align_corners=False
        )
        mean = torch.from_numpy(_MEAN)[None, :, None, None]
        std = torch.from_numpy(_STD)[None, :, None, None]
        return (x - mean) / std

    @torch.no_grad()
    def __call__(self, frame: np.ndarray) -> F32:
        f = self.features(self._prep(frame))
        return self.pool(f).flatten().numpy().astype(np.float32)

    def state_nbytes(self) -> int:
        return 0  # perception is not part of the budgeted memory state

    def weight_nbytes(self) -> int:
        return self.params * 4  # float32 weights; int8 would be ~1/4 of this
