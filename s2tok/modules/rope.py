# Copyright 2026 The S2Tok Authors.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International
# License (CC BY-NC 4.0). See the LICENSE file in the root of this repository or
# https://creativecommons.org/licenses/by-nc/4.0/.
#
# This file is derived from ZipSplat (https://github.com/cvg/ZipSplat), Copyright 2026
# ETH Zurich, and DINOv2 (https://github.com/facebookresearch/dinov2), both licensed
# under the Apache License, Version 2.0. It was modified by The S2Tok Authors.
# The original notice is retained below; the Apache License 2.0 text is in
# THIRD_PARTY_LICENSES. See NOTICE.
#
# Original notice:
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# 2D rotary position embedding for the image-grid attention of the backbone.
#
# Inspired by:
#         https://github.com/meta-llama/codellama/blob/main/llama/model.py
#         https://github.com/naver-ai/rope-vit

import torch
import torch.nn.functional as F
from torch import nn


class PositionGetter:
    """Generates and caches (y, x) grid coordinates for the patches of an image."""

    def __init__(self):
        self.position_cache: dict[tuple[int, int], torch.Tensor] = {}

    def __call__(
        self, batch_size: int, height: int, width: int, device: torch.device
    ) -> torch.Tensor:
        """Returns (batch_size, height * width, 2) integer coordinates."""
        if (height, width) not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            self.position_cache[height, width] = torch.cartesian_prod(y_coords, x_coords)
        positions = self.position_cache[height, width]
        return positions.view(1, height * width, 2).expand(batch_size, -1, -1)


class RotaryPositionEmbedding2D(nn.Module):
    """Rotates half of each head's channels by the y coordinate and half by x."""

    def __init__(self, frequency: float = 100.0):
        super().__init__()
        self.base_frequency = frequency
        self.frequency_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    def _compute_frequency_components(
        self, dim: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_key = (dim, seq_len, device, dtype)
        if cache_key not in self.frequency_cache:
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency**exponents)
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq).to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            self.frequency_cache[cache_key] = (angles.cos().to(dtype), angles.sin().to(dtype))
        return self.frequency_cache[cache_key]

    @staticmethod
    def _rotate_features(x: torch.Tensor) -> torch.Tensor:
        feature_dim = x.shape[-1]
        x1, x2 = x[..., : feature_dim // 2], x[..., feature_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(self, tokens, positions, cos_comp, sin_comp) -> torch.Tensor:
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]
        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """tokens: (B, heads, N, head_dim); positions: (B, N, 2) integer (y, x)."""
        assert tokens.size(-1) % 2 == 0, "Feature dimension must be even"
        assert positions.ndim == 3 and positions.shape[-1] == 2
        feature_dim = tokens.size(-1) // 2
        # The table only has to cover the largest position index. Grid coordinates
        # (offset by one for the camera token) never exceed the token count, so sizing
        # it from the shape avoids reading positions.max() back from the device.
        cos_comp, sin_comp = self._compute_frequency_components(
            feature_dim, positions.shape[1] + 2, tokens.device, tokens.dtype
        )
        vertical, horizontal = tokens.chunk(2, dim=-1)
        vertical = self._apply_1d_rope(vertical, positions[..., 0], cos_comp, sin_comp)
        horizontal = self._apply_1d_rope(horizontal, positions[..., 1], cos_comp, sin_comp)
        return torch.cat((vertical, horizontal), dim=-1)
