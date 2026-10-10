# Copyright 2026 The S2Tok Authors.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International
# License (CC BY-NC 4.0). See the LICENSE file in the root of this repository or
# https://creativecommons.org/licenses/by-nc/4.0/.
#
# This file is derived from ZipSplat (https://github.com/cvg/ZipSplat), Copyright 2026
# ETH Zurich; DINOv2 (https://github.com/facebookresearch/dinov2); and timm
# (https://github.com/huggingface/pytorch-image-models); all licensed under the Apache
# License, Version 2.0. It was modified by The S2Tok Authors.
# The original notice is retained below; the Apache License 2.0 text is in
# THIRD_PARTY_LICENSES. See NOTICE.
#
# Original notice:
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from s2tok.modules.attention import CrossAttention, SelfAttention


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network with the DINOv2 ViT-g hidden-width rounding."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        init_values: float | None = None,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        rope: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SelfAttention(dim, num_heads=num_heads, qk_norm=qk_norm, rope=rope)
        self.ls1 = LayerScale(dim, init_values) if init_values else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = ffn_layer(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim, init_values) if init_values else nn.Identity()

    def forward(self, x: Tensor, pos: Tensor | None = None) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x), pos=pos))
        return x + self.ls2(self.mlp(self.norm2(x)))


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        init_values: float | None = None,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = CrossAttention(dim, num_heads=num_heads, qk_norm=qk_norm)
        self.ls1 = LayerScale(dim, init_values) if init_values else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim, init_values) if init_values else nn.Identity()

    def forward(self, queries: Tensor, context: Tensor, return_delta: bool = False) -> Tensor:
        """Standard cross-attention block.

        With ``return_delta=True`` only the attention + MLP update is returned (the MLP
        still sees the post-attention state), so the caller can add it to a residual
        stream other than ``queries``.
        """
        attn_delta = self.ls1(self.attn(self.norm_q(queries), self.norm_kv(context)))
        post_attn = queries + attn_delta
        mlp_delta = self.ls2(self.mlp(self.norm2(post_attn)))
        if return_delta:
            return attn_delta + mlp_delta
        return post_attn + mlp_delta
