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
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import torch.nn.functional as F
from torch import Tensor, nn


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        qk_norm: bool = False,
        rope: nn.Module | None = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.rope = rope

    def forward(self, x: Tensor, pos: Tensor | None = None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    def forward(self, queries: Tensor, context: Tensor) -> Tensor:
        B, N_q, C = queries.shape
        N_c = context.shape[1]

        q = self.q_proj(queries).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = (
            self.kv_proj(context)
            .reshape(B, N_c, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N_q, C)
        return self.proj(x)
