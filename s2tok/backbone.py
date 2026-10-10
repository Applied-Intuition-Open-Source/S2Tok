# Copyright 2026 The S2Tok Authors.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International
# License (CC BY-NC 4.0). See the LICENSE file in the root of this repository or
# https://creativecommons.org/licenses/by-nc/4.0/.
#
# This file is derived from ZipSplat (https://github.com/cvg/ZipSplat), Copyright 2026
# ETH Zurich; DINOv2 (https://github.com/facebookresearch/dinov2); and Depth Anything 3
# (https://github.com/ByteDance-Seed/Depth-Anything-3), Copyright (c) 2025 ByteDance; all
# licensed under the Apache License, Version 2.0. It was modified by The S2Tok Authors.
# The original notice is retained below; the Apache License 2.0 text is in
# THIRD_PARTY_LICENSES. See NOTICE.
#
# Original notice:
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# ViT-Giant backbone of Depth Anything 3 (DINOv2 architecture with alternating
# frame-local / global attention, QK-norm, 2D RoPE and learned camera tokens).
#
# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/ByteDance-Seed/Depth-Anything-3

import math
from functools import partial

import torch
from torch import nn

from s2tok.modules.blocks import SelfAttentionBlock, SwiGLUFFN
from s2tok.modules.patch_embed import PatchEmbed
from s2tok.modules.rope import PositionGetter, RotaryPositionEmbedding2D


class DinoVisionTransformer(nn.Module):
    """DA3 ViT-G/14. Blocks ``[0, alt_start)`` see one frame; from ``alt_start`` on, odd
    blocks are global and even blocks are frame-local."""

    def __init__(
        self,
        embed_dim: int = 1536,
        depth: int = 40,
        num_heads: int = 24,
        mlp_ratio: float = 4.0,
        patch_size: int = 14,
        pretrain_img_size: int = 518,
        alt_start: int = 13,
        rope_freq: float = 100.0,
        interpolate_offset: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_blocks = depth
        self.patch_size = patch_size
        self.alt_start = alt_start
        self.interpolate_offset = interpolate_offset

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        num_patches = (pretrain_img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # [reference-view token, source-view token], written into the cls slot at alt_start.
        self.camera_token = nn.Parameter(torch.zeros(1, 2, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq)
        self.position_getter = PositionGetter()
        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    init_values=1.0,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    ffn_layer=SwiGLUFFN,
                    qk_norm=i >= alt_start,
                    rope=self.rope if i >= alt_start else None,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def is_global(self, i: int) -> bool:
        return i >= self.alt_start and i % 2 == 1

    def interpolate_pos_encoding(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and h == w:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        h0 = h // self.patch_size
        w0 = w // self.patch_size
        M = int(math.sqrt(N))
        assert N == M * M
        sh = float(h0 + self.interpolate_offset) / M
        sw = float(w0 + self.interpolate_offset) / M
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=False,
            scale_factor=(sh, sw),
        )
        assert (h0, w0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] -> [B, 1 + T, D] (cls slot first)."""
        B, _, H, W = images.shape
        x = self.patch_embed(images)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        return x + self.interpolate_pos_encoding(x, H, W)

    def prepare_rope(self, B: int, H: int, W: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """2D positions for frame-local blocks and the constant positions used by global
        blocks. Patches are offset by one; the cls/camera slot sits at (0, 0)."""
        pos = self.position_getter(B, H // self.patch_size, W // self.patch_size, device=device)
        special = torch.zeros(B, 1, 2, device=device, dtype=pos.dtype)
        pos_global = torch.cat([special, torch.ones_like(pos)], dim=1)
        pos_local = torch.cat([special, pos + 1], dim=1)
        return pos_local, pos_global


class DA3Encoder(nn.Module):
    """Holds the ViT under ``.backbone`` so parameter names match the training checkpoints."""

    def __init__(self):
        super().__init__()
        self.backbone = DinoVisionTransformer()
