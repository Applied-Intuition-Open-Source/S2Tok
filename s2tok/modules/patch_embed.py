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

from torch import Tensor, nn


class PatchEmbed(nn.Module):
    """2D image to patch embedding: (B, C, H, W) -> (B, N, D)."""

    def __init__(self, patch_size: int = 14, in_chans: int = 3, embed_dim: int = 768) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        H, W = x.shape[-2:]
        assert (
            H % self.patch_size == 0 and W % self.patch_size == 0
        ), f"image size {H}x{W} is not a multiple of the patch size {self.patch_size}"
        return self.proj(x).flatten(2).transpose(1, 2)
