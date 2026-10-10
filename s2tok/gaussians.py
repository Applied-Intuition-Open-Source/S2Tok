# Copyright 2026 The S2Tok Authors.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International
# License (CC BY-NC 4.0). See the LICENSE file in the root of this repository or
# https://creativecommons.org/licenses/by-nc/4.0/.
#
# Portions of this file are derived from ZipSplat (https://github.com/cvg/ZipSplat),
# Copyright 2026 ETH Zurich, licensed under the Apache License, Version 2.0, and were
# modified by The S2Tok Authors. See NOTICE and THIRD_PARTY_LICENSES.

"""Gaussian decoding head, the Gaussian container and a standard 3DGS PLY writer."""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from s2tok.modules.blocks import Mlp


@dataclass
class Gaussians:
    """A set of N 3D Gaussians (activated values)."""

    means: torch.Tensor  # [N, 3]
    scales: torch.Tensor  # [N, 3] linear scales
    quats: torch.Tensor  # [N, 4] unit quaternions, wxyz
    opacities: torch.Tensor  # [N] in (0, 1)
    sh: torch.Tensor  # [N, K, 3] spherical-harmonics coefficients, K = (degree + 1) ** 2

    def __len__(self) -> int:
        return self.means.shape[0]

    def to(self, device: str | torch.device) -> "Gaussians":
        return Gaussians(**{k: v.to(device) for k, v in vars(self).items()})

    def save_ply(self, path: str | os.PathLike) -> None:
        write_gaussian_ply(path, self)


def inverse_log_transform(y: torch.Tensor) -> torch.Tensor:
    """sign(y) * (exp(|y|) - 1)."""
    return torch.sign(y) * torch.expm1(torch.abs(y))


class GaussianHead(nn.Module):
    """Decodes every token into ``gaussians_per_token`` Gaussians with free 3D offsets."""

    def __init__(self, embed_dim: int, gaussians_per_token: int = 32, sh_degree: int = 1):
        super().__init__()
        self.gaussians_per_token = gaussians_per_token
        self.num_sh_coeffs = (sh_degree + 1) ** 2
        # per Gaussian: mean 3, scale 3, quat 4, opacity 1, SH 3K
        self.params_per_gaussian = 3 + 3 + 4 + 1 + 3 * self.num_sh_coeffs
        self.gaussian_head = nn.Sequential(
            Mlp(in_features=embed_dim, hidden_features=2 * embed_dim, out_features=embed_dim),
            nn.Linear(embed_dim, gaussians_per_token * self.params_per_gaussian),
        )

    def forward(self, tokens: torch.Tensor) -> Gaussians:
        """tokens: [N_tok, embed_dim] -> Gaussians with N_tok * gaussians_per_token entries,
        ordered token by token."""
        params = self.gaussian_head(tokens)
        g = rearrange(
            params, "N (G C) -> (N G) C", G=self.gaussians_per_token, C=self.params_per_gaussian
        )
        means = inverse_log_transform(g[..., 0:3].clamp(min=-5.0, max=5.0))
        scales = F.softplus(g[..., 3:6] - 4.0).clamp(1e-6, 15.0)
        quats = F.normalize(g[..., 6:10], dim=-1)
        opacities = g[..., 10:11].sigmoid()[..., 0]
        sh = rearrange(g[..., 11:], "N (K C) -> N K C", K=self.num_sh_coeffs, C=3)
        return Gaussians(means=means, scales=scales, quats=quats, opacities=opacities, sh=sh)


def write_gaussian_ply(path: str | os.PathLike, gaussians: Gaussians) -> None:
    """Write Gaussians as a binary PLY with the standard 3DGS property layout
    (log scales, logit opacities, wxyz rotations, channel-major SH rest)."""

    def to_numpy(t: torch.Tensor) -> np.ndarray:
        return t.detach().float().cpu().numpy()

    xyz = to_numpy(gaussians.means)
    n = xyz.shape[0]
    sh = to_numpy(gaussians.sh)
    dc = sh[:, 0]
    rest = sh[:, 1:].transpose(0, 2, 1).reshape(n, -1)
    opacity = np.clip(to_numpy(gaussians.opacities), 1e-6, 1.0 - 1e-6)
    quat = to_numpy(gaussians.quats)
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True).clip(min=1e-12)

    names = (
        ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
        + [f"f_rest_{i}" for i in range(rest.shape[1])]
        + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    )
    values = np.concatenate(
        [
            xyz,
            np.zeros((n, 3), dtype=np.float32),
            dc,
            rest,
            np.log(opacity / (1.0 - opacity))[:, None],
            np.log(np.clip(to_numpy(gaussians.scales), 1e-12, None)),
            quat,
        ],
        axis=1,
    ).astype(np.float32)
    vertices = np.empty(n, dtype=np.dtype([(name, "<f4") for name in names]))
    for i, name in enumerate(names):
        vertices[name] = values[:, i]

    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        + "".join(f"property float {name}\n" for name in names)
        + "end_header\n"
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        vertices.tofile(f)
