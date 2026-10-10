# Copyright 2026 The S2Tok Authors.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International
# License (CC BY-NC 4.0). See the LICENSE file in the root of this repository or
# https://creativecommons.org/licenses/by-nc/4.0/.

"""Continuous 3D rotary position embedding for the scene-local attention.

Scene tokens have no image grid, so they are positioned by the 3D centroid of the
Gaussians they decoded to at the previous step. Each token's centroid is the
opacity-weighted geometric median of its Gaussians; the whole cloud is centered and
its robust radius ``r`` sets the wavelengths, which makes the embedding invariant to
the global scale of the scene.

Channel layout per head mirrors the 2D RoPE in ``rope.py``:
``[x: 2P | y: 2P | z: 2P | unrotated]``. Band ``i`` of every axis rotates with
wavelength ``lam_top**(1 - g_i) * lam_bot**g_i`` with ``g_i = i / (P - 1)``,
``lam_top = (2 / rho_ref) * r`` and ``lam_bot = 24 * r``. Angles are computed in fp32.
"""

import torch
from torch import nn


def weighted_weiszfeld(
    mu: torch.Tensor, w: torch.Tensor, iters: int, eps: float = 1e-6
) -> torch.Tensor:
    """Weighted geometric median (Weiszfeld iterations). mu: [..., G, 3], w: [..., G]."""
    c = (w.unsqueeze(-1) * mu).sum(-2) / w.sum(-1, keepdim=True).clamp(min=eps)
    for _ in range(iters):
        d = (mu - c.unsqueeze(-2)).norm(dim=-1).clamp(min=eps)
        wd = w / d
        c = (wd.unsqueeze(-1) * mu).sum(-2) / wd.sum(-1, keepdim=True).clamp(min=eps)
    return c


def token_centroids(
    means: torch.Tensor,
    opacities: torch.Tensor,
    iters: int = 8,
    clamp_range: float = 10.0,
    min_mass: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token anchor: opacity-weighted geometric median of the token's Gaussians.

    means: [M, G, 3], opacities: [M, G]. Returns (centroids [M, 3], mass [M]). A token
    whose Gaussians are all transparent falls back to uniform weights.
    """
    mu = means.clamp(-clamp_range, clamp_range)
    mass = opacities.sum(-1)
    w = torch.where(mass.unsqueeze(-1) > min_mass, opacities, torch.ones_like(opacities))
    return weighted_weiszfeld(mu, w, iters), mass


def cloud_center_and_radius(
    x: torch.Tensor, mass: torch.Tensor, trim_top: float = 0.25, iters: int = 10
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mass-weighted geometric median of the token anchors and the mass-weighted mean
    distance to it over the closest 75% of tokens. x: [M, 3], mass: [M]."""
    M = x.shape[0]
    c = weighted_weiszfeld(x.unsqueeze(0), mass.unsqueeze(0), iters)[0]
    r = (x - c).norm(dim=-1)
    keep = r.argsort()[: max(M - int(M * trim_top), 1)]
    radius = (r[keep] * mass[keep]).sum() / mass[keep].sum().clamp(min=1e-8)
    return c, radius


def wavelength_anchors(
    radius: torch.Tensor, rho_ref: float, top_scale: float = 2.0, bot_scale: float = 24.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shortest and longest RoPE wavelengths for a cloud of the given radius."""
    return (top_scale / rho_ref) * radius, bot_scale * radius


def _rotate_half(t: torch.Tensor) -> torch.Tensor:
    p = t.shape[-1] // 2
    return torch.cat((-t[..., p:], t[..., :p]), dim=-1)


class Rope3D(nn.Module):
    """Continuous 3D RoPE applied to q/k. No learnable parameters."""

    def __init__(self, pairs_per_axis: int = 10):
        super().__init__()
        self.P = pairs_per_axis
        g = torch.arange(pairs_per_axis, dtype=torch.float32)
        self.register_buffer("gamma", g / max(pairs_per_axis - 1, 1), persistent=False)

    def wavelengths(self, lam_top: torch.Tensor, lam_bot: torch.Tensor) -> torch.Tensor:
        """Per-band wavelengths. Anchors are [..., N] (one value per token) -> [..., N, P]."""
        lt = lam_top.float().clamp(min=1e-8).log().unsqueeze(-1)
        lb = lam_bot.float().clamp(min=1e-8).log().unsqueeze(-1)
        return torch.exp(lt + (lb - lt) * self.gamma)

    def forward(
        self, tokens: torch.Tensor, pos: torch.Tensor, lam_top: torch.Tensor, lam_bot: torch.Tensor
    ) -> torch.Tensor:
        """tokens: [B, H, N, D]; pos: [B, N, 3] centered coordinates; anchors: [B, N]."""
        D = tokens.shape[-1]
        P = self.P
        assert D >= 6 * P, f"head_dim {D} < 6 * pairs_per_axis {6 * P}"
        inv = (2.0 * torch.pi) / self.wavelengths(lam_top, lam_bot)  # [B, N, P]
        ang = pos.float().unsqueeze(-1) * inv.unsqueeze(-2)  # [B, N, 3, P]
        ang = torch.cat((ang, ang), dim=-1)
        cos = ang.cos().to(tokens.dtype)[:, None]  # [B, 1, N, 3, 2P]
        sin = ang.sin().to(tokens.dtype)[:, None]
        out = tokens.clone()
        for a in range(3):
            sl = slice(2 * P * a, 2 * P * (a + 1))
            t = tokens[..., sl]
            out[..., sl] = t * cos[..., a, :] + _rotate_half(t) * sin[..., a, :]
        return out


class Rope3DAdapter(nn.Module):
    """Adapts :class:`Rope3D` to the ``rope(q, pos)`` call of the attention layer.

    The per-step wavelength anchors travel inside the position tensor, packed as
    ``[x, y, z, lam_top, lam_bot]`` per token (see :meth:`pack_pos`).
    """

    def __init__(self, pairs_per_axis: int = 10):
        super().__init__()
        self.rope = Rope3D(pairs_per_axis)

    @staticmethod
    def pack_pos(pos: torch.Tensor, lam_top: torch.Tensor, lam_bot: torch.Tensor) -> torch.Tensor:
        """[M, 3] centered coordinates + scalar anchors -> [1, M, 5]."""
        m = pos.shape[0]
        return torch.cat(
            [pos, lam_top.reshape(1, 1).expand(m, 1), lam_bot.reshape(1, 1).expand(m, 1)],
            dim=-1,
        ).unsqueeze(0)

    def forward(self, tokens: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        assert pos.shape[-1] == 5
        return self.rope(tokens, pos[..., :3], pos[..., 3], pos[..., 4])
