# Copyright 2026 The S2Tok Authors.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International
# License (CC BY-NC 4.0). See the LICENSE file in the root of this repository or
# https://creativecommons.org/licenses/by-nc/4.0/.

"""Admission head: decides which patches of the incoming frame enter the scene memory.

Every candidate patch token of the new frame attends to the current scene memory
(cross-attention). The maximum attention weight and the normalized attention entropy
are fused back into the candidate feature, the candidates then attend to each other
(self-attention), and an MLP outputs one admission logit per patch.

Inputs are the backbone's outputs at its last block for the current step:
    frame_tokens  [T, D]  candidate patch tokens of the new frame
    scene_tokens  [M, D]  scene memory after being updated with the new frame
"""

import torch
from torch import nn


class AdmissionHead(nn.Module):
    def __init__(self, in_dim: int = 1536, dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.proj_q = nn.Linear(in_dim, dim)
        self.ln_q = nn.LayerNorm(dim)
        self.proj_kv = nn.Linear(in_dim, dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln_x = nn.LayerNorm(dim)
        self.fuse = nn.Linear(dim + 2, dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln_s = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, frame_tokens: torch.Tensor, scene_tokens: torch.Tensor) -> torch.Tensor:
        """frame_tokens [T, D], scene_tokens [M, D] with M >= 1 -> logits [T]."""
        assert scene_tokens.shape[0] > 0, "the admission head needs a non-empty scene memory"
        q = self.ln_q(self.proj_q(frame_tokens)).unsqueeze(0)  # [1, T, d]
        kv = self.ln_kv(self.proj_kv(scene_tokens)).unsqueeze(0)  # [1, M, d]
        attn_out, attn_w = self.cross(
            q, kv, kv, need_weights=True, average_attn_weights=True
        )  # [1, T, d], [1, T, M]
        w = attn_w.clamp_min(1e-12)
        max_attn = w.max(dim=-1).values
        log_m = max(torch.log(torch.tensor(float(w.shape[-1]))).item(), 1e-6)
        entropy = -(w * w.log()).sum(-1) / log_m
        x = self.ln_x(q + attn_out)
        x = self.fuse(torch.cat([x, max_attn.unsqueeze(-1), entropy.unsqueeze(-1)], dim=-1))
        s_out, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.ln_s(x + s_out)
        return self.mlp(x).squeeze(-1).squeeze(0)
