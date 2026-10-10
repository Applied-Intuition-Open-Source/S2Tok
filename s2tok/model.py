# Copyright 2026 The S2Tok Authors.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International
# License (CC BY-NC 4.0). See the LICENSE file in the root of this repository or
# https://creativecommons.org/licenses/by-nc/4.0/.
#
# Portions of this file are derived from ZipSplat (https://github.com/cvg/ZipSplat),
# Copyright 2026 ETH Zurich, licensed under the Apache License, Version 2.0, and were
# modified by The S2Tok Authors. See NOTICE and THIRD_PARTY_LICENSES.

"""S2Tok: streaming 3D Gaussian reconstruction with a token scene memory.

The model consumes a video one frame at a time. Its state is a set of scene tokens that
live in the residual space of the ViT backbone. For every new frame:

1. **Backbone with memory.** The frame is encoded by the DA3 ViT-G backbone. From block
   ``alt_start`` on, every global block runs full self-attention over
   ``[frame tokens ; scale token ; scene tokens]``, so the memory is read and rewritten
   by the frame. In parallel to every frame-local block, a scene-local twin block runs
   attention over ``[scale token ; scene tokens]`` with a continuous 3D RoPE on the
   tokens' Gaussian centroids from the previous step.
2. **Admission.** An admission head scores every patch token of the new frame against
   the updated memory; patches with probability above 0.5 are admitted (all patches of
   the first frame are).
3. **Decoding.** Multi-layer features of the scene tokens and the admitted patches are
   projected to the decoder width, refined by cross-/self-attention and concatenated
   with a color skip feature. The Gaussian head turns every token into 32 Gaussians,
   giving the reconstruction of everything seen so far.
4. **Recurrence.** The admitted patches' backbone tokens are appended to the updated
   memory for the next frame.

The scale token is read out into the log of the scene scale.
"""

import contextlib
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from s2tok.admission import AdmissionHead
from s2tok.backbone import DA3Encoder
from s2tok.gaussians import GaussianHead, Gaussians
from s2tok.modules.blocks import CrossAttentionBlock, SelfAttentionBlock, SwiGLUFFN
from s2tok.modules.patch_embed import PatchEmbed
from s2tok.modules.rope3d import (
    Rope3DAdapter,
    cloud_center_and_radius,
    token_centroids,
    wavelength_anchors,
)


@dataclass
class S2TokConfig:
    image_size: int = 252
    out_layers: tuple[int, ...] = (19, 29, 39)
    decoder_heads: int = 24
    decoder_mlp_ratio: float = 4.0
    decoder_layer_scale: float = 0.01
    color_dim: int = 128
    gaussians_per_token: int = 32
    sh_degree: int = 1
    readout_mlp_ratio: float = 0.25
    rope3d_pairs_per_axis: int = 10
    rope3d_rho_ref: float = 4.5
    admission_dim: int = 256
    admission_threshold: float = 0.5
    admission_eps: float = 0.02
    max_scene_tokens: int = 4000


@dataclass
class SceneMemory:
    """Recurrent state carried between frames."""

    tokens: torch.Tensor  # [M, D] backbone-space scene tokens
    colors: torch.Tensor  # [M, Dc] color-skip features of the scene tokens
    positions: torch.Tensor | None = None  # [M, 3] centered token centroids
    lam_top: torch.Tensor | None = None  # scene-local RoPE wavelength anchors
    lam_bot: torch.Tensor | None = None
    num_frames: int = 0

    @property
    def num_tokens(self) -> int:
        return self.tokens.shape[0]


@dataclass
class StepOutput:
    gaussians: Gaussians  # reconstruction of all frames seen so far
    admitted: torch.Tensor  # [K] indices of the admitted patches of this frame
    admission_prob: torch.Tensor | None  # [T] admission probabilities (None for frame 0)
    log_scale: torch.Tensor  # [] predicted log scene scale


def _vit_norm(dim: int) -> nn.LayerNorm:
    return nn.LayerNorm(dim, eps=1e-6)


def _readout_mlp(in_dim: int, out_dim: int, hidden_ratio: float) -> nn.Sequential:
    hidden = max(1, int(in_dim * hidden_ratio))
    return nn.Sequential(
        nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim)
    )


class S2Tok(nn.Module):
    def __init__(self, config: S2TokConfig | None = None):
        super().__init__()
        self.config = c = config or S2TokConfig()
        self.backbone = DA3Encoder()
        vit = self.vit
        D = self.embed_dim = vit.embed_dim
        self.patch_size = vit.patch_size
        L = len(c.out_layers)

        # Projection of the concatenated [local ; global] backbone features to the decoder
        # width, separately for frame tokens and scene tokens.
        self.pre_norm_local = nn.ModuleList([nn.LayerNorm(D) for _ in range(L)])
        self.pre_norm_global = nn.ModuleList([nn.LayerNorm(D) for _ in range(L)])
        self.downscale = nn.ModuleList([nn.Linear(2 * D, D) for _ in range(L)])
        self.scene_pre_norm_local = nn.ModuleList([nn.LayerNorm(D) for _ in range(L)])
        self.scene_pre_norm_global = nn.ModuleList([nn.LayerNorm(D) for _ in range(L)])
        self.scene_downscale = nn.ModuleList([nn.Linear(2 * D, D) for _ in range(L)])

        # Decoder (one cross-/self-attention pair per backbone feature layer).
        self.cross_attention = nn.ModuleList(
            [
                CrossAttentionBlock(
                    D, c.decoder_heads, c.decoder_mlp_ratio, c.decoder_layer_scale, qk_norm=True
                )
                for _ in range(L)
            ]
        )
        self.self_attention = nn.ModuleList(
            [
                SelfAttentionBlock(
                    D, c.decoder_heads, c.decoder_mlp_ratio, c.decoder_layer_scale, qk_norm=True
                )
                for _ in range(L)
            ]
        )

        # Color skip: a patch embedding of the raw frame, attended by the admitted patches.
        self.color_embed = PatchEmbed(patch_size=self.patch_size, embed_dim=c.color_dim)
        self.color_cross_attention = CrossAttentionBlock(
            c.color_dim, max(1, c.color_dim // 64), 1.0, c.decoder_layer_scale, qk_norm=True
        )
        self.gaussian_head = GaussianHead(D + c.color_dim, c.gaussians_per_token, c.sh_degree)

        # Scale token (joins the global and scene-local attention) and its readout.
        self.scale_token = nn.Parameter(torch.zeros(1, 1, D))
        self.scale_head = _readout_mlp(D, 1, c.readout_mlp_ratio)

        # Scene-local twin of every frame-local block in the alternating range.
        self.scene_local_ids = [i for i in range(vit.n_blocks) if i >= vit.alt_start and i % 2 == 0]
        self._scene_local_index = {i: n for n, i in enumerate(self.scene_local_ids)}
        self.rope3d = Rope3DAdapter(c.rope3d_pairs_per_axis)
        self.scene_local_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=D,
                    num_heads=vit.blocks[i].attn.num_heads,
                    init_values=1.0,
                    norm_layer=_vit_norm,
                    ffn_layer=SwiGLUFFN,
                    qk_norm=True,
                    rope=self.rope3d,
                )
                for i in self.scene_local_ids
            ]
        )

        self.admission_head = AdmissionHead(D, c.admission_dim)

    @property
    def vit(self):
        return self.backbone.backbone

    @property
    def device(self) -> torch.device:
        return self.scale_token.device

    def init_memory(self) -> SceneMemory:
        return SceneMemory(
            tokens=torch.zeros(0, self.embed_dim, device=self.device),
            colors=torch.zeros(0, self.config.color_dim, device=self.device),
        )

    # ------------------------------------------------------------------
    # Backbone with the scene memory threaded through
    # ------------------------------------------------------------------
    def _global_block(self, blk, x, scale, scene, pos_global):
        """Full self-attention over [frame ; scale ; scene]; returns the updated parts."""
        n_f, M = x.shape[1], scene.shape[1]
        parts = [x, scale] + ([scene] if M > 0 else [])
        seq = torch.cat(parts, dim=1)
        # Scale and scene tokens take the same constant position as the frame patches.
        extra = torch.ones(1, 1 + M, 2, device=seq.device, dtype=pos_global.dtype)
        seq = blk(seq, pos=torch.cat([pos_global, extra], dim=1))
        return seq[:, :n_f], seq[:, n_f : n_f + 1], seq[:, n_f + 1 :]

    def _encode(self, image: torch.Tensor, memory: SceneMemory) -> dict:
        vit = self.vit
        D = self.embed_dim
        H, W = image.shape[-2:]
        out_layers = set(self.config.out_layers)

        x = vit.prepare_tokens(image)  # [1, 1 + T, D]
        pos_local, pos_global = vit.prepare_rope(1, H, W, image.device)
        scene = memory.tokens.unsqueeze(0)  # [1, M, D]
        M = scene.shape[1]
        scale = None

        scene_pos = None
        if M > 0:
            # Row 0 is the scale token, which is not rotated.
            pos = torch.cat([memory.positions.new_zeros(1, 3), memory.positions], dim=0)
            scene_pos = Rope3DAdapter.pack_pos(pos, memory.lam_top, memory.lam_bot)

        local_x = scene_local = None
        frame_taps, scene_taps = [], []
        for i, blk in enumerate(vit.blocks):
            if i == vit.alt_start:
                token = 0 if memory.num_frames == 0 else 1  # reference / source view
                x[:, 0] = vit.camera_token[:, token]
                scale = self.scale_token

            if vit.is_global(i):
                x, scale, scene = self._global_block(blk, x, scale, scene, pos_global)
            else:
                x = blk(x, pos=pos_local if i >= vit.alt_start else None)
                local_x = x
                if i >= vit.alt_start and M > 0:
                    sl_blk = self.scene_local_blocks[self._scene_local_index[i]]
                    seq = sl_blk(torch.cat([scale, scene], dim=1), pos=scene_pos)
                    scale, scene = seq[:, :1], seq[:, 1:]
                    scene_local = scene

            if i in out_layers:
                frame_taps.append(torch.cat([local_x, x], dim=-1))
                # With an empty memory there is no scene-local pass; the (empty) scene
                # tokens stand in for their local half.
                scene_taps.append((scene_local if scene_local is not None else scene, scene))
                if i == max(out_layers):
                    frame_last = x

        # [local ; norm(global)] features without the cls/camera slot, deepest first.
        frame_feats = [
            torch.cat([t[..., :D], vit.norm(t[..., D:])], dim=-1)[:, 1:] for t in frame_taps
        ][::-1]
        scene_feats = [torch.cat([loc, vit.norm(glob)], dim=-1) for loc, glob in scene_taps][::-1]
        return {
            "frame_feats": frame_feats,  # each [1, T, 2D]
            "scene_feats": scene_feats,  # each [1, M, 2D]
            "scene_tokens": scene[0],  # [M, D] updated memory
            "frame_tokens": frame_last[0, 1:],  # [T, D]
            "scale_token": scale[:, 0],  # [1, D]
        }

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------
    @staticmethod
    def _project(feats, pre_norm_local, pre_norm_global, downscale):
        D = feats[0].shape[-1] // 2
        out = []
        for layer, raw in enumerate(feats):
            local = pre_norm_local[layer](raw[..., :D])
            global_ = pre_norm_global[layer](raw[..., D:])
            out.append(
                downscale[layer](torch.cat([local, global_], dim=-1)) + (local + global_) / 2
            )
        return out

    def _decode(self, scene_feats, frame_feats, admitted):
        """Decoder over [scene tokens ; admitted patches]; returns (scene, patches)."""
        scene = self._project(
            scene_feats, self.scene_pre_norm_local, self.scene_pre_norm_global, self.scene_downscale
        )
        frame = self._project(
            frame_feats, self.pre_norm_local, self.pre_norm_global, self.downscale
        )
        idx = admitted.view(1, -1, 1).expand(1, -1, self.embed_dim)
        patches = [torch.gather(f, 1, idx) for f in frame]
        M = scene[0].shape[1]
        r = torch.cat([scene[0], patches[0]], dim=1)
        for layer in range(len(scene)):
            kv = torch.cat([scene[layer], patches[layer]], dim=1)
            r = r + self.cross_attention[layer](r, kv, return_delta=True)
            r = self.self_attention[layer](r)
        return r[:, :M], r[:, M:]

    def _color_features(self, image, admitted):
        color = self.color_embed(image)  # [1, T, Dc]
        idx = admitted.view(1, -1, 1).expand(1, -1, color.shape[-1])
        return self.color_cross_attention(torch.gather(color, 1, idx), color)  # [1, K, Dc]

    def _admit(self, frame_tokens, scene_tokens, num_scene_tokens):
        c = self.config
        logits = self.admission_head(frame_tokens.float(), scene_tokens.float())
        prob = torch.sigmoid(logits).clamp(c.admission_eps, 1.0 - c.admission_eps)
        admitted = (prob > c.admission_threshold).nonzero(as_tuple=True)[0]
        # Keep the memory within max_scene_tokens: admit the most probable patches first.
        room = max(c.max_scene_tokens - num_scene_tokens, 0)
        if admitted.numel() > room:
            admitted = admitted[prob[admitted].argsort(descending=True)[:room]]
        return admitted.long(), prob

    def _next_positions(self, gaussians: Gaussians):
        """Scene-local RoPE state for the next step from this step's Gaussians."""
        G = self.config.gaussians_per_token
        centroids, mass = token_centroids(
            gaussians.means.reshape(-1, G, 3).float(), gaussians.opacities.reshape(-1, G).float()
        )
        center, radius = cloud_center_and_radius(centroids, mass)
        lam_top, lam_bot = wavelength_anchors(radius, self.config.rope3d_rho_ref)
        return centroids - center, lam_top, lam_bot

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(
        self, image: torch.Tensor, memory: SceneMemory, precision: str = "fp32"
    ) -> tuple[StepOutput, SceneMemory]:
        """Consume one frame and return the reconstruction so far plus the new memory.

        Args:
            image: [1, 3, H, W] or [3, H, W] RGB in [0, 1]; H and W multiples of 14
                (the model was trained at 252 x 252).
            memory: state from the previous step (``init_memory()`` for the first frame).
            precision: "fp32", or "bf16" (bf16 autocast with FlashAttention in the backbone).
        """
        if image.dim() == 3:
            image = image[None]
        assert image.shape[0] == 1, "step() takes one frame"
        if precision not in ("fp32", "bf16"):
            raise ValueError(f"unknown precision {precision!r}")
        bf16 = precision == "bf16"
        M = memory.num_tokens

        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else contextlib.nullcontext()
        )
        with autocast:
            flash = sdpa_kernel(SDPBackend.FLASH_ATTENTION) if bf16 else contextlib.nullcontext()
            with flash:
                enc = self._encode(image, memory)

            if memory.num_frames == 0:
                T = enc["frame_tokens"].shape[0]
                admitted, prob = torch.arange(T, device=image.device), None
            else:
                admitted, prob = self._admit(enc["frame_tokens"], enc["scene_tokens"], M)

            scene_dec, patch_dec = self._decode(enc["scene_feats"], enc["frame_feats"], admitted)
            patch_colors = self._color_features(image, admitted)
            tokens = torch.cat(
                [
                    torch.cat([scene_dec, memory.colors.unsqueeze(0)], dim=-1),
                    torch.cat([patch_dec, patch_colors], dim=-1),
                ],
                dim=1,
            )
            with torch.autocast("cuda", enabled=False):
                gaussians = self.gaussian_head(tokens[0].float())
            positions, lam_top, lam_bot = self._next_positions(gaussians)

            new_patches = torch.gather(
                enc["frame_tokens"], 0, admitted.view(-1, 1).expand(-1, self.embed_dim)
            )
            next_memory = SceneMemory(
                tokens=torch.cat([enc["scene_tokens"], new_patches], dim=0),
                colors=torch.cat([memory.colors, patch_colors[0]], dim=0),
                positions=positions,
                lam_top=lam_top,
                lam_bot=lam_bot,
                num_frames=memory.num_frames + 1,
            )
            log_scale = self.scale_head(enc["scale_token"])[0, 0]

        out = StepOutput(
            gaussians=gaussians, admitted=admitted, admission_prob=prob, log_scale=log_scale.float()
        )
        return out, next_memory
