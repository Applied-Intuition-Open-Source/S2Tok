# Copyright 2026 The S2Tok Authors.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International
# License (CC BY-NC 4.0). See the LICENSE file in the root of this repository or
# https://creativecommons.org/licenses/by-nc/4.0/.

"""Checkpoint loading and conversion."""

import dataclasses
import os
import pickle

import torch

from s2tok.model import S2Tok, S2TokConfig

# Parameters present in training checkpoints that the model does not use: the DA3
# camera-conditioning encoder, which S2Tok never feeds, and the camera-pose readout.
UNUSED_PREFIXES = ("backbone.cam_enc.", "cam_head.")
_WRAPPER_PREFIXES = ("module.", "_orig_mod.")


def clean_state_dict(state: dict) -> dict:
    """Strip DDP / torch.compile prefixes and drop parameters the model does not use."""
    out = {}
    for key, value in state.items():
        for prefix in _WRAPPER_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]
        if key.startswith(UNUSED_PREFIXES):
            continue
        if key in out:
            raise KeyError(f"duplicate parameter {key!r} after prefix stripping")
        out[key] = value
    return out


def load_state_dict_strict(model: S2Tok, state: dict) -> None:
    """Load ``state`` and fail unless it covers exactly the model's parameters."""
    state = clean_state_dict(state)
    expected = set(model.state_dict())
    missing = sorted(expected - set(state))
    unexpected = sorted(set(state) - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint does not match the model: {len(missing)} missing "
            f"(e.g. {missing[:5]}), {len(unexpected)} unexpected (e.g. {unexpected[:5]})"
        )
    model.load_state_dict(state, strict=True)


def save_checkpoint(path: str | os.PathLike, state: dict, config: S2TokConfig) -> None:
    torch.save({"model": state, "config": dataclasses.asdict(config)}, path)


def load_model(
    path: str | os.PathLike, device: str | torch.device = "cuda", config: S2TokConfig | None = None
) -> S2Tok:
    """Build S2Tok and load a checkpoint written by :func:`save_checkpoint`."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except pickle.UnpicklingError as e:
        raise RuntimeError(
            f"{path} is not an S2Tok checkpoint (it may be a raw training checkpoint); "
            "convert it with tools/export_checkpoint.py"
        ) from e
    if config is None:
        saved = ckpt.get("config", {})
        fields = {f.name for f in dataclasses.fields(S2TokConfig)}
        config = S2TokConfig(
            **{k: tuple(v) if isinstance(v, list) else v for k, v in saved.items() if k in fields}
        )
    # Build directly on the target device so the host holds only the memory-mapped file.
    with torch.device(device):
        model = S2Tok(config)
    load_state_dict_strict(model, ckpt["model"] if "model" in ckpt else ckpt)
    return model.eval()
