# Copyright 2026 The S2Tok Authors.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International
# License (CC BY-NC 4.0). See the LICENSE file in the root of this repository or
# https://creativecommons.org/licenses/by-nc/4.0/.

"""Convert a training checkpoint into an S2Tok inference checkpoint.

Training checkpoints also hold the optimizer state and a pickled training config;
this keeps only the model parameters used at inference, checks that they match the
model exactly, and writes them in a format that loads with ``weights_only=True``.

Reading the training checkpoint unpickles its config, which requires ``omegaconf``.

Usage:
    python tools/export_checkpoint.py TRAINING_CKPT checkpoints/s2tok_252.pt
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s2tok.checkpoint import clean_state_dict, load_state_dict_strict, save_checkpoint  # noqa: E402
from s2tok.model import S2Tok, S2TokConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint", type=Path, help="training checkpoint")
    parser.add_argument("output", type=Path, help="output inference checkpoint (.pt)")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    state = clean_state_dict(ckpt.get("model", ckpt))
    config = S2TokConfig()

    # Verify the key set before writing anything.
    with torch.device("meta"):
        model = S2Tok(config)
    load_state_dict_strict(model.to_empty(device="cpu"), state)

    state = {k: v.detach().clone().contiguous() for k, v in state.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.output, state, config)
    n = sum(v.numel() for v in state.values())
    print(f"wrote {args.output}: {len(state)} tensors, {n / 1e6:.1f}M parameters")


if __name__ == "__main__":
    main()
