# Copyright 2026 The S2Tok Authors.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International
# License (CC BY-NC 4.0). See the LICENSE file in the root of this repository or
# https://creativecommons.org/licenses/by-nc/4.0/.

"""Streaming 3D Gaussian reconstruction with S2Tok.

Reads an image folder (frames in filename order) or a video, feeds the frames to the
model one at a time, saves the reconstruction, and plays back the reconstruction after
every frame in a viser viewer.

Usage:
    python demo.py --model_path MODEL_PATH --seq_path SEQ_PATH --output_dir OUT_DIR

Example:
    python demo.py --model_path checkpoints/s2tok_252.pt --seq_path path/to/video.mp4 \
        --frame_interval 10 --output_dir outputs/my_scene
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from s2tok import load_model

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streaming reconstruction with S2Tok.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/s2tok_252.pt",
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "--seq_path", type=str, required=True, help="Directory of frames or a video file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/demo",
        help="Where the results are written.",
    )
    parser.add_argument(
        "--frame_interval", type=int, default=1, help="Use every n-th frame of the sequence."
    )
    parser.add_argument("--max_frames", type=int, default=None, help="Stop after this many frames.")
    parser.add_argument(
        "--size",
        type=int,
        default=252,
        help="Model resolution (square, multiple of 14; trained at 252).",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "bf16"],
        default="fp32",
        help="bf16 is faster; results differ slightly from fp32.",
    )
    parser.add_argument(
        "--save_all_steps",
        action="store_true",
        help="Also save the reconstruction after every frame.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--port", type=int, default=8080, help="Viewer port.")
    parser.add_argument("--no_viewer", action="store_true", help="Do not launch the viewer.")
    return parser.parse_args()


def read_sequence(path: Path, interval: int, max_frames: int | None) -> list[Image.Image]:
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        files = files[::interval][:max_frames]
        frames = [Image.open(p).convert("RGB") for p in files]
    else:
        import cv2

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"cannot open video {path}")
        frames, index = [], 0
        while max_frames is None or len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if index % interval == 0:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            index += 1
        cap.release()
    if not frames:
        raise FileNotFoundError(f"no frames found in {path}")
    return frames


def center_crop_resize(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left, top = (width - side) // 2, (height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    return image.resize((size, size), Image.Resampling.LANCZOS)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    frames = read_sequence(Path(args.seq_path), args.frame_interval, args.max_frames)
    frames = [np.array(center_crop_resize(f, args.size)) for f in frames]
    print(f"Loaded {len(frames)} frames from {args.seq_path}")

    model = load_model(args.model_path, args.device)
    memory = model.init_memory()
    reconstructions, records = [], []
    for i, frame in enumerate(frames):
        image = torch.from_numpy(frame).permute(2, 0, 1).float().div(255).to(args.device)
        out, memory = model.step(image, memory, precision=args.precision)
        reconstructions.append(out.gaussians.to("cpu"))
        records.append(
            {
                "frame": i,
                "log_scale": out.log_scale.item(),
                "admitted_patches": out.admitted.numel(),
                "scene_tokens": memory.num_tokens,
            }
        )
        if args.save_all_steps:
            reconstructions[-1].save_ply(output_dir / "steps" / f"step_{i:04d}.ply")
        print(
            f"[{i + 1}/{len(frames)}] admitted {out.admitted.numel():3d} patches, "
            f"{memory.num_tokens} scene tokens, {len(out.gaussians)} Gaussians",
            flush=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    reconstructions[-1].save_ply(output_dir / "scene.ply")
    (output_dir / "frames.json").write_text(
        json.dumps({"image_size": args.size, "frames": records}, indent=2)
    )
    print(f"Results saved to {output_dir}")

    if not args.no_viewer:
        from s2tok.viewer import launch_viewer

        launch_viewer(reconstructions, frames, port=args.port)


if __name__ == "__main__":
    main()
