# Copyright 2026 The S2Tok Authors.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International
# License (CC BY-NC 4.0). See the LICENSE file in the root of this repository or
# https://creativecommons.org/licenses/by-nc/4.0/.

"""Interactive viewer for streaming reconstructions (requires ``viser``).

Plays back the reconstruction after every frame next to the corresponding input frame.
"""

import time

import numpy as np
import torch

from s2tok.gaussians import Gaussians

SH_C0 = 0.28209479177387814


def _to_numpy(g: Gaussians) -> dict:
    """Centers, covariances, colors (SH band 0) and opacities for viser."""
    q = g.quats.float()
    w, x, y, z = q.unbind(-1)
    R = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        -1,
    ).reshape(-1, 3, 3)
    RS = R * g.scales.float()[:, None, :]
    return {
        "centers": g.means.float().cpu().numpy(),
        "covariances": (RS @ RS.transpose(1, 2)).cpu().numpy(),
        "rgbs": (0.5 + SH_C0 * g.sh[:, 0].float()).clamp(0, 1).cpu().numpy(),
        "opacities": g.opacities.float()[:, None].cpu().numpy(),
    }


def launch_viewer(
    reconstructions: list[Gaussians], images: list[np.ndarray], port: int = 8080
) -> None:
    """Serve the playback on ``port`` until interrupted.

    Args:
        reconstructions: Gaussians after every frame.
        images: model-resolution input frames, uint8 [H, W, 3], shown next to the playback.
    """
    import viser

    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("-y")  # OpenCV camera convention
    num_frames = len(reconstructions)
    # Typical depth of the first frame (about 1 by construction of the training normalization).
    scene_scale = float(reconstructions[0].means[:, 2].float().median())

    with server.gui.add_folder("Playback"):
        gui_frame = server.gui.add_slider(
            "Frame", min=0, max=num_frames - 1, step=1, initial_value=num_frames - 1
        )
        gui_prev = server.gui.add_button("Previous frame")
        gui_next = server.gui.add_button("Next frame")
        gui_playing = server.gui.add_checkbox("Playing", False)
        gui_fps = server.gui.add_slider("FPS", min=1, max=30, step=1, initial_value=10)
        gui_image = server.gui.add_image(images[-1], label="Input frame")

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        # Start slightly behind and above the first camera (y points down).
        client.camera.position = (0.0, -0.3 * scene_scale, -0.6 * scene_scale)
        client.camera.look_at = (0.0, 0.0, scene_scale)

    splats = {"handle": None}

    def show(t: int) -> None:
        if splats["handle"] is not None:
            splats["handle"].remove()
        splats["handle"] = server.scene.add_gaussian_splats(
            "/reconstruction", **_to_numpy(reconstructions[t])
        )
        gui_image.image = images[t]

    @gui_frame.on_update
    def _(_) -> None:
        show(int(gui_frame.value))

    @gui_prev.on_click
    def _(_) -> None:
        gui_frame.value = (int(gui_frame.value) - 1) % num_frames

    @gui_next.on_click
    def _(_) -> None:
        gui_frame.value = (int(gui_frame.value) + 1) % num_frames

    show(num_frames - 1)
    print(f"Viewer running at http://localhost:{port} (Ctrl+C to exit)", flush=True)
    try:
        while True:
            if gui_playing.value:
                gui_frame.value = (int(gui_frame.value) + 1) % num_frames
            time.sleep(1.0 / gui_fps.value)
    except KeyboardInterrupt:
        server.stop()
