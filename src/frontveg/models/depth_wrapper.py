"""Relative depth estimation with Depth-Anything-V2."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from frontveg.resources import add_to_sys_path

logger = logging.getLogger(__name__)

# Architectures published with Depth-Anything-V2.
ENCODER_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


class DepthAnythingWrapper:
    def __init__(
        self,
        encoder: str = "vitl",
        device: str | None = None,
        checkpoint_path: str | Path = "checkpoints/depthanything_ckpts",
        repo_path: str | Path = "external/Depth-Anything-V2",
    ):
        if encoder not in ENCODER_CONFIGS:
            raise ValueError(
                f"Unknown encoder {encoder!r}, expected one of {list(ENCODER_CONFIGS)}"
            )
        if repo_path is None or not Path(repo_path).is_dir():
            raise FileNotFoundError(
                f"Depth-Anything-V2 repository not found at '{repo_path}'. "
                "Clone it or set FRONTVEG_DEPTH_REPO."
            )
        self.ckpt_path = Path(checkpoint_path) / f"depth_anything_v2_{encoder}.pth"
        if not self.ckpt_path.is_file():
            raise FileNotFoundError(
                f"Depth-Anything-V2 checkpoint not found at '{self.ckpt_path}'. "
                "Download it or set FRONTVEG_DEPTH_CKPT_DIR."
            )

        add_to_sys_path(repo_path)
        from depth_anything_v2.dpt import DepthAnythingV2

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DepthAnythingV2(**ENCODER_CONFIGS[encoder])
        self.model.load_state_dict(
            torch.load(self.ckpt_path, map_location="cpu", weights_only=True)
        )
        self.model.to(self.device).eval()
        logger.info("Depth-Anything-V2 (%s) loaded on %s", encoder, self.device)

    def infer(self, rgb_image: np.ndarray, input_size: int = 518) -> np.ndarray:
        """Relative depth of an RGB uint8 image, min-max normalized to a [0, 255] uint8 map."""
        # Depth-Anything-V2 expects OpenCV BGR input.
        bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        depth = self.model.infer_image(bgr, input_size)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255.0
        return depth.astype(np.uint8)
