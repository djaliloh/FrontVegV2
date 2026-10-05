"""Binary segmentation of normalized depth maps."""

from __future__ import annotations

import cv2
import numpy as np


class BinarySegmenter:
    @staticmethod
    def apply_threshold(depth_norm: np.ndarray, threshold: float) -> np.ndarray:
        """Binary mask (0 or 255) of the pixels strictly above ``threshold``."""
        _, binary_mask = cv2.threshold(depth_norm, threshold, 255, cv2.THRESH_BINARY)
        return binary_mask
