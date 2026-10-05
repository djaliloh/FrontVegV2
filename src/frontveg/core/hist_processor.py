"""Depth normalization and histogram computation."""

from __future__ import annotations

import numpy as np


class HistogramProcessor:
    def __init__(self, hist_size: int = 256):
        self.hist_size = hist_size

    @staticmethod
    def compute_global_max(depth_list) -> float:
        """Maximum depth over a batch, used as a common normalization reference."""
        return float(max((np.max(d) for d in depth_list), default=1.0))

    def normalize_and_hist(self, depth_map: np.ndarray, global_max: float):
        """Scale a depth map to [0, 255] with ``global_max`` and compute its histogram.

        Zero-depth pixels (background or invalid) are excluded from the histogram.

        Returns:
            tuple: ``(normalized_depth, histogram)``; uint8 (H, W) map and float32 counts.
        """
        global_max = max(float(global_max), 1e-8)
        normalized = np.clip(depth_map.astype(np.float32) / global_max * 255, 0, 255)
        clean_depth = normalized.astype(np.uint8)
        valid = clean_depth[depth_map > 0]
        hist = np.bincount(valid, minlength=self.hist_size)[: self.hist_size].astype(np.float32)
        return clean_depth, hist
