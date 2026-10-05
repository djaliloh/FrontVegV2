"""Automatic foreground threshold from the valleys of a depth histogram."""

from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


class ValleyProcessor:
    """Find the depth threshold separating the foreground row from the background.

    Peaks of the (optionally smoothed) histogram are the depth modes. Valleys are
    searched between consecutive peaks, from the second peak onwards, and the
    lowest one gives the threshold.
    """

    def __init__(
        self,
        sigma: float = 1.1,
        peak_dist: float = 8.0,
        peak_height: float = 0.5,
        smooth: bool = True,
        prominence: float = 0.5,
        max_valleys: int = 4,
    ):
        self.sigma = sigma
        self.peak_dist = peak_dist
        self.peak_height = peak_height
        self.smooth = smooth
        self.prominence = prominence
        self.max_valleys = max_valleys

    def find_optimal_threshold(self, histogram, sigma=None, peak_dist=None) -> dict | None:
        """Analyze ``histogram``; ``sigma`` and ``peak_dist`` override the instance values.

        Returns:
            dict with ``optimal_threshold``, ``peaks``, ``valleys`` and ``smoothed_hist``,
            or None when fewer than three peaks are found.
        """
        sigma = self.sigma if sigma is None else sigma
        peak_dist = self.peak_dist if peak_dist is None else peak_dist

        y = np.array(histogram, copy=True)
        if self.smooth:
            y = gaussian_filter(y, sigma=sigma)

        peaks, _ = find_peaks(
            y, height=self.peak_height, prominence=self.prominence, distance=peak_dist
        )
        if len(peaks) < 3:
            logger.debug("Only %d histogram peak(s) found, no threshold", len(peaks))
            return None

        valleys = []
        for left, right in zip(peaks[1:-1], peaks[2:]):
            index = int(np.argmin(y[left:right]) + left)
            valleys.append(
                {
                    "index": index,
                    # Valley depth relative to the mean height of its two peaks.
                    "depth": float(y[index] - (y[left] + y[right]) / 2),
                    "y_value": float(y[index]),
                }
            )
        valleys = valleys[: self.max_valleys]

        optimal = min(valleys, key=lambda v: v["y_value"])
        return {
            "optimal_threshold": optimal["index"],
            "peaks": peaks,
            "valleys": valleys,
            "smoothed_hist": y,
        }
