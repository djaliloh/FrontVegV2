"""Area-based cleanup of an instance map, applied after tiling + NMS.

Used by ablation baselines B and D to remove tiny or giant segments that
survived NMS (edge artefacts, background bleeds). Not used by baselines A and C,
nor by the graph-based merging of :mod:`frontveg.core.sam3_tiler`.
"""

from __future__ import annotations

import numpy as np


class NMSProcessor:
    """Remove instances that are too small or cover too much of the image.

    Parameters
    ----------
    min_area : int
        Instances smaller than this (px) are removed.
    max_area_ratio : float
        Instances covering more than this fraction of the image are removed.
    """

    def __init__(self, min_area: int = 200, max_area_ratio: float = 0.55):
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio

    def suppress(self, id_map: np.ndarray | None) -> np.ndarray | None:
        """Return a cleaned uint16 copy of ``id_map`` (0 = background)."""
        if id_map is None:
            return None

        areas = np.bincount(id_map.ravel())
        ratios = areas / id_map.size
        drop = (areas < self.min_area) | (ratios > self.max_area_ratio)
        drop[0] = False  # Background is not an instance
        cleaned = np.where(drop[id_map], 0, id_map)
        return cleaned.astype(np.uint16)
