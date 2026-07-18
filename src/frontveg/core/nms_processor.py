"""
nms_processor.py
================
Post-hoc area-based cleanup applied AFTER the tiling+NMS step.

Role in the ablation
--------------------
In baselines B and D the primary duplicate-suppression is already done
by the greedy box-IoU NMS inside SAM3StandardTiledInference.
This module then removes any residual tiny or giant segments that
survived NMS (e.g. edge artefacts or background bleeds).

In baseline A and C (no tiling) this module is NOT used.

This is NOT the graph-based merging from sam3_tiler.py.
"""

import numpy as np


class NMSProcessor:
    """
    Post-hoc area filter on a SAM3 instance ID map.

    Parameters
    ----------
    min_area : int
        Instances with fewer pixels than this are removed.
    max_area_ratio : float
        Instances occupying more than this fraction of the full image
        are removed (likely background / hallucinations).
    """

    def __init__(self, min_area: int = 200, max_area_ratio: float = 0.55):
        self.min_area        = min_area
        self.max_area_ratio  = max_area_ratio

    def suppress(self, id_map: np.ndarray) -> np.ndarray:
        """
        Remove small / giant instances from *id_map*.

        Parameters
        ----------
        id_map : np.ndarray  (H, W) uint16  –  0 = background

        Returns
        -------
        np.ndarray  (H, W) uint16  – cleaned id_map
        """
        if id_map is None:
            return id_map

        H, W       = id_map.shape
        total_px   = H * W
        cleaned    = id_map.copy()

        for inst_id in np.unique(id_map):
            if inst_id == 0:
                continue
            area  = int(np.sum(id_map == inst_id))
            ratio = area / total_px
            if area < self.min_area or ratio > self.max_area_ratio:
                cleaned[cleaned == inst_id] = 0

        return cleaned.astype(np.uint16)
