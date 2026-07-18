"""
Single-pass SAM3 inference (NO tiling, NO graph-merging).

Used by ablation baselines A and C where the full image is passed
directly to SAM3 without any tiling strategy. The result is a raw
instance ID map produced by a single forward pass over the entire image.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


class SAM3FullImageInference:
    """
    Runs SAM3 on the full image in a single forward pass.

    Parameters
    ----------
    predictor : SAM3Predictor
        Loaded SAM3Predictor wrapper (frontveg.models.sam3_wrapper).
    max_area_ratio : float
        Segments covering more than this fraction of the image are
        treated as hallucinations and discarded.
    """

    def __init__(self, predictor, max_area_ratio: float = 0.55):
        self.predictor = predictor
        self.max_area_ratio = max_area_ratio

    def _is_hallucination(self, mask: np.ndarray, W: int, H: int) -> bool:
        area = mask.sum()
        if area < 50:
            return True
        if area / (W * H) > self.max_area_ratio:
            return True
        return False

    def process(self, pil_image: Image.Image, prompt: str):
        """
        Run a single SAM3 forward pass on the full image.

        Returns
        -------
        id_map : np.ndarray (H, W) uint16  –  0 = background
        n_instances : int
        """
        W, H = pil_image.size
        masks, scores = self.predictor.predict_crop(pil_image, prompt)

        if masks is None:
            return None, 0

        # Build a winner-takes-all id_map (highest score wins per pixel)
        final_id_map = np.zeros((H, W), dtype=np.uint16)
        max_score_map = np.zeros((H, W), dtype=np.float32)

        inst_id = 0
        for i, m_raw in enumerate(masks):
            m_bool = m_raw[0] > 0
            if self._is_hallucination(m_bool, W, H):
                continue
            inst_id += 1
            score = float(scores[i])
            update = m_bool & (score > max_score_map)
            max_score_map[update] = score
            final_id_map[update] = inst_id

        torch.cuda.empty_cache()
        return final_id_map, inst_id
