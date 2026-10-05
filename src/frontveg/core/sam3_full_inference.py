"""Single-pass SAM3 inference, without tiling or graph merging.

Ablation baselines A and C: the full image goes through SAM3 in one forward pass.
"""

from __future__ import annotations

import numpy as np

from frontveg.core.tiling import is_hallucination, release_cache


class SAM3FullImageInference:
    """Run SAM3 on the full image in a single forward pass.

    Parameters
    ----------
    predictor : SAM3Predictor
        Loaded predictor.
    max_area_ratio : float
        Segments covering more than this fraction of the image are dropped.
    min_area : int
        Segments smaller than this (px) are dropped.
    """

    def __init__(self, predictor, max_area_ratio: float = 0.55, min_area: int = 50):
        self.predictor = predictor
        self.max_area_ratio = max_area_ratio
        self.min_area = min_area

    def process(self, pil_image, prompt: str):
        """Return ``(id_map, n_instances)``, or ``(None, 0)`` if nothing was detected."""
        width, height = pil_image.size
        masks, scores = self.predictor.predict_crop(pil_image, prompt)
        release_cache(self.predictor)
        if masks is None:
            return None, 0

        # Winner-takes-all: each pixel keeps its highest-score mask.
        id_map = np.zeros((height, width), dtype=np.uint16)
        best_score = np.zeros((height, width), dtype=np.float32)
        inst_id = 0
        for raw, score in zip(masks, scores):
            mask = raw[0] > 0
            if is_hallucination(mask, width * height, self.min_area, self.max_area_ratio):
                continue
            inst_id += 1
            update = mask & (float(score) > best_score)
            best_score[update] = score
            id_map[update] = inst_id
        return id_map, inst_id
