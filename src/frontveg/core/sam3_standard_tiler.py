"""
sam3_standard_tiler.py
======================
Standard sliding-window tiling + conventional box-IoU NMS.

This is the ABLATION baseline tiler (Baselines B and D).
It is intentionally different from `sam3_tiler.py`:

  sam3_tiler.py  (your method)
      → fragments → connected-components graph merge → winner-takes-all

  sam3_standard_tiler.py  (ablation baseline)
      → fragments → sort by score → greedy box-IoU NMS → direct stitch

No graph, no connected components, no hybrid mask-IoU refinement.
This reflects a standard detection-style tiling pipeline as used
in YOLO-Seg, Mask-RCNN + NMS, etc.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


class SAM3StandardTiledInference:
    """
    Standard tiling + greedy box-IoU NMS for SAM3.

    Parameters
    ----------
    predictor : SAM3Predictor
        Loaded SAM3Predictor wrapper.
    tile_size : int
        Side length of each square tile in pixels.
    overlap : float
        Fraction of overlap between adjacent tiles (e.g. 0.5 = 50 %).
    nms_iou_threshold : float
        Box-IoU threshold for standard NMS suppression.
        Detections whose IoU with a higher-scoring box exceeds this
        value are discarded. Typical values: 0.3 – 0.5.
    min_area : int
        Segments with fewer pixels than this are discarded before NMS.
    max_area_ratio : float
        Segments covering more than this fraction of their tile are
        treated as hallucinations and discarded before NMS.
    """

    def __init__(
        self,
        predictor,
        tile_size: int = 640,
        overlap: float = 0.50,
        nms_iou_threshold: float = 0.30,
        min_area: int = 50,
        max_area_ratio: float = 0.55,
    ):
        self.predictor       = predictor
        self.tile_size       = tile_size
        self.stride          = int(tile_size * (1 - overlap))
        self.nms_iou_thr     = nms_iou_threshold
        self.min_area        = min_area
        self.max_area_ratio  = max_area_ratio

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_hallucination(self, mask: np.ndarray, w_t: int, h_t: int) -> bool:
        area = mask.sum()
        if area < self.min_area:
            return True
        if area / (w_t * h_t) > self.max_area_ratio:
            return True
        return False

    @staticmethod
    def _box_iou_single(b1: list, b2: list) -> float:
        """Compute IoU between two boxes [x1, y1, x2, y2]."""
        ix1 = max(b1[0], b2[0])
        iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2])
        iy2 = min(b1[3], b2[3])
        inter_w = max(0, ix2 - ix1)
        inter_h = max(0, iy2 - iy1)
        inter   = inter_w * inter_h
        if inter == 0:
            return 0.0
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / (area1 + area2 - inter + 1e-9)

    def _standard_nms(self, fragments: list) -> list:
        """
        Greedy box-IoU NMS (standard approach).

        1. Sort all detections by score descending.
        2. Greedily keep the top detection.
        3. Suppress any remaining detection whose box IoU with the
           kept box exceeds `nms_iou_threshold`.
        4. Repeat until the list is empty.

        Returns
        -------
        list of kept fragment dicts.
        """
        if not fragments:
            return []

        # Sort by score descending
        sorted_frags = sorted(fragments, key=lambda f: float(f["score"]), reverse=True)

        kept   = []
        active = list(range(len(sorted_frags)))

        while active:
            # Always keep the highest-score remaining detection
            best_idx = active.pop(0)
            best     = sorted_frags[best_idx]
            kept.append(best)

            # Suppress detections with high box-IoU with the kept one
            surviving = []
            for idx in active:
                iou = self._box_iou_single(best["bbox"], sorted_frags[idx]["bbox"])
                if iou <= self.nms_iou_thr:
                    surviving.append(idx)
            active = surviving

        return kept

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, pil_image: Image.Image, prompt: str):
        """
        Run standard tiling → NMS on *pil_image*.

        Returns
        -------
        id_map : np.ndarray (H, W) uint16  –  0 = background
        n_instances : int
        """
        W, H = pil_image.size
        fragments = []

        # ── Step 1: Sliding-window tiling ─────────────────────────────
        for y in range(0, H, self.stride):
            for x in range(0, W, self.stride):
                x_end, y_end = min(x + self.tile_size, W), min(y + self.tile_size, H)
                w_t,   h_t   = x_end - x, y_end - y

                crop             = pil_image.crop((x, y, x_end, y_end))
                masks, scores    = self.predictor.predict_crop(crop, prompt)

                if masks is not None:
                    for i, m_raw in enumerate(masks):
                        m_bool = m_raw[0] > 0
                        if self._is_hallucination(m_bool, w_t, h_t):
                            continue

                        ys, xs = np.where(m_bool)
                        if len(ys) == 0:
                            continue

                        # Convert local bbox → global image coordinates
                        ly1, lx1, ly2, lx2 = ys.min(), xs.min(), ys.max(), xs.max()
                        gx1, gy1 = lx1 + x, ly1 + y
                        gx2, gy2 = lx2 + x, ly2 + y

                        fragments.append({
                            "bbox":  [gx1, gy1, gx2, gy2],   # global coords
                            "mask":  m_bool[ly1:ly2+1, lx1:lx2+1],  # local crop
                            "score": float(scores[i]),
                        })

                torch.cuda.empty_cache()

        if not fragments:
            return None, 0

        # ── Step 2: Standard box-IoU NMS ──────────────────────────────
        kept = self._standard_nms(fragments)

        # ── Step 3: Stitch surviving masks into the final id_map ───────
        # Simple sequential assignment – highest-score fragment painted
        # first; later (lower-score) fragments do not overwrite.
        final_id_map    = np.zeros((H, W), dtype=np.uint16)
        max_score_map   = np.zeros((H, W), dtype=np.float32)

        for inst_id, frag in enumerate(kept, start=1):
            x1, y1, x2, y2 = frag["bbox"]
            score           = frag["score"]
            patch           = frag["mask"]

            h_p, w_p = patch.shape
            y2c  = min(y1 + h_p, H)
            x2c  = min(x1 + w_p, W)
            ph   = y2c - y1
            pw   = x2c - x1

            local_mask = patch[:ph, :pw]
            roi_scores = max_score_map[y1:y2c, x1:x2c]

            # Only paint pixels that belong to this mask AND have no
            # higher-score fragment already written there.
            update = local_mask & (score > roi_scores)
            max_score_map[y1:y2c, x1:x2c][update] = score
            final_id_map[y1:y2c, x1:x2c][update]  = inst_id

        n_instances = int(final_id_map.max())
        return final_id_map, n_instances
