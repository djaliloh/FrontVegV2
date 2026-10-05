"""Standard sliding-window tiling with greedy box-IoU NMS.

Ablation baseline (B and D) for :mod:`frontveg.core.sam3_tiler`: fragments are
sorted by score, filtered with box-IoU NMS and stitched directly, without graph
merging or mask-IoU refinement, as in YOLO-Seg or Mask R-CNN pipelines.
"""

from __future__ import annotations

from frontveg.core.tiling import (
    collect_tile_fragments,
    count_instances,
    paint_winner_takes_all,
    stride_from_overlap,
)


class SAM3StandardTiledInference:
    """Standard tiling + greedy box-IoU NMS for SAM3.

    Parameters
    ----------
    predictor : SAM3Predictor
        Loaded predictor.
    tile_size : int
        Side of the square tiles, in pixels.
    overlap : float
        Overlap between adjacent tiles, in [0, 1).
    nms_iou_threshold : float
        Detections whose box IoU with a higher-score box exceeds this value are dropped.
    min_area : int
        Fragments smaller than this (px) are dropped before NMS.
    max_area_ratio : float
        Fragments covering more than this fraction of their tile are dropped before NMS.
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
        self.predictor = predictor
        self.tile_size = tile_size
        self.stride = stride_from_overlap(tile_size, overlap)
        self.nms_iou_thr = nms_iou_threshold
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio

    @staticmethod
    def _box_iou_single(b1: list, b2: list) -> float:
        """IoU of two ``[x1, y1, x2, y2]`` boxes."""
        inter_w = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
        inter_h = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
        inter = inter_w * inter_h
        if inter == 0:
            return 0.0
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / (area1 + area2 - inter + 1e-9)

    def _standard_nms(self, fragments: list) -> list:
        """Greedy NMS: keep the best remaining fragment, drop those overlapping it."""
        remaining = sorted(fragments, key=lambda f: f["score"], reverse=True)
        kept = []
        while remaining:
            best = remaining.pop(0)
            kept.append(best)
            remaining = [
                f
                for f in remaining
                if self._box_iou_single(best["bbox"], f["bbox"]) <= self.nms_iou_thr
            ]
        return kept

    def process(self, pil_image, prompt: str):
        """Return ``(id_map, n_instances)``, or ``(None, 0)`` if nothing was detected."""
        fragments = collect_tile_fragments(
            self.predictor,
            pil_image,
            prompt,
            self.tile_size,
            self.stride,
            self.min_area,
            self.max_area_ratio,
        )
        if not fragments:
            return None, 0

        kept = self._standard_nms(fragments)
        width, height = pil_image.size
        id_map = paint_winner_takes_all(kept, range(1, len(kept) + 1), height, width)
        return id_map, count_instances(id_map)
