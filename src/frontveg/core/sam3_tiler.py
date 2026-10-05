"""FrontVeg V2 tiled SAM3 inference.

Fragments of the same object cut by tile borders are grouped with a connectivity
graph (box IoU pre-filter, then mask IoU), then stitched with a per-pixel
winner-takes-all on the SAM3 score.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from frontveg.core.tiling import (
    box_iou_matrix,
    collect_tile_fragments,
    count_instances,
    paint_winner_takes_all,
    stride_from_overlap,
)


class SAM3TiledInference:
    def __init__(
        self,
        predictor,
        tile_size: int = 640,
        overlap: float = 0.5,
        max_tile_ratio: float = 0.55,
        min_area: int = 50,
        box_iou_threshold: float = 0.05,
        mask_iou_threshold: float = 0.10,
    ):
        self.predictor = predictor
        self.tile_size = tile_size
        self.overlap = overlap
        self.max_tile_ratio = max_tile_ratio
        self.min_area = min_area
        self.box_iou_threshold = box_iou_threshold
        self.mask_iou_threshold = mask_iou_threshold

    def process(self, pil_image, prompt: str, overlap: float | None = None):
        """Segment every instance matching ``prompt``.

        Returns ``(id_map, n_instances)``; ``id_map`` is a uint16 (H, W) map with
        0 as background, or None if nothing was detected.
        """
        stride = stride_from_overlap(self.tile_size, self.overlap if overlap is None else overlap)
        fragments = collect_tile_fragments(
            self.predictor,
            pil_image,
            prompt,
            self.tile_size,
            stride,
            self.min_area,
            self.max_tile_ratio,
        )
        if not fragments:
            return None, 0

        width, height = pil_image.size
        labels = self._group_fragments(fragments)
        id_map = paint_winner_takes_all(fragments, labels + 1, height, width)
        return id_map, count_instances(id_map)

    def _group_fragments(self, fragments: list[dict]) -> np.ndarray:
        """Label fragments belonging to the same object (connected components)."""
        n = len(fragments)
        adjacency = np.eye(n, dtype=bool)

        # Mask IoU is only computed for pairs whose boxes already overlap.
        box_iou = box_iou_matrix([f["bbox"] for f in fragments])
        for i, j in np.argwhere(np.triu(box_iou > self.box_iou_threshold, k=1)):
            if self._mask_iou(fragments[i], fragments[j]) > self.mask_iou_threshold:
                adjacency[i, j] = adjacency[j, i] = True

        _, labels = connected_components(csgraph=csr_matrix(adjacency), directed=False)
        return labels

    @staticmethod
    def _mask_iou(f1: dict, f2: dict) -> float:
        """IoU of two fragments, measured on the intersection of their boxes."""
        b1, b2 = f1["bbox"], f2["bbox"]
        ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        if ix1 > ix2 or iy1 > iy2:
            return 0.0

        # Bboxes are inclusive, hence the +1 on slice ends.
        p1 = f1["mask"][iy1 - b1[1] : iy2 - b1[1] + 1, ix1 - b1[0] : ix2 - b1[0] + 1]
        p2 = f2["mask"][iy1 - b2[1] : iy2 - b2[1] + 1, ix1 - b2[0] : ix2 - b2[0] + 1]
        intersection = np.logical_and(p1, p2).sum()
        if intersection == 0:
            return 0.0
        union = f1["mask"].sum() + f2["mask"].sum() - intersection
        return float(intersection / union)
