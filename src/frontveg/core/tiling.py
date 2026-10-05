"""Building blocks shared by the SAM3 inference strategies.

A *fragment* is one SAM3 mask detected in one tile, stored as a dict:
``bbox`` (inclusive ``[x1, y1, x2, y2]`` in image coordinates), ``mask`` (boolean
crop of the bbox) and ``score``.
"""

from __future__ import annotations

import numpy as np


def stride_from_overlap(tile_size: int, overlap: float) -> int:
    """Distance between two consecutive tiles."""
    if not 0 <= overlap < 1:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    return max(1, int(tile_size * (1 - overlap)))


def tile_origins(length: int, tile_size: int, stride: int) -> list[int]:
    """Tile start offsets along one axis.

    Stops at the first tile reaching the border: later tiles would be fully
    contained in it and only add tiny, noisy crops.
    """
    origins = []
    for start in range(0, length, stride):
        origins.append(start)
        if start + tile_size >= length:
            break
    return origins


def is_hallucination(mask: np.ndarray, area_ref: int, min_area: int, max_ratio: float) -> bool:
    """True if the mask is too small, or too large compared to ``area_ref``."""
    area = mask.sum()
    return area < min_area or area / area_ref > max_ratio


def extract_fragments(
    masks: np.ndarray,
    scores: np.ndarray,
    origin: tuple[int, int],
    tile_shape: tuple[int, int],
    min_area: int,
    max_ratio: float,
) -> list[dict]:
    """Convert the SAM3 masks of one tile into fragments in image coordinates."""
    x0, y0 = origin
    h_t, w_t = tile_shape
    fragments = []
    for raw, score in zip(masks, scores):
        mask = raw[0] > 0
        if is_hallucination(mask, w_t * h_t, min_area, max_ratio):
            continue
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        y1, x1, y2, x2 = ys.min(), xs.min(), ys.max(), xs.max()
        fragments.append(
            {
                "bbox": [int(x1 + x0), int(y1 + y0), int(x2 + x0), int(y2 + y0)],
                "mask": mask[y1 : y2 + 1, x1 : x2 + 1],  # Cropped to save memory
                "score": float(score),
            }
        )
    return fragments


def collect_tile_fragments(
    predictor,
    pil_image,
    prompt: str,
    tile_size: int,
    stride: int,
    min_area: int,
    max_ratio: float,
) -> list[dict]:
    """Run SAM3 on every tile of the image and gather the valid fragments."""
    width, height = pil_image.size
    fragments = []
    for y in tile_origins(height, tile_size, stride):
        for x in tile_origins(width, tile_size, stride):
            x_end, y_end = min(x + tile_size, width), min(y + tile_size, height)
            masks, scores = predictor.predict_crop(pil_image.crop((x, y, x_end, y_end)), prompt)
            if masks is not None:
                fragments += extract_fragments(
                    masks, scores, (x, y), (y_end - y, x_end - x), min_area, max_ratio
                )
    release_cache(predictor)
    return fragments


def release_cache(predictor) -> None:
    """Free the GPU cache of the predictor, if it exposes one."""
    release = getattr(predictor, "release_cache", None)
    if callable(release):
        release()


def box_iou_matrix(boxes: np.ndarray) -> np.ndarray:
    """Pairwise IoU of ``[x1, y1, x2, y2]`` boxes (same formula as ``torchvision.ops.box_iou``)."""
    boxes = np.asarray(boxes, dtype=np.float32)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    top_left = np.maximum(boxes[:, None, :2], boxes[None, :, :2])
    bottom_right = np.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
    wh = np.clip(bottom_right - top_left, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        return inter / (area[:, None] + area[None, :] - inter)


def paint_winner_takes_all(
    fragments: list[dict], instance_ids, height: int, width: int
) -> np.ndarray:
    """Stitch fragments into an instance map; each pixel keeps its highest-score fragment."""
    id_map = np.zeros((height, width), dtype=np.uint16)
    best_score = np.zeros((height, width), dtype=np.float32)
    for frag, inst_id in zip(fragments, instance_ids):
        x1, y1 = frag["bbox"][:2]
        patch = frag["mask"]
        y2, x2 = min(y1 + patch.shape[0], height), min(x1 + patch.shape[1], width)
        update = patch[: y2 - y1, : x2 - x1] & (frag["score"] > best_score[y1:y2, x1:x2])
        best_score[y1:y2, x1:x2][update] = frag["score"]
        id_map[y1:y2, x1:x2][update] = inst_id
    return id_map


def count_instances(id_map: np.ndarray) -> int:
    """Number of distinct non-zero ids."""
    return int(np.count_nonzero(np.unique(id_map)))
