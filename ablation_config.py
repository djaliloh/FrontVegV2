"""
ablation_config.py
==================
Central configuration dataclass and preset factory for the ablation study.

Ablation matrix
---------------
  baseline_a  use_frontveg=False  use_tiling=False  use_nms=False
  baseline_b  use_frontveg=False  use_tiling=True   use_nms=True
  baseline_c  use_frontveg=True   use_tiling=False  use_nms=False
  baseline_d  use_frontveg=True   use_tiling=True   use_nms=True

Tiling engine
-------------
  use_tiling=True   → SAM3StandardTiledInference
                       (sliding window + greedy box-IoU NMS inside the tiler)
                       NOT sam3_tiler.py – your graph-merge is the full pipeline.

  use_tiling=False  → SAM3FullImageInference
                       (single forward pass, no tiling at all)

NMS
---
  The box-IoU NMS is INSIDE SAM3StandardTiledInference (nms_iou_threshold).
  use_nms=True additionally runs NMSProcessor (post-hoc area filter) after
  tiling to remove any residual tiny/giant segments.
"""

from __future__ import annotations
import copy
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class AblationConfig:
    # ── Dataset / I/O ──────────────────────────────────────────────────────
    input_dir:  str = ""
    output_dir: str = "ablation_results"
    mode:       str = "baseline_d"

    # ── SAM3 ───────────────────────────────────────────────────────────────
    sam3_ckpt: str = ""
    prompt:    str = "leaf"

    # ── Tiling (baselines B & D) ────────────────────────────────────────
    use_tiling:        bool  = True
    tile_size:         int   = 640
    tile_overlap:      float = 0.50
    # Box-IoU NMS threshold *inside* SAM3StandardTiledInference
    nms_iou_threshold: float = 0.30

    # ── FrontVeg foreground filter (baselines C & D) ─────────────────────
    use_frontveg:    bool  = True
    encoder:         str   = "vitl"
    depth_ckpt_path: str   = "checkpoints/depthanything_ckpts"
    depth_repo_path: str   = "external/Depth-Anything-V2"
    sigma:           float = 1.1
    peak_dist:       int   = 20
    peak_height:     float = 0.5
    smoothed:        bool  = False

    # ── Post-hoc area NMS (baselines B & D) ──────────────────────────────
    use_nms:            bool  = True
    nms_min_area:       int   = 200
    nms_max_area_ratio: float = 0.55


# ---------------------------------------------------------------------------
# Pre-built presets
# ---------------------------------------------------------------------------

PRESETS: dict[str, AblationConfig] = {
    # A: SAM3 only – single pass, no tiling, no FrontVeg, no NMS
    "baseline_a": AblationConfig(
        mode="baseline_a",
        use_tiling=False,
        use_frontveg=False,
        use_nms=False,
    ),
    # B: SAM3 + standard tiling + standard box-IoU NMS (no FrontVeg, no graph merge)
    "baseline_b": AblationConfig(
        mode="baseline_b",
        use_tiling=True,
        use_frontveg=False,
        use_nms=True,
    ),
    # C: SAM3 + FrontVeg foreground filter (full-image SAM3, no tiling, no NMS)
    "baseline_c": AblationConfig(
        mode="baseline_c",
        use_tiling=False,
        use_frontveg=True,
        use_nms=False,
    ),
    # D: SAM3 + FrontVeg + standard tiling + standard NMS (no graph merge)
    "baseline_d": AblationConfig(
        mode="baseline_d",
        use_tiling=True,
        use_frontveg=True,
        use_nms=True,
    ),
}


def get_preset(mode: str) -> AblationConfig:
    """Return a shallow copy of the preset for *mode*."""
    if mode not in PRESETS:
        raise ValueError(
            f"Unknown ablation mode '{mode}'. Valid options: {list(PRESETS)}"
        )
    return copy.copy(PRESETS[mode])
