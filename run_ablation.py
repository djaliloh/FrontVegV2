"""
run_ablation.py
===============
Unified entry-point for the FrontVeg2 ablation study.

Ablation matrix
---------------
  baseline_a  SAM3 only
              → full-image SAM3, no tiling, no FrontVeg, no NMS

  baseline_b  SAM3 + standard tiling + standard NMS
              → SAM3StandardTiledInference (greedy box-IoU NMS),
                no FrontVeg, no graph merging

  baseline_c  SAM3 + FrontVeg foreground filter
              → full-image SAM3, FrontVeg mask intersection,
                no tiling, no NMS

  baseline_d  SAM3 + FrontVeg + standard tiling + standard NMS
              → SAM3StandardTiledInference (greedy box-IoU NMS)
                + FrontVeg mask intersection

NOTE: baselines B and D intentionally do NOT use sam3_tiler.py
(your connected-components graph merge). That is your novel method
and lives only in the full production pipeline (main_final.py).

Usage
-----
  python run_ablation.py \\
      --mode baseline_d \\
      --input  ./input/leaf \\
      --output ./ablation_results/baseline_d \\
      --sam3_ckpt ./checkpoints/sam3_ckpts/sam3.pt \\
      --prompt leaf \\
      --peak_dist 8 --sigma 1.1 --smoothed

Output (per mode)
-----------------
  <output>/
    sam3_overlay/                       <subdir>/*.png
    final_masks/                        <subdir>/*.png
    colored_finalid_overlay_input_rgb/  <subdir>/*.png
    run_log.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── project root -> src on PYTHONPATH ─────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from frontveg.models.sam3_wrapper import SAM3Predictor
from frontveg.core.sam3_standard_tiler import SAM3StandardTiledInference   # ablation tiler
from frontveg.core.sam3_full_inference import SAM3FullImageInference         # no-tiling wrapper
from frontveg.core.nms_processor import NMSProcessor                         # post-hoc area filter
from frontveg.core.post_processor import PostProcessor
from frontveg.utils.visualization import Visualizer
from frontveg.utils.image_loader import (
    ImageLoader,
    get_image_files,
    convert_output_filename,
)

from ablation_config import AblationConfig, get_preset


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _build_output_dirs(output_root: Path, subdir_name: str) -> dict:
    """Create and return the three mandatory output sub-directories."""
    dirs = {
        "sam3_overlay":                 output_root / "sam3_overlay" / subdir_name,
        "masks":                        output_root / "final_masks"  / subdir_name,
        "colored_finalid_on_input_rgb": output_root / "colored_finalid_overlay_input_rgb" / subdir_name,
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _save_outputs(
    paths: dict,
    output_filename: str,
    rgb_np: np.ndarray,
    final_id_map: np.ndarray,
    final_mask: np.ndarray,
) -> None:
    """Write the three standard output images."""
    # 1. sam3_overlay – colored instances blended onto the masked crop
    final_rgb_cutout = PostProcessor.apply_overlay(rgb_np, final_mask)
    colored_overlay  = Visualizer.create_overlay(final_rgb_cutout, final_id_map, alpha=0.5)
    cv2.imwrite(
        str(paths["sam3_overlay"] / output_filename),
        cv2.cvtColor(colored_overlay, cv2.COLOR_RGB2BGR),
    )

    # 2. final_masks – binary foreground mask
    cv2.imwrite(str(paths["masks"] / output_filename), final_mask)

    # 3. colored_finalid_overlay_input_rgb – instances overlaid on raw RGB
    colored_rgb = Visualizer.create_overlay(rgb_np, final_id_map, alpha=0.5)
    cv2.imwrite(str(paths["colored_finalid_on_input_rgb"] / output_filename), colored_rgb)


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline runner
# ═══════════════════════════════════════════════════════════════════════════

def run_ablation_pipeline(cfg: AblationConfig) -> None:
    """
    Execute one ablation variant defined by *cfg*.

    Engine selection
    ----------------
    use_tiling=False  →  SAM3FullImageInference   (single forward pass)
    use_tiling=True   →  SAM3StandardTiledInference
                            (sliding window + greedy box-IoU NMS)
                            NOT sam3_tiler.py (no graph merge)

    use_nms=True      →  additional NMSProcessor pass after stitching
                            (removes tiny / giant residual segments)
    use_frontveg=True →  FrontVegPipeline depth mask intersected with
                            SAM3 binary mask via PostProcessor
    """
    print(f"\n{'='*62}")
    print(f"  ABLATION MODE : {cfg.mode.upper()}")
    print(f"  FrontVeg      : {cfg.use_frontveg}")
    print(f"  Tiling engine : {'SAM3StandardTiledInference (box-IoU NMS)' if cfg.use_tiling else 'SAM3FullImageInference (no tiling)'}")
    print(f"  Post-NMS area filter : {cfg.use_nms}")
    print(f"{'='*62}\n")

    start       = time.time()
    output_root = Path(cfg.output_dir)

    # ── 1. Load SAM3 ──────────────────────────────────────────────────────
    print(">>> Loading SAM3 model...")
    sam3_model = SAM3Predictor(checkpoint_path=cfg.sam3_ckpt)

    if cfg.use_tiling:
        # Standard tiling + greedy box-IoU NMS  (ablation approach)
        sam3_engine = SAM3StandardTiledInference(
            sam3_model,
            tile_size=cfg.tile_size,
            overlap=cfg.tile_overlap,
            nms_iou_threshold=cfg.nms_iou_threshold,
        )
    else:
        # Single forward pass on the full image (no tiling at all)
        sam3_engine = SAM3FullImageInference(sam3_model)

    # ── 2. Load FrontVeg (depth-based foreground filter) ──────────────────
    frontveg_pipe = None
    if cfg.use_frontveg:
        print(">>> Loading FrontVeg (DepthAnything) model...")
        from frontveg.core.pipeline import FrontVegPipeline
        frontveg_config = {
            "encoder":         cfg.encoder,
            "depth_ckpt_path": cfg.depth_ckpt_path,
            "depth_repo_path": cfg.depth_repo_path,
            "sigma":           cfg.sigma,
            "peak_dist":       cfg.peak_dist,
            "peak_height":     cfg.peak_height,
            "smoothed":        cfg.smoothed,
        }
        frontveg_pipe = FrontVegPipeline(frontveg_config)

    # ── 3. Optional post-hoc area NMS ─────────────────────────────────────
    nms_proc = (
        NMSProcessor(min_area=cfg.nms_min_area, max_area_ratio=cfg.nms_max_area_ratio)
        if cfg.use_nms else None
    )

    # ── 4. Iterate over input sub-directories ─────────────────────────────
    input_path = Path(cfg.input_dir)
    subdirs    = [p for p in input_path.iterdir() if p.is_dir()]
    if not subdirs:
        subdirs = [input_path]

    total_images = 0
    total_ok     = 0

    for subdir in subdirs:
        print(f"\n--- Processing: {subdir.name} ---")

        # ── 4a. FrontVeg depth-mask pass ──────────────────────────────
        temp_fv_root = output_root / "temp_frontveg"
        if cfg.use_frontveg and frontveg_pipe is not None:
            frontveg_pipe.process_folder(str(subdir), temp_fv_root, plot_graphics=False)

        images = get_image_files(str(subdir), supported_only=True)
        if not images:
            print(f"  ! No supported images in {subdir.name}")
            continue

        out_dirs = _build_output_dirs(output_root, subdir.name)

        for img_p in images:
            total_images += 1
            try:
                # ── Load RGB ──────────────────────────────────────────
                rgb_np = ImageLoader.load_image(str(img_p), return_format="rgb")
                if rgb_np is None or rgb_np.ndim != 3 or rgb_np.shape[2] not in (3, 4):
                    print(f"  ! Cannot load / invalid format: {img_p.name}")
                    continue

                # ── SAM3 inference ────────────────────────────────────
                pil_img = ImageLoader.load_pil_image(str(img_p))
                if pil_img is None:
                    print(f"  ! Cannot open PIL image: {img_p.name}")
                    continue

                id_map_raw, _ = sam3_engine.process(pil_img, prompt=cfg.prompt)
                if id_map_raw is None:
                    print(f"  ! SAM3 found nothing for {img_p.name}, generating blank outputs.")
                    id_map_raw = np.zeros(rgb_np.shape[:2], dtype=np.int32)

                # ── Optional post-hoc area NMS ────────────────────────
                if nms_proc is not None:
                    id_map_raw = nms_proc.suppress(id_map_raw)
                    if id_map_raw is None or id_map_raw.max() == 0:
                        print(f"  ! Post-NMS area filter removed all segments: {img_p.name}, generating blank outputs.")
                        if id_map_raw is None:
                            id_map_raw = np.zeros(rgb_np.shape[:2], dtype=np.int32)

                # ── Optional FrontVeg mask intersection ───────────────
                if cfg.use_frontveg:
                    output_name = convert_output_filename(img_p.name, "png")
                    fv_mask_p   = temp_fv_root / "masks" / subdir.name / img_p.name
                    if not fv_mask_p.exists():
                        fv_mask_p = temp_fv_root / "masks" / subdir.name / output_name

                    if not fv_mask_p.exists():
                        print(f"  ! FrontVeg mask not found: {img_p.name}")
                        continue

                    mask_fv = cv2.imread(str(fv_mask_p), cv2.IMREAD_GRAYSCALE)
                    if mask_fv is None:
                        print(f"  ! Cannot load FrontVeg mask: {img_p.name}")
                        continue

                    final_id_map, final_mask = PostProcessor.get_final_colored_map(
                        id_map_raw, mask_fv
                    )
                else:
                    # No FrontVeg: use the raw SAM3 binary mask directly
                    final_id_map = id_map_raw
                    final_mask   = (id_map_raw > 0).astype(np.uint8) * 255

                # ── Save outputs ──────────────────────────────────────
                output_filename = convert_output_filename(img_p.name, "png")
                _save_outputs(out_dirs, output_filename, rgb_np, final_id_map, final_mask)
                total_ok += 1
                print(f"  ✓ {img_p.name}  →  {output_filename}")

            except Exception as exc:
                print(f"  ! Error on {img_p.name}: {exc}")
                continue

    # ── 5. Timing + run-log ───────────────────────────────────────────────
    elapsed = time.time() - start
    print(f"\n  Total  : {total_ok}/{total_images} images processed")
    print(f"  Time   : {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"  Output → {output_root}")

    log = {
        "mode":              cfg.mode,
        "use_tiling":        cfg.use_tiling,
        "tiling_engine":     "SAM3StandardTiledInference" if cfg.use_tiling else "SAM3FullImageInference",
        "use_frontveg":      cfg.use_frontveg,
        "use_nms_area":      cfg.use_nms,
        "nms_iou_threshold": cfg.nms_iou_threshold if cfg.use_tiling else None,
        "tile_size":         cfg.tile_size,
        "prompt":            cfg.prompt,
        "input_dir":         str(cfg.input_dir),
        "output_dir":        str(cfg.output_dir),
        "total_images":      total_images,
        "processed_ok":      total_ok,
        "elapsed_sec":       round(elapsed, 2),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "run_log.json").write_text(json.dumps(log, indent=2))
    print(f"  Log    → {output_root / 'run_log.json'}\n")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FrontVeg2 Ablation Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument("--mode",      required=True,
                   choices=["baseline_a", "baseline_b", "baseline_c", "baseline_d"])
    p.add_argument("--input",     required=True, help="Input image directory.")
    p.add_argument("--output",    default=None,
                   help="Output dir (default: ablation_results/<mode>).")
    p.add_argument("--sam3_ckpt", required=True)

    # SAM3
    p.add_argument("--prompt",       default="leaf")

    # Tiling  (baselines B & D only – uses SAM3StandardTiledInference)
    p.add_argument("--tile_size",        type=int,   default=640)
    p.add_argument("--tile_overlap",     type=float, default=0.50)
    p.add_argument("--nms_iou_threshold",type=float, default=0.30,
                   help="Box-IoU NMS threshold inside SAM3StandardTiledInference.")

    # FrontVeg  (baselines C & D only)
    p.add_argument("--encoder",          default="vitl")
    p.add_argument("--depth_ckpt_path",  default="checkpoints/depthanything_ckpts")
    p.add_argument("--depth_repo_path",  default="external/Depth-Anything-V2")
    p.add_argument("--sigma",            type=float, default=1.1)
    p.add_argument("--peak_dist",        type=int,   default=20)
    p.add_argument("--peak_height",      type=float, default=0.5)
    p.add_argument("--smoothed",         action="store_true")

    # Post-hoc area NMS  (baselines B & D only)
    p.add_argument("--nms_min_area",       type=int,   default=200)
    p.add_argument("--nms_max_area_ratio", type=float, default=0.55)

    return p


def main():
    args = build_parser().parse_args()

    # Load the correct preset (sets use_tiling / use_frontveg / use_nms flags)
    cfg = get_preset(args.mode)

    # Override I/O and hyperparams from CLI
    cfg.input_dir        = args.input
    cfg.output_dir       = args.output or f"ablation_results/{args.mode}"
    cfg.sam3_ckpt        = args.sam3_ckpt
    cfg.prompt           = args.prompt
    cfg.tile_size        = args.tile_size
    cfg.tile_overlap     = args.tile_overlap
    cfg.nms_iou_threshold= args.nms_iou_threshold
    cfg.encoder          = args.encoder
    cfg.depth_ckpt_path  = args.depth_ckpt_path
    cfg.depth_repo_path  = args.depth_repo_path
    cfg.sigma            = args.sigma
    cfg.peak_dist        = args.peak_dist
    cfg.peak_height      = args.peak_height
    cfg.smoothed         = args.smoothed
    cfg.nms_min_area         = args.nms_min_area
    cfg.nms_max_area_ratio   = args.nms_max_area_ratio

    run_ablation_pipeline(cfg)


if __name__ == "__main__":
    main()
