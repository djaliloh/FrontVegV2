"""
run_all_ablations.py
====================
Convenience script that runs ALL four ablation baselines sequentially
and then aggregates metrics into a single comparison CSV / Excel file.

Usage
-----
  python run_all_ablations.py \\
      --input       ./input/leaf \\
      --gt_dir      ./ground_truth/leaf \\
      --output_root ./ablation_results \\
      --sam3_ckpt   ./checkpoints/sam3_ckpts/sam3.pt \\
      --prompt      leaf \\
      --peak_dist   8 \\
      --sigma       1.1 \\
      --smoothed

It calls run_ablation.py for each mode and then run_metrics.py for each
result folder, finally printing a comparison table to stdout and saving
ablation_comparison.csv / .xlsx under <output_root>.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import numpy as np

MODES = ["baseline_a", "baseline_b", "baseline_c", "baseline_d"]
MODE_LABELS = {
    "baseline_a": "A: SAM3 only",
    "baseline_b": "B: SAM3 + Tiling + NMS",
    "baseline_c": "C: SAM3 + FrontVeg",
    "baseline_d": "D: SAM3 + FrontVeg + Tiling + NMS",
}


def run_one_mode(mode: str, args: argparse.Namespace, output_root: Path) -> Path:
    """Invoke run_ablation.py for a single mode and return its output dir."""
    out_dir = output_root / mode
    cmd = [
        sys.executable, "run_ablation.py",
        "--mode",         mode,
        "--input",        args.input,
        "--output",       str(out_dir),
        "--sam3_ckpt",    args.sam3_ckpt,
        "--prompt",       args.prompt,
        "--tile_size",    str(args.tile_size),
        "--tile_overlap", str(args.tile_overlap),
        "--encoder",      args.encoder,
        "--depth_ckpt_path", args.depth_ckpt_path,
        "--depth_repo_path", args.depth_repo_path,
        "--sigma",        str(args.sigma),
        "--peak_dist",    str(args.peak_dist),
        "--peak_height",  str(args.peak_height),
        "--nms_min_area", str(args.nms_min_area),
        "--nms_max_area_ratio", str(args.nms_max_area_ratio),
    ]
    if args.smoothed:
        cmd.append("--smoothed")

    print(f"\n{'#'*60}")
    print(f"  Running {MODE_LABELS[mode]}")
    print(f"{'#'*60}")
    subprocess.run(cmd, check=True)
    return out_dir


def collect_metrics(mode: str, out_dir: Path, gt_dir: str,
                    metrics_root: Path) -> dict | None:
    """
    Run run_metrics.py against a single mode's final_masks output and
    return a dict with aggregated stats.
    """
    # Find the first subdir that has masks (mirrors the main pipeline layout)
    masks_root = out_dir / "final_masks"
    if not masks_root.exists():
        print(f"  [WARN] No final_masks found in {out_dir}")
        return None

    # Collect all immediate subdirs (one per dataset subfolder)
    mask_subdirs = [p for p in masks_root.iterdir() if p.is_dir()]
    if not mask_subdirs:
        mask_subdirs = [masks_root]  # flat layout fallback

    all_dices, all_ious = [], []

    for mask_subdir in mask_subdirs:
        # GT subdir mirrors input subdir name
        gt_subdir = Path(gt_dir) / mask_subdir.name
        if not gt_subdir.exists():
            gt_subdir = Path(gt_dir)  # flat fallback

        metrics_out = metrics_root / mode / mask_subdir.name
        metrics_out.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, "run_metrics.py",
            "--gt_dir",    str(gt_subdir),
            "--mask_dir",  str(mask_subdir),
            "--output_dir", str(metrics_out),
            "--method",    mode,
            "--skip_confusion",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"  [WARN] Metrics failed for {mode}/{mask_subdir.name}: {result.stderr}")
            continue

        # Parse Excel result
        excel_path = metrics_out / f"results_{mode}.xlsx"
        if excel_path.exists():
            df = pd.read_excel(excel_path, sheet_name="Summary")
            all_dices.append(float(df["Mean Dice"].iloc[0]))
            all_ious.append(float(df["Mean IoU"].iloc[0]))

    if not all_dices:
        return None

    return {
        "Mode":      MODE_LABELS[mode],
        "Mean Dice": round(float(np.mean(all_dices)), 4),
        "Std Dice":  round(float(np.std(all_dices)),  4),
        "Mean IoU":  round(float(np.mean(all_ious)),  4),
        "Std IoU":   round(float(np.std(all_ious)),   4),
    }


def main():
    p = argparse.ArgumentParser(
        description="Run all FrontVeg2 ablation baselines and aggregate metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",       type=str, required=True)
    p.add_argument("--gt_dir",      type=str, required=True,
                   help="Ground-truth binary mask directory.")
    p.add_argument("--output_root", type=str, default="ablation_results")
    p.add_argument("--sam3_ckpt",   type=str, required=True)
    p.add_argument("--prompt",      type=str, default="leaf")
    p.add_argument("--tile_size",   type=int, default=640)
    p.add_argument("--tile_overlap",type=float, default=0.50)
    p.add_argument("--encoder",     type=str, default="vitl")
    p.add_argument("--depth_ckpt_path", type=str, default="checkpoints/depthanything_ckpts")
    p.add_argument("--depth_repo_path", type=str, default="external/Depth-Anything-V2")
    p.add_argument("--sigma",       type=float, default=1.1)
    p.add_argument("--peak_dist",   type=int,   default=20)
    p.add_argument("--peak_height", type=float, default=0.5)
    p.add_argument("--smoothed",    action="store_true")
    p.add_argument("--nms_min_area",        type=int,   default=200)
    p.add_argument("--nms_max_area_ratio",  type=float, default=0.55)
    p.add_argument("--modes", nargs="+", default=MODES,
                   help="Subset of modes to run (default: all four).")
    p.add_argument("--skip_inference", action="store_true",
                   help="Skip inference and only re-compute metrics on existing outputs.")

    args = p.parse_args()
    output_root  = Path(args.output_root)
    metrics_root = output_root / "metrics"

    # ── Run inference for each mode ───────────────────────────────────────
    if not args.skip_inference:
        for mode in args.modes:
            run_one_mode(mode, args, output_root)

    # ── Compute and aggregate metrics ─────────────────────────────────────
    rows = []
    for mode in args.modes:
        out_dir = output_root / mode
        row = collect_metrics(mode, out_dir, args.gt_dir, metrics_root)
        if row:
            rows.append(row)

    if not rows:
        print("\n[WARN] No metric rows collected – check gt_dir paths.")
        return

    df = pd.DataFrame(rows)
    print("\n" + "="*70)
    print("  ABLATION STUDY RESULTS")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70 + "\n")

    csv_path   = output_root / "ablation_comparison.csv"
    excel_path = output_root / "ablation_comparison.xlsx"
    df.to_csv(csv_path,   index=False)
    df.to_excel(excel_path, index=False)
    print(f"  Saved → {csv_path}")
    print(f"  Saved → {excel_path}")


if __name__ == "__main__":
    main()
