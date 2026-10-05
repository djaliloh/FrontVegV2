"""Command-line batch processing: ``frontveg --input <images> --output <results>``."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

from frontveg.config import FrontVegConfig
from frontveg.resources import resolve_paths

DEFAULTS = FrontVegConfig()


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


def add_io_args(parser: argparse.ArgumentParser, default_output: str) -> None:
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Image folder (subfolders are processed separately)",
    )
    parser.add_argument("--output", default=Path(default_output), type=Path, help="Output folder")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show debug messages")


def add_depth_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("FrontVeg foreground (Depth-Anything-V2)")
    group.add_argument(
        "--encoder", default=DEFAULTS.encoder, choices=["vits", "vitb", "vitl", "vitg"]
    )
    group.add_argument("--sigma", type=float, default=DEFAULTS.sigma, help="Histogram smoothing")
    group.add_argument(
        "--peak_dist",
        type=float,
        default=DEFAULTS.peak_dist,
        help="Minimal distance between histogram peaks (bins)",
    )
    group.add_argument(
        "--peak_height",
        type=float,
        default=DEFAULTS.peak_height,
        help="Minimal height of histogram peaks",
    )
    group.add_argument(
        "--smoothed",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS.smoothed,
        help="Smooth the histogram",
    )
    group.add_argument("--depth_repo_path", type=Path, help="Depth-Anything-V2 repository")
    group.add_argument(
        "--depth_ckpt_path", type=Path, help="Folder holding depth_anything_v2_<encoder>.pth"
    )


def add_sam3_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Instance segmentation (SAM3)")
    group.add_argument(
        "--prompt", default=DEFAULTS.prompt, help="Text prompt, e.g. 'leaf' or 'grape'"
    )
    group.add_argument(
        "--tile_size", type=int, default=DEFAULTS.tile_size, help="Tile side in pixels"
    )
    group.add_argument(
        "--overlap", type=float, default=DEFAULTS.overlap, help="Tile overlap in [0, 1)"
    )
    group.add_argument(
        "--conf_threshold",
        type=float,
        default=DEFAULTS.conf_threshold,
        help="SAM3 confidence threshold",
    )
    group.add_argument("--sam3_repo_path", type=Path, help="SAM3 repository")
    group.add_argument("--sam3_ckpt", type=Path, help="SAM3 checkpoint (sam3.pt)")
    group.add_argument("--device", choices=["cuda", "cpu"], help="Default: cuda when available")


def paths_from_args(args: argparse.Namespace):
    """Resource paths, with command-line values taking precedence."""
    overrides = {
        "depth_repo": getattr(args, "depth_repo_path", None),
        "depth_ckpt_dir": getattr(args, "depth_ckpt_path", None),
        "sam3_repo": getattr(args, "sam3_repo_path", None),
        "sam3_ckpt": getattr(args, "sam3_ckpt", None),
    }
    return replace(resolve_paths(), **{k: v for k, v in overrides.items() if v is not None})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="FrontVeg V2 batch processing: FrontVeg foreground + SAM3 instances."
    )
    add_io_args(parser, default_output="output_final")
    add_depth_args(parser)
    add_sam3_args(parser)
    parser.add_argument("--no_plots", action="store_true", help="Do not save histogram plots")
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    log = logging.getLogger("frontveg")

    from frontveg.master_pipeline import MasterPipeline, MissingResourceError

    start = time.time()
    try:
        config = FrontVegConfig.from_mapping(vars(args))
        pipeline = MasterPipeline(config, paths_from_args(args), device=args.device)
    except (MissingResourceError, ValueError) as e:
        log.error("%s", e)
        return 2

    processed, failed = pipeline.run_full_stack(
        args.input, args.output, plot_graphics=not args.no_plots
    )
    elapsed = time.time() - start
    log.info("%d image(s) processed in %.1f min. Results: %s", processed, elapsed / 60, args.output)
    if failed:
        log.error("%d image(s) failed: %s", len(failed), ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
