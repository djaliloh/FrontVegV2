"""FrontVeg foreground masks only (no SAM3): ``frontveg-depth --input <images>``."""

from __future__ import annotations

import argparse
import logging
import sys

from frontveg.cli import add_depth_args, add_io_args, configure_logging, paths_from_args
from frontveg.config import FrontVegConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FrontVeg depth-based foreground segmentation.")
    add_io_args(parser, default_output="output_frontveg")
    add_depth_args(parser)
    parser.add_argument("--plot", action="store_true", help="Save histogram plots")
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    from frontveg.core.pipeline import FrontVegPipeline

    paths = paths_from_args(args)
    missing = paths.missing(args.encoder, sam3=False)
    if missing:
        logging.getLogger("frontveg").error("Missing resources:\n- %s", "\n- ".join(missing))
        return 2
    pipeline = FrontVegPipeline(
        FrontVegConfig.from_mapping(vars(args)),
        depth_repo_path=paths.depth_repo,
        depth_ckpt_path=paths.depth_ckpt_dir,
    )
    subdirs = sorted(p for p in args.input.iterdir() if p.is_dir()) or [args.input]
    for folder in subdirs:
        masks = pipeline.process_folder(folder, args.output, plot_graphics=args.plot)
        logging.getLogger("frontveg").info("%s: %d mask(s) written", folder.name, len(masks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
