"""SAM3 tiled instance segmentation only (no FrontVeg filtering): ``frontveg-sam3 --input <images>``."""

from __future__ import annotations

import argparse
import logging
import sys

from frontveg.cli import add_io_args, add_sam3_args, configure_logging, paths_from_args
from frontveg.utils.image_loader import ImageLoader, ImageLoadError, get_image_files
from frontveg.utils.visualization import Visualizer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SAM3 tiled instance segmentation.")
    add_io_args(parser, default_output="output_sam3")
    add_sam3_args(parser)
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    log = logging.getLogger("frontveg")

    from frontveg.core.sam3_tiler import SAM3TiledInference
    from frontveg.models.sam3_wrapper import SAM3Predictor

    paths = paths_from_args(args)
    predictor = SAM3Predictor(
        paths.sam3_ckpt,
        repo_path=paths.sam3_repo,
        device=args.device,
        conf_threshold=args.conf_threshold,
    )
    tiler = SAM3TiledInference(predictor, tile_size=args.tile_size, overlap=args.overlap)

    for img_path in get_image_files(args.input):
        try:
            image = ImageLoader.load_pil_image(img_path)
        except ImageLoadError as e:
            log.warning("Skipping %s: %s", img_path.name, e)
            continue
        id_map, n_found = tiler.process(image, prompt=args.prompt)
        log.info("%s: %d instance(s)", img_path.name, n_found)
        if id_map is not None:
            Visualizer.save_results(args.output, img_path.name, id_map, image)
    return 0


if __name__ == "__main__":
    sys.exit(main())
