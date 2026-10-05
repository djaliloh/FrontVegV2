"""Complete FrontVeg V2 workflow: FrontVeg foreground, SAM3 instances, then fusion.

Single entry point shared by the napari plugin and the command-line tools.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from frontveg.config import FrontVegConfig
from frontveg.core.pipeline import DepthAnalysis, FrontVegPipeline
from frontveg.core.post_processor import PostProcessor
from frontveg.core.sam3_tiler import SAM3TiledInference
from frontveg.resources import ResourcePaths, resolve_paths
from frontveg.utils.image_loader import (
    ImageLoader,
    ImageLoadError,
    convert_output_filename,
    get_image_files,
)
from frontveg.utils.visualization import Visualizer, write_image

logger = logging.getLogger(__name__)


class MissingResourceError(FileNotFoundError):
    """Raised when an external repository or a checkpoint cannot be found."""


@dataclass
class SegmentationResult:
    id_map: np.ndarray  # Final instances, uint16 (H, W), 0 = background
    mask: np.ndarray  # Final binary mask, uint8 0/255
    foreground: np.ndarray  # FrontVeg foreground mask, uint8 0/255
    depth: DepthAnalysis | None = None  # None when the foreground was provided

    @property
    def instance_areas(self) -> dict[int, int]:
        """Area in pixels of each instance id."""
        ids, counts = np.unique(self.id_map[self.id_map > 0], return_counts=True)
        return dict(zip(ids.tolist(), counts.tolist()))


class MasterPipeline:
    """Load Depth-Anything-V2 and SAM3 once, then segment any number of images."""

    def __init__(
        self,
        config: FrontVegConfig | None = None,
        paths: ResourcePaths | None = None,
        device: str | None = None,
    ):
        self.config = config or FrontVegConfig()
        self.paths = paths or resolve_paths()
        missing = self.paths.missing(self.config.encoder)
        if missing:
            raise MissingResourceError("Missing FrontVeg V2 resources:\n- " + "\n- ".join(missing))

        from frontveg.models.sam3_wrapper import SAM3Predictor

        self.frontveg = FrontVegPipeline(
            self.config,
            depth_repo_path=self.paths.depth_repo,
            depth_ckpt_path=self.paths.depth_ckpt_dir,
        )
        self.sam3 = SAM3TiledInference(
            SAM3Predictor(
                self.paths.sam3_ckpt,
                repo_path=self.paths.sam3_repo,
                device=device,
                conf_threshold=self.config.conf_threshold,
            ),
            tile_size=self.config.tile_size,
            overlap=self.config.overlap,
            max_tile_ratio=self.config.max_tile_ratio,
            min_area=self.config.min_fragment_area,
        )
        self.processor = PostProcessor()

    def segment(
        self,
        image_rgb: np.ndarray,
        prompt: str | None = None,
        sigma: float | None = None,
        peak_dist: float | None = None,
        overlap: float | None = None,
        foreground: np.ndarray | None = None,
    ) -> SegmentationResult:
        """Segment the ``prompt`` instances of the front row of an RGB image.

        Unset parameters fall back to the pipeline configuration. ``foreground``
        skips depth estimation (e.g. masks computed with a folder-wide normalization).
        """
        prompt = prompt or self.config.prompt
        depth = None
        if foreground is None:
            depth = self.frontveg.analyze(image_rgb, sigma=sigma, peak_dist=peak_dist)
            foreground = depth.mask

        sam3_ids, _ = self.sam3.process(Image.fromarray(image_rgb), prompt=prompt, overlap=overlap)
        if sam3_ids is None:
            sam3_ids = np.zeros(image_rgb.shape[:2], dtype=np.uint16)

        id_map, mask = self.processor.get_final_colored_map(sam3_ids, foreground)
        return SegmentationResult(id_map=id_map, mask=mask, foreground=foreground, depth=depth)

    def run_full_stack(self, input_dir, output_dir, prompt: str | None = None, plot_graphics=True):
        """Process ``input_dir`` (or each of its subfolders) and write every output.

        Returns:
            tuple: ``(n_processed, failed_image_names)``.
        """
        input_dir, output_dir = Path(input_dir), Path(output_dir)
        subdirs = sorted(p for p in input_dir.iterdir() if p.is_dir()) or [input_dir]
        processed, failed = 0, []

        for subdir in subdirs:
            logger.info("Processing folder %s", subdir.name)
            # Foreground masks are computed per folder for a common depth normalization.
            fg_masks = self.frontveg.process_folder(
                subdir, output_dir / "temp_frontveg", plot_graphics=plot_graphics
            )
            rows = []
            for img_p in get_image_files(subdir):
                if img_p not in fg_masks:
                    failed.append(img_p.name)
                    continue
                try:
                    rows.append(self._process_image(img_p, fg_masks[img_p], output_dir, subdir.name, prompt))
                    processed += 1
                except ImageLoadError as e:
                    logger.warning("Skipping %s: %s", img_p.name, e)
                    failed.append(img_p.name)
                except Exception:
                    logger.exception("Failed to process %s", img_p.name)
                    failed.append(img_p.name)
            if rows:
                _write_summary(output_dir / "summary" / f"{subdir.name}.csv", rows)
        return processed, failed

    def _process_image(self, img_p: Path, fg_path: Path, output_dir: Path, group: str, prompt):
        logger.info("Segmenting %s", img_p.name)
        rgb = ImageLoader.load_image(img_p)
        foreground = cv2.imread(str(fg_path), cv2.IMREAD_GRAYSCALE)
        result = self.segment(rgb, prompt=prompt, foreground=foreground)

        name = convert_output_filename(img_p.name)
        cutout = self.processor.apply_overlay(rgb, result.mask)
        write_image(output_dir / "final_masks" / group / name, result.mask)
        write_image(
            output_dir / "sam3_overlay" / group / name,
            Visualizer.create_overlay(cutout, result.id_map),
        )
        write_image(
            output_dir / "colored_finalid_overlay_input_rgb" / group / name,
            Visualizer.create_overlay(rgb, result.id_map),
        )
        areas = result.instance_areas
        return {
            "image": img_p.name,
            "instances": len(areas),
            "total_area_px": sum(areas.values()),
            "mean_area_px": round(float(np.mean(list(areas.values()))), 1) if areas else 0,
        }


def _write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
