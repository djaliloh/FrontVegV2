"""FrontVeg foreground extraction: depth estimation, then histogram-valley thresholding."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from frontveg.config import FrontVegConfig
from frontveg.core.hist_processor import HistogramProcessor
from frontveg.core.segmenter import BinarySegmenter
from frontveg.core.valley_processor import ValleyProcessor
from frontveg.utils.image_loader import (
    ImageLoader,
    ImageLoadError,
    convert_output_filename,
    get_image_files,
)
from frontveg.utils.reporting import AnalysisReporter

logger = logging.getLogger(__name__)

# Used when no valley is found: middle of the normalized [0, 255] depth range.
FALLBACK_THRESHOLD = 127


@dataclass
class DepthAnalysis:
    """Intermediate results of the foreground extraction of one image."""

    depth: np.ndarray  # Normalized depth, uint8 (H, W)
    histogram: np.ndarray
    analysis: dict | None  # ValleyProcessor output, None if no valley was found
    threshold: float
    mask: np.ndarray  # Foreground mask, uint8 0/255 (H, W)

    @property
    def used_fallback(self) -> bool:
        return self.analysis is None


class FrontVegPipeline:
    """Depth-based foreground mask of the front crop row.

    Args:
        config: ``FrontVegConfig``, or a mapping with its fields plus the optional
            ``depth_repo_path`` and ``depth_ckpt_path`` keys.
        depth_repo_path: Depth-Anything-V2 repository.
        depth_ckpt_path: Folder holding ``depth_anything_v2_<encoder>.pth``.
    """

    def __init__(self, config, depth_repo_path=None, depth_ckpt_path=None):
        if isinstance(config, Mapping):
            depth_repo_path = depth_repo_path or config.get("depth_repo_path")
            depth_ckpt_path = depth_ckpt_path or config.get("depth_ckpt_path")
            config = FrontVegConfig.from_mapping(config)
        self.config = config

        # Imported here so that the thresholding code can be used without torch.
        from frontveg.models.depth_wrapper import DepthAnythingWrapper

        self.depth_model = DepthAnythingWrapper(
            encoder=config.encoder, checkpoint_path=depth_ckpt_path, repo_path=depth_repo_path
        )
        self.hist_engine = HistogramProcessor()
        self.valley_engine = ValleyProcessor(
            sigma=config.sigma,
            peak_dist=config.peak_dist,
            peak_height=config.peak_height,
            smooth=config.smoothed,
            prominence=config.peak_prominence,
            max_valleys=config.max_valleys,
        )
        self.reporter = AnalysisReporter()

    def infer_depth(self, image_rgb: np.ndarray) -> np.ndarray:
        return self.depth_model.infer(image_rgb, input_size=self.config.depth_input_size)

    def threshold_depth(self, depth, reference_max, sigma=None, peak_dist=None) -> DepthAnalysis:
        """Foreground mask of a depth map normalized by ``reference_max``."""
        clean_depth, hist = self.hist_engine.normalize_and_hist(depth, reference_max)
        analysis = self.valley_engine.find_optimal_threshold(hist, sigma=sigma, peak_dist=peak_dist)
        threshold = analysis["optimal_threshold"] if analysis else FALLBACK_THRESHOLD
        mask = BinarySegmenter.apply_threshold(clean_depth, threshold)
        return DepthAnalysis(clean_depth, hist, analysis, threshold, mask)

    def analyze(self, image_rgb: np.ndarray, sigma=None, peak_dist=None) -> DepthAnalysis:
        """Depth estimation and thresholding of a single RGB image."""
        depth = self.infer_depth(image_rgb)
        result = self.threshold_depth(depth, depth.max(), sigma=sigma, peak_dist=peak_dist)
        if result.used_fallback:
            logger.warning("No histogram valley found, using threshold %d", FALLBACK_THRESHOLD)
        return result

    def predict(self, image_rgb: np.ndarray, sigma=None, peak_dist=None) -> np.ndarray:
        """Foreground mask (uint8 0/255) of a single RGB image."""
        return self.analyze(image_rgb, sigma=sigma, peak_dist=peak_dist).mask

    def process_folder(self, input_dir, output_root, plot_graphics: bool = False) -> dict:
        """Write the depth map and foreground mask of every image of ``input_dir``.

        Depth maps are normalized with the maximum of the whole folder so that
        thresholds are comparable between images of the same acquisition.

        Returns:
            dict mapping each processed image path to its foreground mask path.
        """
        input_path = Path(input_dir)
        output_root = Path(output_root)
        depth_dir = output_root / "depth" / input_path.name
        mask_dir = output_root / "masks" / input_path.name
        plot_dir = output_root / "graphics" / input_path.name

        # 1. Depth estimation.
        logger.info("Depth estimation for %s", input_path.name)
        depths = {}
        for img_p in get_image_files(input_path):
            try:
                image = ImageLoader.load_image(img_p)
            except ImageLoadError as e:
                logger.warning("Skipping %s: %s", img_p.name, e)
                continue
            depths[img_p] = self.infer_depth(image)
            _write_png(depth_dir / convert_output_filename(img_p.name), depths[img_p])

        # 2. Thresholding with a folder-wide normalization.
        logger.info("Thresholding and mask generation for %s", input_path.name)
        global_max = self.hist_engine.compute_global_max(depths.values())
        masks = {}
        for img_p, depth in depths.items():
            result = self.threshold_depth(depth, global_max)
            if result.used_fallback:
                logger.warning(
                    "%s: no histogram valley found, using threshold %d",
                    img_p.name,
                    FALLBACK_THRESHOLD,
                )
            masks[img_p] = _write_png(mask_dir / convert_output_filename(img_p.name), result.mask)
            if plot_graphics and result.analysis:
                self.reporter.plot_histogram_analysis(
                    result.histogram,
                    result.analysis,
                    plot_dir / f"hist_{img_p.stem}.png",
                    title=f"Analysis: {img_p.name}",
                )
        return masks


def _write_png(path: Path, image: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Cannot write {path}")
    return path
