"""Default parameters shared by the napari plugin and the command-line tools."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FrontVegConfig:
    """Pipeline parameters. Defaults are the values recommended for trellised crops."""

    # Depth estimation (Depth-Anything-V2)
    encoder: str = "vitl"
    depth_input_size: int = 518

    # Foreground thresholding (histogram valley search)
    sigma: float = 1.1  # Gaussian smoothing of the depth histogram
    peak_dist: float = 8.0  # Minimal distance between histogram peaks, in bins
    peak_height: float = 0.5
    peak_prominence: float = 0.5
    max_valleys: int = 4  # Valleys considered when selecting the threshold
    smoothed: bool = True

    # Instance segmentation (SAM3)
    prompt: str = "leaf"
    tile_size: int = 640
    overlap: float = 0.5  # Fraction of overlap between adjacent tiles
    conf_threshold: float = 0.35
    min_fragment_area: int = 50  # Fragments smaller than this (px) are discarded
    max_tile_ratio: float = 0.55  # Fragments covering more of their tile are discarded

    def __post_init__(self) -> None:
        if not 0 <= self.overlap < 1:
            raise ValueError(f"overlap must be in [0, 1), got {self.overlap}")
        if self.peak_dist < 1:
            raise ValueError(f"peak_dist must be >= 1, got {self.peak_dist}")
        if self.tile_size <= 0:
            raise ValueError(f"tile_size must be positive, got {self.tile_size}")
