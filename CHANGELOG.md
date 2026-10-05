# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `MasterPipeline.segment()`: single entry point (depth, SAM3, fusion) shared by napari and the command line.
- `frontveg`, `frontveg-depth` and `frontveg-sam3` commands, with `--overlap`, `--conf_threshold`, `--device` and resource path options.
- Per-folder `summary/<folder>.csv` in batch mode.
- napari: output folder selector, optional FrontVeg foreground overlay, background processing.
- `FrontVegConfig`, the single source of default parameters.
- Test suite (no GPU required), GitHub Actions CI, pre-commit hooks.
- Documentation with step-by-step guides and screenshots (`docs/`, MkDocs).

### Changed

- Command-line defaults now match napari: `peak_dist=8` and histogram smoothing enabled (use `--no-smoothed` for the previous behavior).
- napari statistics count SAM3 instances, matching the displayed labels, instead of connected components.
- `ImageLoader` returns RGB arrays and raises `ImageLoadError` instead of returning `None`.
- All masks and depth maps are written as PNG, whatever the input format.
- SAM3 tiling skips edge tiles fully covered by the previous tile.
- Packaging moved to `pyproject.toml` (PEP 621); `opencv-python-headless` replaces `opencv-python`.
- Library messages use `logging` instead of `print`.

### Fixed

- Depth-Anything-V2 received RGB instead of BGR input in napari and for RAW files.
- napari ignored Sigma, Peak distance and Tile overlap changes after the first run.
- RAW images were never fused in batch mode; masks of JPEG inputs were saved as lossy JPEG.
- `sam3_overlay` images were saved with swapped colors.
- SAM3 failed on machines without CUDA.
- Git submodules could not be cloned (missing `.gitmodules`).

### Removed

- Empty modules (`dino_wrapper.py`, `sam2_wrapper.py`, `image_io.py`), unused helpers and commented-out code.

## [2.2.0]

Version published on PyPI before this changelog was introduced.
