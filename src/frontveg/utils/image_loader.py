"""Image loading for standard (PNG, JPG, TIF, BMP) and RAW (CR3, CR2, NEF, ARW, DNG) files.

Every loader returns RGB ``uint8`` arrays of shape (H, W, 3): this is the
convention used throughout FrontVeg V2.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

STANDARD_FORMATS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
RAW_FORMATS = frozenset({".cr3", ".cr2", ".nef", ".arw", ".dng"})


class ImageLoadError(RuntimeError):
    """Raised when an image file cannot be read."""


def to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a grayscale, RGB or RGBA array of any numeric type to RGB uint8.

    Floats are expected in [0, 1] (or [0, 255] if their maximum exceeds 1);
    16-bit and other integer types are rescaled from their full range.
    """
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError(f"Expected an (H, W), (H, W, 3) or (H, W, 4) image, got {image.shape}")
    image = image[..., :3]

    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    if image.dtype == np.bool_:
        return image.astype(np.uint8) * 255
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if np.nanmax(image) <= 1.0 else 1.0
        return np.clip(np.nan_to_num(image) * scale, 0, 255).astype(np.uint8)
    if np.issubdtype(image.dtype, np.integer):
        return (image.astype(np.float64) / np.iinfo(image.dtype).max * 255).astype(np.uint8)
    raise ValueError(f"Unsupported image dtype: {image.dtype}")


class ImageLoader:
    """Unified image loading interface for diverse formats."""

    STANDARD_FORMATS = STANDARD_FORMATS
    RAW_FORMATS = RAW_FORMATS
    ALL_FORMATS = STANDARD_FORMATS | RAW_FORMATS

    @staticmethod
    def get_supported_extensions() -> tuple[str, ...]:
        return tuple(sorted(ImageLoader.ALL_FORMATS))

    @staticmethod
    def load_raw_image(path: str | Path) -> np.ndarray:
        """Develop a RAW file with the camera white balance. Requires ``rawpy``."""
        try:
            import rawpy
        except ImportError as e:
            raise ImageLoadError(
                f"Reading {Path(path).suffix.upper()} files requires rawpy: "
                "pip install 'frontvegV2[raw]'"
            ) from e
        try:
            with rawpy.imread(str(path)) as raw:
                return raw.postprocess(use_camera_wb=True, output_bps=8)
        except Exception as e:
            raise ImageLoadError(f"Cannot decode RAW file {path}: {e}") from e

    @staticmethod
    def load_standard_image(path: str | Path) -> np.ndarray:
        """Read a standard image with PIL, falling back to OpenCV (e.g. exotic TIFF)."""
        try:
            with Image.open(path) as img:
                if img.mode.startswith("I"):  # 16-bit grayscale, stored by PIL as int32
                    array = np.clip(np.array(img), 0, 65535).astype(np.uint16)
                elif img.mode == "F":
                    array = np.array(img)
                else:
                    array = np.array(img.convert("RGB"))
            return to_rgb_uint8(array)
        except Exception as pil_error:
            bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if bgr is None:
                raise ImageLoadError(f"Cannot read image {path}: {pil_error}") from pil_error
            if bgr.ndim == 3:
                code = cv2.COLOR_BGRA2RGBA if bgr.shape[2] == 4 else cv2.COLOR_BGR2RGB
                bgr = cv2.cvtColor(bgr, code)
            return to_rgb_uint8(bgr)

    @staticmethod
    def load_image(path: str | Path, return_format: str = "rgb") -> np.ndarray:
        """Load any supported image as an (H, W, 3) uint8 array.

        Args:
            path: Image file.
            return_format: ``"rgb"`` (default) or ``"bgr"`` (for OpenCV).

        Raises:
            ImageLoadError: The file is missing or cannot be decoded.
        """
        if return_format not in ("rgb", "bgr"):
            raise ValueError(f"return_format must be 'rgb' or 'bgr', got {return_format!r}")
        path = Path(path)
        if not path.is_file():
            raise ImageLoadError(f"File does not exist: {path}")

        if path.suffix.lower() in RAW_FORMATS:
            rgb = to_rgb_uint8(ImageLoader.load_raw_image(path))
        else:
            rgb = ImageLoader.load_standard_image(path)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if return_format == "bgr" else rgb

    @staticmethod
    def load_pil_image(path: str | Path) -> Image.Image:
        """Load any supported image (RAW included) as an RGB PIL image."""
        return Image.fromarray(ImageLoader.load_image(path))


def convert_output_filename(filename: str, target_format: str = "png") -> str:
    """Output file name for an input image: same stem, lossless ``target_format``.

    Masks and depth maps must never be written with a lossy format such as JPEG.
    """
    return Path(filename).with_suffix(f".{target_format}").name


def get_image_files(directory: str | Path, supported_only: bool = True) -> list[Path]:
    """Sorted image files of ``directory`` (extension matching is case-insensitive)."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return []
    files = (p for p in dir_path.iterdir() if p.is_file())
    if supported_only:
        files = (p for p in files if p.suffix.lower() in ImageLoader.ALL_FORMATS)
    return sorted(files)


def check_format_support() -> dict:
    """Report which formats can be read with the installed dependencies."""
    try:
        import rawpy  # noqa: F401

        rawpy_installed = True
    except ImportError:
        rawpy_installed = False
    return {
        "standard_formats": {ext.lstrip(".").upper(): True for ext in sorted(STANDARD_FORMATS)},
        "raw_formats": {ext.lstrip(".").upper(): rawpy_installed for ext in sorted(RAW_FORMATS)},
        "rawpy_installed": rawpy_installed,
    }
