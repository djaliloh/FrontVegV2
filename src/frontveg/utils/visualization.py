"""Colored instance overlays and result files."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from frontveg.utils.image_loader import convert_output_filename, to_rgb_uint8


def instance_palette(n_ids: int) -> np.ndarray:
    """Deterministic RGB color per instance id; id 0 (background) is black."""
    ids = np.arange(n_ids)
    # Golden-ratio hue spacing keeps neighbouring ids visually distinct.
    hsv = np.stack(
        [(ids * 0.618034 % 1.0) * 179, np.full(n_ids, 200), np.full(n_ids, 230)], axis=-1
    ).astype(np.uint8)
    palette = cv2.cvtColor(hsv[None], cv2.COLOR_HSV2RGB)[0]
    palette[0] = 0
    return palette


def write_image(path, image: np.ndarray) -> Path:
    """Write an RGB, grayscale or 16-bit image, creating the parent folder."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Cannot write {path}")
    return path


class Visualizer:
    @staticmethod
    def create_overlay(image_rgb: np.ndarray, id_map: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """RGB image with translucent instance colors and white outlines."""
        canvas = to_rgb_uint8(image_rgb)
        foreground = id_map > 0
        overlay = canvas.copy()
        overlay[foreground] = instance_palette(int(id_map.max()) + 1)[id_map[foreground]]
        combined = cv2.addWeighted(canvas, 1 - alpha, overlay, alpha, 0)

        contours, _ = cv2.findContours(
            foreground.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(combined, contours, -1, (255, 255, 255), 1)
        return combined

    @staticmethod
    def save_results(output_dir, img_name: str, id_map: np.ndarray, original_pil) -> None:
        """Write the 16-bit id map, the binary mask and the overlay of one image."""
        output_dir = Path(output_dir)
        stem = Path(img_name).stem
        write_image(output_dir / "masks_id" / f"{stem}_id.png", id_map.astype(np.uint16))
        write_image(
            output_dir / "masks_visible" / f"{stem}_visible.png",
            (id_map > 0).astype(np.uint8) * 255,
        )
        overlay = Visualizer.create_overlay(np.array(original_pil), id_map)
        write_image(output_dir / "overlay" / f"visu_{convert_output_filename(img_name)}", overlay)
