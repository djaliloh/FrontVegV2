"""Fusion of the FrontVeg foreground mask with the SAM3 instances."""

from __future__ import annotations

import cv2
import numpy as np


class PostProcessor:
    @staticmethod
    def _match_size(mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        if mask.shape[:2] == shape[:2]:
            return mask
        return cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def apply_overlay(rgb_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Black out every pixel of ``rgb_image`` outside ``mask`` (0/255)."""
        keep = PostProcessor._match_size(mask, rgb_image.shape) > 127
        return np.where(keep[..., None], rgb_image, 0).astype(np.uint8)

    @staticmethod
    def get_final_colored_map(id_map_sam: np.ndarray, mask_fv: np.ndarray):
        """Keep the SAM3 instances only inside the FrontVeg foreground.

        Returns:
            tuple: ``(final_id_map, final_binary_mask)``; uint16 ids and uint8 0/255 mask.
        """
        mask_fv = PostProcessor._match_size(mask_fv, id_map_sam.shape)
        foreground = (id_map_sam > 0) & (mask_fv > 127)
        final_id_map = np.where(foreground, id_map_sam, 0).astype(np.uint16)
        return final_id_map, foreground.astype(np.uint8) * 255
