import cv2
import numpy as np

class BinarySegmenter:
    """
    Applies binary thresholding to normalized depth maps.
    """
    @staticmethod
    def apply_threshold(depth_norm, threshold):
        """
        Performs binary thresholding.
        Args:
            depth_norm (np.ndarray): Normalized uint8 depth map.
            threshold (int): The valley index found by ValleyProcessor.
        Returns:
            np.ndarray: Binary mask (0 or 255).
        """
        if threshold is None:
            return None
        
        _, binary_mask = cv2.threshold(depth_norm, threshold, 255, cv2.THRESH_BINARY)
        return binary_mask