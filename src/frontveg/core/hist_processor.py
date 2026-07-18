import numpy as np
import cv2 as cv

class HistogramProcessor:
    """
    Handles depth normalization and histogram computation logic.
    """
    def __init__(self, hist_size=256, range_max=256):
        self.hist_size = hist_size
        self.range = [0, range_max]

    def compute_global_max(self, depth_list):
        """
        Finds the maximum depth value across a list of depth arrays
        to ensure consistent normalization across a batch.
        """
        if not depth_list:
            return 1.0
        
        # Calculate max, ignoring 0/NaN if necessary
        return max([np.max(d) for d in depth_list])

    def normalize_and_hist(self, depth_map, global_max):
        """
        Normalizes a depth map based on a global max and computes its histogram.
        
        Args:
            depth_map (np.ndarray): Raw depth map.
            global_max (float): The reference maximum value for scaling.
            
        Returns:
            tuple: (normalized_depth, flattened_histogram)
        """
        # Filter out zero values (often background/null) to avoid skewing the hist
        depth_filtered = np.where(depth_map > 0, depth_map, np.nan)
        
        # Normalize to 0-255 range
        depth_normalized = (depth_filtered / global_max * 255).astype('float32')
        
        # Prepare for histogram (OpenCV calcHist requires uint8)
        valid_pixels = depth_normalized[~np.isnan(depth_normalized)].astype('uint8')
        
        # Compute histogram
        hist = cv.calcHist([valid_pixels], [0], None, [self.hist_size], self.range)
        
        # Clean up depth for return (convert NaN back to 0)
        clean_depth = np.nan_to_num(depth_normalized).astype('uint8')
        
        return clean_depth, hist.flatten()