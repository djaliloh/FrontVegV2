import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter

class ValleyProcessor:
    """
    Handles the detection of optimal thresholds in depth histograms 
    using peak and valley analysis.
    """
    def __init__(self, sigma=1.1, peak_dist=20, peak_height=0.5, smooth=False):
        self.sigma = sigma
        self.peak_dist = peak_dist
        self.peak_height = peak_height
        self.smooth = smooth

    def find_optimal_threshold(self, histogram):
        """
        Analyzes the histogram to find the deepest valley between peaks.
        
        Returns:
            dict: A dictionary containing the optimal threshold, all peaks, and all valleys.
        """
        y = histogram.copy()
        x = np.arange(len(y))

        # Optional smoothing using Gaussian filter
        if self.smooth:
            print("  > [Applying Gaussian] smoothing to histogram...")
            y_smoothed = gaussian_filter(y, sigma=self.sigma)
        else:
            print("  > [Skipping Gaussian] smoothing, using raw histogram...")
            y_smoothed = y

        # Detect peaks (modes) in the histogram 
        peaks, _ = find_peaks(
            y_smoothed, 
            height=self.peak_height, 
            prominence=0.5, 
            distance=self.peak_dist
        )

        if len(peaks) < 3:
            return None

        # Detect minima between peaks (starting from the 2nd mode)
        # valleys = []
        # for i in range(1, len(peaks) - 1):
        #     start, end = peaks[i], peaks[i + 1]
        #     min_index = np.argmin(y_smoothed[start:end]) + start
            
        #     # Calculate depth relative to neighboring peaks
        #     left_peak = peaks[i]
        #     right_peak = peaks[i + 1]
        #     depth_val = y_smoothed[min_index] - np.mean([y_smoothed[left_peak], y_smoothed[right_peak]])
            
        #     valleys.append({
        #         'index': int(min_index),
        #         'depth': float(depth_val),
        #         'y_value': float(y_smoothed[min_index])
        #     })

        valleys = []

        for i in range(1, len(peaks) - 1):

            start, end = peaks[i], peaks[i + 1]
            min_index = np.argmin(y_smoothed[start:end]) + start

            left_peak = peaks[i + 1]
            right_peak = peaks[i + 2] if i + 2 < len(peaks) else len(y_smoothed) - 1

            depth_val = y_smoothed[min_index] - np.mean(
                [y_smoothed[left_peak], y_smoothed[right_peak]]
            )

            valleys.append({
                "index": int(min_index),
                "depth": float(depth_val),
                "y_value": float(y_smoothed[min_index])
            })

        valleys = valleys[:4]

        if not valleys:
            return None

        # Selection of the optimal threshold (the deepest/lowest valley)
        # Based on your logic: minimum y value among detected valleys
        optimal_valley = min(valleys, key=lambda v: v['y_value'])
        
        return {
            'optimal_threshold': optimal_valley['index'],
            'peaks': peaks,
            'valleys': valleys,
            'smoothed_hist': y_smoothed
        }