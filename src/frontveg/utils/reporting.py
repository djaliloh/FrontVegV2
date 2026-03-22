import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

class AnalysisReporter:
    """
    Handles logging analysis results to Excel and generating diagnostic plots.
    """
    @staticmethod
    def log_to_excel(excel_path, data_list, image_name):
        """Logs detected minima depths to an Excel file."""
        df = pd.DataFrame(data_list)
        df["image"] = image_name
        
        file_exists = Path(excel_path).exists()
        with pd.ExcelWriter(excel_path, mode="a" if file_exists else "w", 
                            engine="openpyxl", 
                            if_sheet_exists="overlay" if file_exists else None) as writer:
            df.to_excel(writer, index=False, header=not file_exists)

    @staticmethod
    def plot_histogram_analysis(hist, analysis_results, output_path, title="Histogram Analysis"):
        """Generates and saves the histogram plot with peaks and threshold."""
        if not analysis_results:
            return

        plt.figure(figsize=(10, 6))
        x = np.arange(len(hist))
        y_smooth = analysis_results['smoothed_hist']
        
        plt.plot(x, hist, label="Raw Histogram", color="lightgray", alpha=0.5)
        plt.plot(x, y_smooth, label="Smoothed", color="gray")
        
        # Mark peaks
        peaks = analysis_results['peaks']
        plt.plot(x[peaks], y_smooth[peaks], "ro", label="Modes")
        
        # Mark threshold
        opt_thresh = analysis_results['optimal_threshold']
        plt.axvline(opt_thresh, color="blue", linestyle="--", 
                    label=f"Optimal Threshold = {opt_thresh}")
        
        plt.legend()
        plt.title(title)
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()