"""Diagnostic plots of the depth histogram analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure


class AnalysisReporter:
    @staticmethod
    def plot_histogram_analysis(hist, analysis_results, output_path, title="Histogram Analysis"):
        """Save the histogram with its peaks, valleys and selected threshold."""
        if not analysis_results:
            return

        # Figure (not pyplot) is thread-safe and needs no GUI backend.
        fig = Figure(figsize=(10, 6))
        ax = fig.subplots()
        x = np.arange(len(hist))
        y_smooth = analysis_results["smoothed_hist"]
        peaks = analysis_results["peaks"]
        valleys = [v["index"] for v in analysis_results.get("valleys", [])]
        threshold = analysis_results["optimal_threshold"]

        ax.plot(x, hist, label="Raw histogram", color="lightgray")
        ax.plot(x, y_smooth, label="Smoothed", color="gray")
        ax.plot(x[peaks], y_smooth[peaks], "ro", label="Modes (peaks)")
        ax.plot(x[valleys], y_smooth[valleys], "go", label="Valleys (minima)")
        ax.axvline(threshold, color="blue", linestyle="--", label=f"Threshold = {threshold}")
        ax.set(title=title, xlabel="Normalized depth", ylabel="Pixel count")
        ax.legend()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
