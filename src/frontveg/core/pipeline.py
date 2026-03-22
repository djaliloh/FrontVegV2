import os
import cv2 as cv
import numpy as np
from pathlib import Path
from frontveg.models.depth_wrapper import DepthAnythingWrapper
from frontveg.core.hist_processor import HistogramProcessor
from frontveg.core.valley_processor import ValleyProcessor
from frontveg.core.segmenter import BinarySegmenter
from frontveg.utils.reporting import AnalysisReporter

class FrontVegPipeline:
    def __init__(self, config):
        self.config = config
        # Initialize sub-modules
        self.depth_model = DepthAnythingWrapper(
            encoder=config.get('encoder', 'vitl'),
            repo_path=config.get('depth_repo_path') #, 'Depth-Anything-V2'
        )
        self.hist_engine = HistogramProcessor()
        self.valley_engine = ValleyProcessor(
            sigma=config['sigma'],
            peak_dist=config['peak_dist'],
            peak_height=config['peak_height'],
            smooth=config['smoothed']
        )
        self.reporter = AnalysisReporter() 

    def process_folder(self, input_dir, output_root, plot_graphics=False):
        """
        Processes a full folder of images, following your batch normalization logic.
        """
        input_path = Path(input_dir)
        images = list(input_path.glob("*.*")) # Add specific extensions if needed
        
        # 1. Generate raw depth maps and find global max for this folder
        raw_depths = []
        valid_images = []
        global_max = 0
        
        print(f"--> Step 1: Depth Estimation for {input_path.name}")
        for img_p in images:
            # print(f"[DEBUG] Processing {img_p.name}...")
            img = cv.imread(str(img_p))
            if img is None: continue
            
            # print("[DEBUG] PIPELINE: calling depth inference")
            depth = self.depth_model.infer(img)
            raw_depths.append(depth)
            valid_images.append(img_p)
            global_max = max(global_max, depth.max())

            # Save raw depth for reference
            depth_dir = Path(output_root) / "depth" / input_path.name
            depth_dir.mkdir(parents=True, exist_ok=True)
            cv.imwrite(str(depth_dir / img_p.name), depth)

        # 2. Process each image for thresholding
        print(f"--> Step 2: Thresholding and Mask Generation")
        for idx, (depth, img_p) in enumerate(zip(raw_depths, valid_images)):
            # Normalize & Hist
            clean_depth, hist = self.hist_engine.normalize_and_hist(depth, global_max)
            
            # Find Valley
            analysis = self.valley_engine.find_optimal_threshold(hist)
            
            if analysis:
                threshold = analysis['optimal_threshold']
                mask = BinarySegmenter.apply_threshold(clean_depth, threshold)
                
                # Save Mask
                out_dir = Path(output_root) / "masks" / input_path.name
                out_dir.mkdir(parents=True, exist_ok=True)
                cv.imwrite(str(out_dir / img_p.name), mask)
                
                # Optional Plotting
                if plot_graphics:
                    plot_path = Path(output_root) / "graphics" / input_path.name / f"hist_{img_p.stem}.png"
                    self.reporter.plot_histogram_analysis(hist, analysis, plot_path, title=f"Analysis: {img_p.name}")
            else:
                print(f"Warning: No threshold found for {img_p.name}")