from pathlib import Path
import cv2
from frontveg.core.pipeline import FrontVegPipeline
from frontveg.core.sam3_tiler import SAM3TiledInference
from frontveg.core.post_processor import PostProcessor

class MasterPipeline:
    def __init__(self, config):
        self.frontveg = FrontVegPipeline(config)
        self.sam3 = SAM3TiledInference(config['sam3_predictor']) 
        self.processor = PostProcessor()

    def run_full_stack(self, rgb_dir, output_dir, prompt):
        """
        Executes the complete workflow.
        """
        out_root = Path(output_dir)
        
        # 1. Run FrontVeg (generates Mask1)
        print("--- Running FrontVeg ---")
        self.frontveg.process_folder(rgb_dir, out_root / "frontveg")
        
        # 2. Run SAM3 (generates Mask2/ID Map)
        # Note: In a real 'Pro' setup, we'd pass the same loaded image to both
        # to avoid re-reading from disk.
        
        # 3. Intersection & Final Overlay
        print("--- Finalizing Intersection and Overlay ---")
        mask1_dir = out_root / "frontveg" / "masks"
        
        # We'll need a loop here that pairs RGB, Mask1 (FrontVeg) and Mask2 (SAM3)
        # For simplicity in this example, let's assume Mask1 and Mask2 are generated.
        # [Logique de boucle utilisant sync_directories ici...]