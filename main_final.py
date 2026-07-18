import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import warnings
import time

sys.path.append(str(Path(__file__).resolve().parent / "src"))  

# /home/adjalil/Working/FrontVegetation/data_t_pipeline
# /home/adjalil/Working/FrontVegetation/outputs_t_pipeline

# our modules
from frontveg.core.pipeline import FrontVegPipeline
from frontveg.models.sam3_wrapper import SAM3Predictor
from frontveg.core.sam3_tiler import SAM3TiledInference
from frontveg.core.post_processor import PostProcessor
from frontveg.utils.visualization import Visualizer
from frontveg.utils.image_loader import ImageLoader, get_image_files, convert_output_filename

def main():
    parser = argparse.ArgumentParser(description="Master Pipeline: FrontVeg + SAM3")
    
    # Paths
    parser.add_argument("--input", type=str, required=True, help="Path to raw RGB images")
    parser.add_argument("--output", type=str, default="output_final", help="Final output directory")

    # DepthAnything Params
    parser.add_argument("--encoder", type=str, default="vitl")
    parser.add_argument("--depth_ckpt_path", type=str, default="checkpoints/depthanything_ckpts", help="Path to DepthAnything checkpoints")
    parser.add_argument("--depth_repo_path", type=str, default="external/Depth-Anything-V2", help="Path to Depth Anything repository")
    
    # FrontVeg Params
    parser.add_argument("--sigma", type=float, default=1.1)
    parser.add_argument("--peak_dist", type=int, default=20)
    parser.add_argument("--peak_height", type=float, default=0.5)
    parser.add_argument("--smoothed", action="store_true")

    # SAM3 Params
    parser.add_argument("--sam3_ckpt", type=str, required=True, help="Path to SAM3 .pt file")
    parser.add_argument("--prompt", type=str, default="leaf", help="Text prompt for SAM3")
    parser.add_argument("--tile_size", type=int, default=640)
    
    args = parser.parse_args()

    # --- 1. INITIALIZATION ---
    config = vars(args) 
    
    # Load Models
    print(">>> Loading FrontVeg (Depth) and SAM3 models...")
    
    start = time.time()

    frontveg_pipe = FrontVegPipeline(config)
    sam3_model = SAM3Predictor(checkpoint_path=args.sam3_ckpt)
    sam3_tiler = SAM3TiledInference(sam3_model, tile_size=args.tile_size)
    post_proc = PostProcessor()

    input_path = Path(args.input)
    output_root = Path(args.output)
    
    subdirs = [p for p in input_path.iterdir() if p.is_dir()]
    if not subdirs: subdirs = [input_path]

    # --- 2. EXECUTION ---
    for subdir in subdirs:
        print(f"\n--- Processing Subdirectory: {subdir.name} ---")
        
        frontveg_pipe.process_folder(str(subdir), output_root / "temp_frontveg", plot_graphics=True)
        
        # Get all supported image formats (PNG, JPG, TIF, TIFF, CR3, etc.)
        images = get_image_files(str(subdir), supported_only=True)
        
        if not images:
            print(f"    ! No supported images found in {subdir.name}")
            continue
        
        for img_p in images:
                    try:
                        print(f"--> Step 3 [Fusing]: {img_p.name}")
                        
                        # Load image with automatic format detection
                        rgb_np = ImageLoader.load_image(str(img_p), return_format="rgb")
                        
                        if rgb_np is None:
                            print(f"    ! Failed to load image: {img_p.name}")
                            continue
                        
                        if len(rgb_np.shape) != 3 or rgb_np.shape[2] not in [3, 4]:
                            print(f"    ! Invalid image format (not RGB): {img_p.name}")
                            continue
                        
                        # Get FrontVeg Mask (Mask1)
                        fv_mask_p = output_root / "temp_frontveg" / "masks" / subdir.name / img_p.name
                        if not fv_mask_p.exists(): 
                            print(f"    ! FrontVeg mask not found: {fv_mask_p.name}")
                            continue
                        mask_fv = cv2.imread(str(fv_mask_p), cv2.IMREAD_GRAYSCALE)
                        
                        if mask_fv is None:
                            print(f"    ! Failed to load FrontVeg mask: {img_p.name}")
                            continue

                        #  Get SAM3 ID Map
                        pil_img = ImageLoader.load_pil_image(str(img_p))
                        if pil_img is None:
                            print(f"    ! Failed to load image for SAM3: {img_p.name}")
                            continue
                        
                        id_map_raw, _ = sam3_tiler.process(pil_img, prompt=args.prompt)
                        if id_map_raw is None: 
                            print(f"    ! SAM3 found nothing for {img_p.name}, generating blank outputs.")
                            id_map_raw = np.zeros(rgb_np.shape[:2], dtype=np.int32)

                        final_id_map, final_mask = post_proc.get_final_colored_map(id_map_raw, mask_fv)

                        final_rgb_cutout = post_proc.apply_overlay(rgb_np, final_mask)
                        colored_overlay = Visualizer.create_overlay(final_rgb_cutout, final_id_map, alpha=0.5)

                        colored_finalid_overlay_rgb = Visualizer.create_overlay(rgb_np, final_id_map, alpha=0.5)

                        # output paths
                        paths = {
                            "sam3_overlay": output_root / "sam3_overlay" / subdir.name,
                            # "rgb_cutout": output_root / "rgb_overlay_finalmask" / subdir.name,
                            "masks": output_root / "final_masks" / subdir.name,
                            # "id_maps": output_root / "final_id_maps" / subdir.name,
                            "colored_finalid_on_input_rgb": output_root / "colored_finalid_overlay_input_rgb" / subdir.name
                        }
                        
                        for d in paths.values(): d.mkdir(parents=True, exist_ok=True)

                        # Convert CR3/RAW filenames to PNG for output (OpenCV can't write RAW formats)
                        output_filename = convert_output_filename(img_p.name, target_format="png")

                        cv2.imwrite(str(paths["sam3_overlay"] / output_filename), cv2.cvtColor(colored_overlay, cv2.COLOR_RGB2BGR)) 
                        cv2.imwrite(str(paths["colored_finalid_on_input_rgb"] / output_filename), colored_finalid_overlay_rgb)
                        # cv2.imwrite(str(paths["rgb_cutout"] / output_filename), cv2.cvtColor(final_rgb_cutout, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(str(paths["masks"] / output_filename), final_mask)
                        # cv2.imwrite(str(paths["id_maps"] / output_filename), final_id_map.astype(np.uint8))
                        
                    except Exception as e:
                        print(f"    ! Error processing {img_p.name}: {str(e)}")
                        continue
         
    end = time.time()
    elapsed = end - start
    print(f"\n  Total Execution Time: {elapsed:.2f} seconds = {elapsed/60:.2f} minutes")
    print(f"\n[DONE] Pipeline finished. Results are in: {args.output}")

if __name__ == "__main__":
    main()



# python main_final.py \
# --input ./data/leaf \
# --output ./results\
# --sam3_ckpt ./checkpoints/sam3_ckpts/sam3.pt \
# --prompt "leaf" \
# --peak_dist 1 \ (peak_dist=8-10 OK; peak_dist=1-7 sur segmentation)  
# --sigma 1.1 \
# --smoothed

# arena ai 
# battle ai
# https://www.facebook.com/reel/1390142589587261



# python main_final.py --input F:\EXPERIMENTS-ECCV-annot_code\annotaion-labelbox\vine\fruit\images --output F:\EXPERIMENTS-ECCV-annot_code\annotaion-labelbox\vine\fruit\fvg_predictions --sam3_ckpt ./checkpoints/sam3_ckpts/sam3.pt --prompt "grapes" --peak_dist 8  --sigma 1.1 --smoothed