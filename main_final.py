import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent / "src"))  

# /home/adjalil/Working/FrontVegetation/data_t_pipeline
# /home/adjalil/Working/FrontVegetation/outputs_t_pipeline

# our modules
from frontveg.core.pipeline import FrontVegPipeline
from frontveg.models.sam3_wrapper import SAM3Predictor
from frontveg.core.sam3_tiler import SAM3TiledInference
from frontveg.core.post_processor import PostProcessor
from frontveg.utils.visualization import Visualizer

def main():
    parser = argparse.ArgumentParser(description="Master Pipeline: FrontVeg + SAM3")
    
    # Paths
    parser.add_argument("--input", type=str, required=True, help="Path to raw RGB images")
    parser.add_argument("--output", type=str, default="output_final", help="Final output directory")

    # DepthAnything Params
    parser.add_argument("--encoder", type=str, default="vitl")
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
        
        frontveg_pipe.process_folder(str(subdir), output_root / "temp_frontveg")
        
        images = list(subdir.glob("*.[jJ][pP][gG]")) + list(subdir.glob("*.[pP][nN][gG]"))
        
        for img_p in images:
                    print(f"--> Step 3 [Fusing]: {img_p.name}")
                    
        
                    pil_img = Image.open(img_p).convert("RGB")
                    rgb_np = np.array(pil_img)
                    
                    # Get FrontVeg Mask (Mask1)
                    fv_mask_p = output_root / "temp_frontveg" / "masks" / subdir.name / img_p.name
                    if not fv_mask_p.exists(): continue
                    mask_fv = cv2.imread(str(fv_mask_p), cv2.IMREAD_GRAYSCALE)

                    #  Get SAM3 ID Map
                    id_map_raw, _ = sam3_tiler.process(pil_img, prompt=args.prompt)
                    if id_map_raw is None: 
                        print(f"    ! SAM3 found nothing for {img_p.name}")
                        continue

                    final_id_map, final_mask = post_proc.get_final_colored_map(id_map_raw, mask_fv)

                    final_rgb_cutout = post_proc.apply_overlay(rgb_np, final_mask)
                    colored_overlay = Visualizer.create_overlay(final_rgb_cutout, final_id_map, alpha=0.5)

                    colored_overlay_phenoweek = Visualizer.create_overlay(rgb_np, final_id_map, alpha=0.5)

                    # output paths
                    paths = {
                        "sam3_overlay": output_root / "sam3_overlay" / subdir.name,
                        "rgb_cutout": output_root / "rgb_overlay_finalmask" / subdir.name,
                        "masks": output_root / "final_masks" / subdir.name,
                        "id_maps": output_root / "final_id_maps" / subdir.name,
                        "sam3_on_input_rgb": output_root / "sam3_overlay_input_rgb" / subdir.name
                    }
                    
                    for d in paths.values(): d.mkdir(parents=True, exist_ok=True)

                    cv2.imwrite(str(paths["sam3_overlay"] / img_p.name), cv2.cvtColor(colored_overlay, cv2.COLOR_RGB2BGR)) 

                    cv2.imwrite(str(paths["sam3_on_input_rgb"] / img_p.name), colored_overlay_phenoweek)
                    
                    cv2.imwrite(str(paths["rgb_cutout"] / img_p.name), cv2.cvtColor(final_rgb_cutout, cv2.COLOR_RGB2BGR))
                    
                    cv2.imwrite(str(paths["masks"] / img_p.name), final_mask)
                    
                    cv2.imwrite(str(paths["id_maps"] / img_p.name), final_id_map.astype(np.uint8))
         

    print(f"\n[DONE] Pipeline finished. Results are in: {args.output}")

if __name__ == "__main__":
    main()



# python main_final.py \
# --input ./data/leaf \
# --output ./results\
# --sam3_ckpt ./checkpoints/sam3_ckpts/sam3.pt \
# --prompt "leaf" \
# --peak_dist 1 \
# --sigma 1.1 \
# --smoothed