import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent / "src"))  # Pour eviter de faire ""PYTHONPATH=src python -m frontveg.main_final

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
    config = vars(args) # Convert argparse Namespace to dict for easier passing to classes
    
    # Load Models
    print(">>> Loading FrontVeg (Depth) and SAM3 models...")
    frontveg_pipe = FrontVegPipeline(config)
    sam3_model = SAM3Predictor(checkpoint_path=args.sam3_ckpt)
    sam3_tiler = SAM3TiledInference(sam3_model, tile_size=args.tile_size)
    post_proc = PostProcessor()

    # Prep folders
    input_path = Path(args.input)
    output_root = Path(args.output)
    
    # On détecte les sous-dossiers (ton amélioration)
    subdirs = [p for p in input_path.iterdir() if p.is_dir()]
    if not subdirs: subdirs = [input_path]

    # --- 2. EXECUTION ---
    for subdir in subdirs:
        print(f"\n--- Processing Subdirectory: {subdir.name} ---")
        
        # A. Run FrontVeg on the whole folder (for batch normalization)
        # This creates masks in output_root/frontveg/masks/subdir.name
        frontveg_pipe.process_folder(str(subdir), output_root / "temp_frontveg")
        
        # B. Loop for SAM3 + Intersection + Overlay
        images = list(subdir.glob("*.[jJ][pP][gG]")) + list(subdir.glob("*.[pP][nN][gG]"))
        
        for img_p in images:
            print(f"  > Fusing: {img_p.name}")
            
            # 1. Load Original RGB
            pil_img = Image.open(img_p).convert("RGB")
            rgb_np = np.array(pil_img)
            
            # 2. Get FrontVeg Mask (Mask1)
            # We fetch it from the temp directory created just before
            fv_mask_p = output_root / "temp_frontveg" / "masks" / subdir.name / img_p.name
            if not fv_mask_p.exists(): continue
            mask1 = cv2.imread(str(fv_mask_p), cv2.IMREAD_GRAYSCALE)

            # 3. Get SAM3 Mask (Mask2)
            # SAM3 generates an ID Map, we convert it to binary
            id_map, _ = sam3_tiler.process(pil_img, prompt=args.prompt)
            if id_map is None: 
                print(f"    ! SAM3 found nothing for {img_p.name}")
                continue
            mask2 = (id_map > 0).astype(np.uint8) * 255

            # # Filter the ID map to keep only regions where FrontVeg mask is white
            # # . FILTRAGE ET INTERSECTION
            # final_id_map = post_proc.filter_id_map(id_map, mask1)
            # # . GÉNÉRATION DE L'OVERLAY COLORÉ
            # # rendu "SAM3-style" propre
            # colored_final_overlay = Visualizer.create_overlay(rgb_np, final_id_map, alpha=0.5)

            final_id_map, final_binary_mask = post_proc.get_final_colored_map(id_map, mask1)

            # 4. Logical Intersection (AND)
            final_mask = post_proc.intersect_masks(mask1, mask2)

            # 5. Final Overlay
            final_result = post_proc.apply_overlay(rgb_np, final_mask)


            # --- 3. SAVING ---
            color_sam3_overlay_path = output_root / "sam3_overlay" / subdir.name
            final_out_dir = output_root / "final_results" / subdir.name
            mask_out_dir = output_root / "final_masks" / subdir.name
            
            for d in [final_out_dir, mask_out_dir, color_sam3_overlay_path]: d.mkdir(parents=True, exist_ok=True)

            # Save colored SAM3 overlay for visualization
            cv2.imwrite(str(color_sam3_overlay_path / img_p.name), cv2.cvtColor(final_id_map, cv2.COLOR_RGB2BGR)) 
            # Save final image (BGR for OpenCV)
            cv2.imwrite(str(final_out_dir / img_p.name), cv2.cvtColor(final_result, cv2.COLOR_RGB2BGR))
            # Save final mask
            cv2.imwrite(str(mask_out_dir / img_p.name), final_mask)
         

    print(f"\n[DONE] Pipeline finished. Results are in: {args.output}")

if __name__ == "__main__":
    main()