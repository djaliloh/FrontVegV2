import csv
from pathlib import Path

import numpy as np
import napari
import time
from magicgui import magicgui
from skimage.measure import label, regionprops
from napari.utils.notifications import show_info, show_error
from PIL import Image

_GLOBAL_STATE = {}

# --- LOADING MODELS ---

def get_fv_pipeline(config):
    if "pipeline" not in _GLOBAL_STATE:
        from frontveg.core.pipeline import FrontVegPipeline
        show_info("Chargement de Depth-Anything V2...")
        _GLOBAL_STATE["pipeline"] = FrontVegPipeline(config)
    return _GLOBAL_STATE["pipeline"]

def get_sam3_stack(config):
    """Charge SAM3 and the Tiler at once to ensure consistencye"""
    if "sam3_inference" not in _GLOBAL_STATE:
        from frontveg.models.sam3_wrapper import SAM3Predictor
        from frontveg.core.sam3_tiler import SAM3TiledInference
        
        show_info("Loading SAM3 ...")
        model = SAM3Predictor(checkpoint_path=config['sam3_ckpt'])
        _GLOBAL_STATE["sam3_model"] = model
        _GLOBAL_STATE["sam3_inference"] = SAM3TiledInference(model, tile_size=config['tile_size'], overlap=config['overlap'])
    return _GLOBAL_STATE["sam3_inference"]

def get_postprocessor():
    if "postproc" not in _GLOBAL_STATE:
        from frontveg.core.post_processor import PostProcessor
        _GLOBAL_STATE["postproc"] = PostProcessor()
    return _GLOBAL_STATE["postproc"]

# --- WIDGET NAPARI ---

def make_frontveg_widget():
    @magicgui(
        call_button="Run Complete Pipeline",
        prompt={"label": "Object to be segmented", "widget_type": "LineEdit"}, 
        tile_overlap={"label": "Tile Overlap (SAM3)", "min": 0.1, "max": 0.9, "step": 0.05},
        sigma={"label": "Sigma (Depth)", "min": 0.1, "max": 5.0, "step": 0.1},
        peak_dist={"label": "Peak Distance (Depth)", "min": 1, "max": 20, "step": 0.05},
        auto_save={"label": "Save CSV - Masks", "widget_type": "CheckBox"},
    )
    def widget(
        image: "napari.layers.Image", 
        prompt: str = "leaf",
        tile_overlap: float = 0.3,
        sigma: float = 1.1,
        peak_dist: float = 1.0,
        auto_save: bool = False,

    ) -> napari.types.LayerDataTuple:
        
        if image is None:
            show_error("Open an RGB image first.")
            return
        
        start_time = time.time() # for timing the whole process

        # Configuration 
        config = {
            'sigma': sigma, 
            'encoder': 'vitl', 
            'depth_repo_path': 'external/Depth-Anything-V2',
            'peak_dist': 1.0, 
            'peak_height': 0.5, 
            'smoothed': True, 
            'sam3_ckpt': 'checkpoints/sam3_ckpts/sam3.pt', 
            'tile_size': 640,
            'overlap': tile_overlap
        }

        try:
            # 1. Access to models (with lazy loading)
            fv_pipe = get_fv_pipeline(config)
            sam3_tiler = get_sam3_stack(config)
            post_proc = get_postprocessor()

            # 2. Napari (Numpy) -> PIL (pour SAM3)
            img_np = np.array(image.data)
            pil_img = Image.fromarray(img_np.astype('uint8'))

            # 3. Step 1: FrontVeg Mask (Geometry)
            show_info("Step 1: Depth analysis...")
            mask_fv = fv_pipe.predict(img_np, sigma=sigma)

            # 4. Step 2: SAM3 (Semantic)
            show_info(f"Step 2: SAM3 Segmentation ({prompt})...")
            id_map, _ = sam3_tiler.process(pil_img, prompt=prompt)

            if id_map is None:
                show_error("SAM3 detected nothing.")
                return

            # 5. Step 3: Fusion & Post-Processing
            show_info("Step 3: Mask Fusion...")
            final_id_map, _ = post_proc.get_final_colored_map(id_map, mask_fv)

            # --- Instance Analysis ---
            # We label each distinct object
            labeled_map = label(final_id_map > 0)
            regions = regionprops(labeled_map)

            # We extract the surface areas of each object (in pixels)
            areas = [r.area for r in regions]
            # areas = [r.area for r in regions if r.area > 500] # Filter small objects

            if len(areas) == 0:
                show_info("No objects detected.")
                return
            
            # --- Compute statistics ---
            count = len(areas)
            total_area = np.sum(areas)
            mean_area = np.mean(areas)
            std_area = np.std(areas)

            # Display stats in Napari and console
            stats_msg = (
                f"Results ({prompt}):\n"
                f"Count: {count}\n"
                f"Mean: {mean_area:.1f} px\n"
                f"Std Dev: {std_area:.1f} px"
            )
            show_info(stats_msg)
            print(f"\n--- DETAILED STATISTICS ---\n{stats_msg}\n")

            # Optionally, save results to CSV
            if auto_save:
                save_dir = Path("outputs/napari_results")
                save_dir.mkdir(parents=True, exist_ok=True)
                
                base_name = getattr(image, "name", "image")
                csv_path = save_dir / f"{base_name}_{prompt}_stats.csv"
                
                with open(csv_path, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["ID_Instance", "Count", "Surface_Pixels", "Prompt", "Sigma"])
                    for i, area in enumerate(areas):
                        writer.writerow([i+1, count, area, prompt, sigma])
                
                show_info(f"Statistics exported: {csv_path.name}")

            show_info("Success !")
            elapsed = time.time() - start_time 
            show_info(f"Finished in {elapsed:.2f} seconds !") # time in seconds
            return (final_id_map, {"name": f"Result_{prompt}", "opacity": 0.8}, "labels")


        except Exception as e:
            show_error(f"Crash: {str(e)}")
            import traceback
            traceback.print_exc()

    return widget




# LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGL.so.1 napari -v 