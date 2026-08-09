import csv
from pathlib import Path

import numpy as np
import napari
import time
from magicgui import magicgui
from skimage.measure import label, regionprops
from napari.utils.notifications import show_info, show_error
from PIL import Image

import os

# Resolution based on environment variable or marker-based traversal to support packaged distribution
def _has_frontveg_markers(path: Path) -> bool:
    """Returns True if path contains both 'external/' and 'checkpoints/' subdirectories."""
    return (path / "external").is_dir() and (path / "checkpoints").is_dir()


def get_project_root() -> Path:
    # 1. Priority: explicit environment variable set by the user
    if "FRONTVEG_ROOT" in os.environ:
        return Path(os.environ["FRONTVEG_ROOT"]).resolve()

    # 2. Traverse up from the installed package file
    #    Works for editable / development installs where __file__ is inside the repo.
    for parent in Path(__file__).resolve().parents:
        if _has_frontveg_markers(parent):
            return parent

    # 3. Traverse up from the current working directory
    #    Works when the user launches napari from inside or near the repo.
    for parent in Path.cwd().resolve().parents:
        if _has_frontveg_markers(parent):
            return parent
        # Also check direct children of each ancestor (e.g. cwd parent contains FrontVeg2/)
        try:
            for child in parent.iterdir():
                if child.is_dir() and _has_frontveg_markers(child):
                    return child
        except (PermissionError, OSError):
            pass

    # 4. Search common roots on this machine (home, Desktop, Documents, Downloads, …)
    #    Scans 2 levels deep so it finds FrontVeg2 wherever it was placed.
    home = Path.home()
    search_roots = [
        home,
        home / "Desktop",
        home / "Documents",
        home / "Downloads",
        home / "data",
        home / "projects",
        home / "repos",
        home / "workspace",
    ]
    for root in search_roots:
        if not root.is_dir():
            continue
        # Level 1: root itself
        if _has_frontveg_markers(root):
            return root
        # Level 2: direct children of root
        try:
            for child in root.iterdir():
                if child.is_dir() and _has_frontveg_markers(child):
                    return child
        except (PermissionError, OSError):
            continue

    # 5. Last resort: current working directory (user should set FRONTVEG_ROOT)
    return Path.cwd()

_PROJECT_ROOT = get_project_root()

_GLOBAL_STATE = {}

# --- LOADING MODELS ---

def get_fv_pipeline(config):
    if "pipeline" not in _GLOBAL_STATE:
        from frontveg.core.pipeline import FrontVegPipeline
        show_info("Loading Depth-Anything V2...")
        _GLOBAL_STATE["pipeline"] = FrontVegPipeline(config)
    return _GLOBAL_STATE["pipeline"]

def get_sam3_stack(config):
    """Charge SAM3 and the Tiler at once to ensure consistency"""
    if "sam3_inference" not in _GLOBAL_STATE:
        from frontveg.models.sam3_wrapper import SAM3Predictor
        from frontveg.core.sam3_tiler import SAM3TiledInference
        
        show_info("Loading SAM3 ...")
        model = SAM3Predictor(checkpoint_path=config['sam3_ckpt'], repo_path=config['sam3_repo_path'])
        _GLOBAL_STATE["sam3_model"] = model
        _GLOBAL_STATE["sam3_inference"] = SAM3TiledInference(model, tile_size=config['tile_size'], overlap=config['overlap'])
    return _GLOBAL_STATE["sam3_inference"]

def get_postprocessor():
    if "postproc" not in _GLOBAL_STATE:
        from frontveg.core.post_processor import PostProcessor
        _GLOBAL_STATE["postproc"] = PostProcessor()
    return _GLOBAL_STATE["postproc"]

# --- WIDGET NAPARI --- 

# print("DEBUG: _PROJECT_ROOT = ", _PROJECT_ROOT)  

def make_frontveg_widget():
    @magicgui(
        call_button="Run Complete Pipeline",
        prompt={"label": "Object to be segmented", "widget_type": "LineEdit"},
        tile_overlap={"label": "Tile Overlap (SAM3)", "min": 0.1, "max": 0.9, "step": 0.05},
        sigma={"label": "Sigma (Depth)", "min": 0.1, "max": 5.0, "step": 0.1},
        peak_dist={"label": "Peak Distance (Depth)", "min": 1, "max": 20, "step": 0.05},
        auto_save={"label": "Save CSV (statistics)", "widget_type": "CheckBox"},
        save_mask={"label": "Save Mask (PNG)", "widget_type": "CheckBox"},
    )
    def widget(
        image: "napari.layers.Image",
        prompt: str = "leaf",
        tile_overlap: float = 0.5,
        sigma: float = 1.1,
        peak_dist: float = 8.0,
        auto_save: bool = False,
        save_mask: bool = False,

    ) -> napari.types.LayerDataTuple:
        
        if image is None:
            show_error("Open an RGB image first.")
            return
        
        start_time = time.time() # for timing the whole process

        # Configuration — chemins absolus pour fonctionner quel que soit le CWD
        config = {
            'sigma': sigma,
            'encoder': 'vitl',
            'depth_repo_path': os.environ.get('FRONTVEG_DEPTH_REPO', str(_PROJECT_ROOT / 'external' / 'Depth-Anything-V2')),
            'depth_ckpt_path': os.environ.get('FRONTVEG_DEPTH_CKPT_DIR', str(_PROJECT_ROOT / 'checkpoints' / 'depthanything_ckpts')),
            'peak_dist': peak_dist,
            'peak_height': 0.5,
            'smoothed': True,
            'sam3_ckpt': os.environ.get('FRONTVEG_SAM3_CKPT', str(_PROJECT_ROOT / 'checkpoints' / 'sam3_ckpts' / 'sam3.pt')),
            'sam3_repo_path': os.environ.get('FRONTVEG_SAM3_REPO', str(_PROJECT_ROOT / 'external' / 'sam3')),
            'tile_size': 640,
            'overlap': tile_overlap
        }

        # # Check critical paths
        # if not Path(config['depth_repo_path']).exists():
        #     show_error(f"Missing Depth-Anything-V2 at {config['depth_repo_path']}. Use FRONTVEG_DEPTH_REPO env var.")
        #     return
        # if not Path(config['sam3_repo_path']).exists():
        #     show_error(f"Missing SAM3 repo at {config['sam3_repo_path']}. Use FRONTVEG_SAM3_REPO env var.")
        #     return
        # if not Path(config['sam3_ckpt']).exists():
        #     show_error(f"Missing SAM3 checkpoint at {config['sam3_ckpt']}. Use FRONTVEG_SAM3_CKPT env var.")
        #     return

        # Check critical paths
        missing_resources = []

        if not Path(config['depth_repo_path']).exists():
            missing_resources.append(f"- Missing Depth-Anything-V2 at: {config['depth_repo_path']}")

        if not Path(config['sam3_repo_path']).exists():
            missing_resources.append(f"- Missing SAM3 repo at: {config['sam3_repo_path']}")

        if not Path(config['sam3_ckpt']).exists():
            missing_resources.append(f"- Missing SAM3 checkpoint at: {config['sam3_ckpt']}")

        if missing_resources:
            error_msg = (
                "Incomplete FrontVeg setup!\n\n"
                + "\n".join(missing_resources) + "\n\n"
                "Please check the README setup steps:\n"
                "1. Ensure submodules are cloned in 'external/'\n"
                "2. Ensure checkpoints are downloaded in 'checkpoints/'\n"
                "3. Or set the FRONTVEG_ROOT environment variable to your project folder."
            )
            show_error(error_msg)
            return


        try:
            # 1. Access to models (with lazy loading)
            fv_pipe = get_fv_pipeline(config)
            sam3_tiler = get_sam3_stack(config)
            post_proc = get_postprocessor()

            # 2. Napari (Numpy) -> PIL (for SAM3)
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
            final_id_map, final_binary_mask = post_proc.get_final_colored_map(id_map, mask_fv)

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

            # Dossier de sauvegarde — chemin absolu basé sur la racine du projet
            save_dir = _PROJECT_ROOT / "outputs" / "napari_results"
            base_name = getattr(image, "name", "image")

            # Optionally, save statistics to CSV
            if auto_save:
                save_dir.mkdir(parents=True, exist_ok=True)
                csv_path = save_dir / f"{base_name}_{prompt}_stats.csv"

                with open(csv_path, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["ID_Instance", "Count", "Total_Area_Pixels", "Surface_Pixels", "Prompt", "Sigma", "Peak_Dist"])
                    for i, area in enumerate(areas):
                        writer.writerow([i+1, count, int(total_area), area, prompt, sigma, peak_dist])

                show_info(f"CSV exported: {csv_path.name}")
                print(f"[SAVE] CSV → {csv_path}")

            # Optionally, save annotated mask as PNG
            if save_mask:
                save_dir.mkdir(parents=True, exist_ok=True)

                # 1. Masque binaire (blanc = foreground, noir = background)
                #    final_binary_mask est déjà en 0/255 uint8 — prêt à sauvegarder
                binary_path = save_dir / f"{base_name}_{prompt}_mask_binary.png"
                Image.fromarray(final_binary_mask).save(str(binary_path))

                # 2. Masque colorisé par instance (chaque ID = une couleur)
                #    On normalise les IDs 0..max → 0..255 pour appliquer un colormap
                import cv2 as _cv2
                if final_id_map.max() > 0:
                    id_normalized = (final_id_map.astype(np.float32) / final_id_map.max() * 255).astype(np.uint8)
                else:
                    id_normalized = np.zeros_like(final_id_map, dtype=np.uint8)
                colored = _cv2.applyColorMap(id_normalized, _cv2.COLORMAP_TURBO)
                # Background (id=0) → noir
                colored[final_id_map == 0] = 0
                colored_path = save_dir / f"{base_name}_{prompt}_mask_colored.png"
                _cv2.imwrite(str(colored_path), colored)

                show_info(f"Masks saved: {binary_path.name} + {colored_path.name}")
                print(f"[SAVE] Binary mask  → {binary_path}")
                print(f"[SAVE] Colored mask → {colored_path}")

            show_info("Success !")
            elapsed = time.time() - start_time
            show_info(f"Finished in {elapsed:.2f} seconds !")
            return (final_id_map, {"name": f"Result_{prompt}", "opacity": 0.8}, "labels")


        except Exception as e:
            show_error(f"Crash: {str(e)}")
            import traceback
            traceback.print_exc()

    return widget




# LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGL.so.1 napari -v 