import os
import glob
import cv2
from pathlib import Path
from valley_search import analyze_histogram_detect_valley
from compute_hist import individual_histogram

def thresholding(depth_maps_dir: str, output_method: str = "opt_min"):
# def thresholding(prms: str):
    # depth_maps_dir = prms["depth_maps_dir"]
    # output_method = prms =["output_method"]
    """
    This perform a binary threshold in estimated depth images
    Autor: @Abdoul Djalil Ousseini Hamza

    Args:
        depth_maps_dir (str): Chemin vers le dossier contenant les images de profondeur.
        output_method (str): Nom de la méthode utilisée (pour créer le dossier de sortie).
    """

    depth_maps = sorted(glob.glob(os.path.join(depth_maps_dir, "*", "*")))
    _, individual_hist = individual_histogram(depth_maps_dir)

    output_dir = Path(f"{output_method}-outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    global_max = float('-inf')

    for idx, (histogram, depth_map_path) in enumerate(zip(individual_hist, depth_maps)):
        subfolder = os.path.basename(os.path.dirname(depth_map_path))
        print(f"Processing {subfolder.capitalize()}: {idx + 1}/{len(depth_maps)} | {depth_map_path}")

        depth = cv2.imread(depth_map_path, cv2.IMREAD_ANYDEPTH)
        if depth is None:
            print(f"Error: Cannot load {depth_map_path}")
            continue
        
        # Normalization
        global_max = max(global_max, depth.max())
        depth_norm = (depth / global_max * 255).astype('uint8')

        # Analyse de l'histogramme
        _, optimal_minimum_index = analyze_histogram_detect_valley(histogram, idx, label="Individual")

        if optimal_minimum_index is not None:
            _, binary_image = cv2.threshold(depth_norm, optimal_minimum_index, 255, cv2.THRESH_BINARY)

            # Output dir
            mask_output_dir = output_dir / "masks" / subfolder
            mask_output_dir.mkdir(parents=True, exist_ok=True)

            # Sauvegarde du masque
            output_filename = os.path.basename(depth_map_path)
            binary_image_path = mask_output_dir / output_filename
            cv2.imwrite(str(binary_image_path), binary_image)

            print(f"Image binarisée sauvegardée : {binary_image_path}")
        else:
            print(f"Aucun seuil optimal trouvé pour {depth_map_path}, image ignorée.")

if __name__ == "__main__":
    # prameters = {
    #     "depth_maps_dir": "Depth-Anything-V2/depth_maps",
    #     "output_method": "opt_min"
    # }  /home/utilisateur/Bureau/segmentation/ForegroundVegSeg/utilis/tresholding.py
    _depth_map_path = "Depth-Anything-V2/depth_maps"
    thresholding(_depth_map_path)

