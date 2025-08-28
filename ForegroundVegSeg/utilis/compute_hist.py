import os
import glob
import cv2 as cv
import numpy as np
import argparse

def individual_histogram(depth_maps_dir="./Depth-Anything-V2/depth_maps"):
    """
    Compute individual histograms from depth maps.
    Author: Abdoul Djalil Ousseini Hamza
    """
    depthImg = sorted(glob.glob(os.path.join(depth_maps_dir, "*", "*")))
    total_depth_image = len(depthImg)

    depth_imgs = []
    individual_histograms = []
    global_max = float('-inf')

    for idx, depth_path in enumerate(depthImg):
        subfold = os.path.basename(os.path.dirname(depth_path))
        print(f"Processing {subfold.capitalize()} {idx + 1}/{total_depth_image}: {depth_path}")

        depth = cv.imread(depth_path, cv.IMREAD_ANYDEPTH)
        if depth is None:
            print(f"Erreur : impossible to read {depth_path}")
            continue

        global_max = max(global_max, depth.max())
        depth_filtered = np.where(depth > 0, depth, np.nan)
        depth_normalized = (depth_filtered / global_max * 255).astype('float32')
        depth_imgs.append(np.nan_to_num(depth_normalized))

        valid_pixels = depth_normalized[~np.isnan(depth_normalized)].astype('uint8')
        histogram = cv.calcHist([valid_pixels], [0], None, [256], [0, 256])
        # histogram = histogram.astype('uint8') # convert hist to uint8
        individual_histograms.append(histogram.flatten())

    return depth_imgs, individual_histograms

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Compute individual histograms from depth maps.")
#     parser.add_argument("--depth_maps_dir", type=str, default="./Depth-Anything-V2/depth_maps",
#                         help="Path to the directory containing depth map images")
    
#     args = parser.parse_args()

#     depth_imgs, individual_histograms = individual_histogram(args.depth_maps_dir)
#     print(f"\n Traitement terminé : {len(individual_histograms)} histogrammes générés.")
