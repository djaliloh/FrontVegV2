import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from utilis_fgveg import get_matching_files

def apply_AND_operation(foreground_mask_path, vegetation_mask_path, output_path):
    """
    This function compute intersection between 2 masks (foreground & vegetation masks)
    Autor: Abdoul Djalil
    """
    
    foreground_mask_ = np.array(Image.open(foreground_mask_path))
    vegetation_mask_ = np.array(Image.open(vegetation_mask_path))

    
    if foreground_mask_.shape != vegetation_mask_.shape:
        print(f"Resizing vegetation mask image to match foreground mask dimensions for '{os.path.basename(vegetation_mask_path)}'")
        vegetation_mask_ = cv2.resize(vegetation_mask_, (foreground_mask_.shape[1], foreground_mask_.shape[0]))

    intersection = cv2.bitwise_and(vegetation_mask_, foreground_mask_)
    intersection_image = Image.fromarray(intersection)
    intersection_image.save(output_path)

    return intersection

# Define paths 
foreground_mask_pth = "opt_min-test-outputs/masks"
vegetation_mask_pth = "dinoSam2-outputs_fruits_/masks" 
output_dir = "AND_oper-outputs-test"

foreground_masks, vegetation_masks = get_matching_files(foreground_mask_pth, vegetation_mask_pth) # insure files are the same  
total_length = len(foreground_masks)

# loop over dataset
for cc, (foreground_mask, vegetation_mask) in enumerate(zip(foreground_masks, vegetation_masks)):
    filename1 = os.path.basename(foreground_mask)
    filename2 = os.path.basename(vegetation_mask)

    if filename1 != filename2:
        print(f"Attention : {filename1} et {filename2} ne correspondent pas. Ignoré.")
        continue

    subdirname = os.path.basename(os.path.dirname(foreground_mask))
    output_path = Path(output_dir) / subdirname
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"**AND** oper. between: {filename1} -&- {filename2} || file {cc+1}/{total_length}")
    output_filepath = output_path / filename1
    apply_AND_operation(foreground_mask, vegetation_mask, output_filepath)