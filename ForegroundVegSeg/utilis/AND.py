import argparse
import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from utilis_fgveg import get_matching_files


def apply_AND_operation(foreground_mask_path, vegetation_mask_path, output_path):
    foreground_mask_ = np.array(Image.open(foreground_mask_path))
    vegetation_mask_ = np.array(Image.open(vegetation_mask_path))
    
    if foreground_mask_.shape != vegetation_mask_.shape:
        print(f"Resizing vegetation mask image to match foreground mask dimensions for '{os.path.basename(vegetation_mask_path)}'")
        vegetation_mask_ = cv2.resize(vegetation_mask_, (foreground_mask_.shape[1], foreground_mask_.shape[0]))

    intersection = cv2.bitwise_and(vegetation_mask_, foreground_mask_)
    intersection_image = Image.fromarray(intersection)
    intersection_image.save(output_path)

    return intersection


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Intersection between foreground mask (fgm) and foreground vegetation mask (fgvm)")

    parser.add_argument("--fgm_path", type=str, default="opt_min-outputs/masks", required=True, help="path to foreground mask")
    parser.add_argument("--fgvm_path", type=str, default="dinoSam2-outputs/green_foliage__fruits_/masks", help="path to foreground vegetation mask")
    parser.add_argument("--output_path", type=str, default="AND_oper-outputs/green_foliage__fruits_", help="path to save output images")
    parser.add_argument("--text_prompts", type=str, nargs='+', default=["green foliage . fruits ."])

    args = parser.parse_args()

    for text_prompt in args.text_prompts:
        
        prompt_clean = text_prompt.replace(" ", "_").replace(".", "").replace(",", "") # clean the prompt text
        # Define paths 
        foreground_mask_pth = Path(args.fgm_path) / "masks" 
        vegetation_mask_pth = Path(args.fgvm_path) / f"{prompt_clean}" / "masks" 
        
        output_pth = Path(args.output_path) / f"{prompt_clean}" #output path
        output_pth.mkdir(parents=True, exist_ok=True)


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
            output_path = Path(output_pth) / subdirname
            output_path.mkdir(parents=True, exist_ok=True)

            # print(filename1)
            # print(subdirname)
            # break
            print(f"**AND** oper. between: {filename1} -&- {filename2} || file {cc+1}/{total_length}")
            output_filepath = output_path / filename1
            apply_AND_operation(foreground_mask, vegetation_mask, output_filepath)