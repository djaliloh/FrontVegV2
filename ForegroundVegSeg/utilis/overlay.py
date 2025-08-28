import os
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from utilis_fgveg import get_matching_files



def overlay(rgb, mask, mask_on_rgb_dir, image_name):

    rgb = np.array(Image.open(rgb))[:, :, :3]
    mask = np.array(Image.open(mask)) 

    mask_for_superposition = np.stack([mask]*3, axis=-1)  # (H, W, 3)
    mask_for_superposition = (mask_for_superposition * 255).astype(np.uint8)

    # Application of the mask on the original image
    masked_rgb = (rgb * mask_for_superposition).astype('uint8')

    mask_on_rgb_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(masked_rgb).save(mask_on_rgb_dir / image_name)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Intersection between foreground mask (fgm) and foreground vegetation mask (fgvm)")

    parser.add_argument("--RGB_path", type=str, default="dataset", required=True, help="path to RGB images")
    parser.add_argument("--AND_path", type=str, default="AND_oper-outputs/green_foliage__fruits_/masks", help="path to intersection images masks")
    parser.add_argument("--output_path", type=str, default="Superpose-outputs/green_foliage__fruits_", help="path to save output images")
    parser.add_argument("--text_prompts", type=str, nargs='+', default=["green foliage . fruits ."])

    args = parser.parse_args()


    for text_prompt in args.text_prompts:
        
        prompt_clean = text_prompt.replace(" ", "_").replace(".", "").replace(",", "") # clean the prompt text
        rgb_path = args.RGB_path 
        and_path = Path(args.AND_path) / f"{prompt_clean}"
        
        RGBs, masks = get_matching_files(rgb_path, and_path) # insure files are the same


        pre_subdir = None
        subdir_cpt = 0
        for cc, (rgb, mask) in enumerate(zip(RGBs, masks)):

            # image_name  = os.path.basename(rgb)
            # # image_name  = os.path.splitext(image_name)[0]
            # subdir_name = os.path.basename(os.path.dirname(rgb))

            image_name = Path(rgb).name
            subdir_name = Path(rgb).parent.name

            if subdir_name != pre_subdir:
                subdir_cpt=0
                pre_subdir = subdir_name
            
            subdir_cpt+=1

            subdir_images_size = len(os.listdir(os.path.dirname(rgb)))
            print(f"Image: {subdir_cpt}/{subdir_images_size} in {prompt_clean}/{subdir_name}")


            mask_on_rgb_dir = Path(args.output_path) / prompt_clean / subdir_name
            # mask_on_rgb_dir.mkdir(parents=True, exist_ok=True)
    
            overlay(rgb, mask, mask_on_rgb_dir, image_name)