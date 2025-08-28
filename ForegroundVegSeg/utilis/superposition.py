import os
import numpy as np
from PIL import Image
from pathlib import Path
from utilis_fgveg import get_matching_files

def image_overlay(RGB_dir, intersection_mask_dir, output_dir):
    """
    This function superpose 2 images (RGB and the intersection mask)
    :inputs --> RGB image and intersection mask (AND operation mask)
    :output --> image of masked RGB
    Autor: Abdoul Djalil
    """
    RGBs, masks = get_matching_files(RGB_dir, intersection_mask_dir) # insure files are the same

    pre_subdir = None
    subdir_cpt = 0
    for cc, (rgb, mask) in enumerate(zip(RGBs, masks)):

        image_name  = os.path.basename(rgb)
        # image_name  = os.path.splitext(image_name)[0]
        subdir_name = os.path.basename(os.path.dirname(rgb))
        

        if subdir_name != pre_subdir:
            subdir_cpt=0
            pre_subdir = subdir_name
        
        subdir_cpt+=1

        subdir_images_size = len(os.listdir(os.path.dirname(rgb)))
        print(f"Image: {subdir_cpt}/{subdir_images_size} in {subdir_name}")


        mask_on_rgb_dir = Path(output_dir) / subdir_name
        mask_on_rgb_dir.mkdir(parents=True, exist_ok=True)

        rgb = np.array(Image.open(rgb))[:, :, :3]
        mask = np.array(Image.open(mask)) 


        mask_for_superposition = np.stack([mask]*3, axis=-1)  # (H, W, 3)
        mask_for_superposition = (mask_for_superposition * 255).astype(np.uint8)

        # Application du masque sur l'image originale
        masked_rgb = (rgb * mask_for_superposition).astype('uint8')
        Image.fromarray(masked_rgb).save(os.path.join(mask_on_rgb_dir, f"{image_name}"))


RGB_pth = "../dataset/test-for-deploy" 
AND_oper_mask_pth = "AND_oper-outputs-test"
output_pth = "superpose-outputs-test" 

image_overlay(RGB_pth, AND_oper_mask_pth, output_pth)
