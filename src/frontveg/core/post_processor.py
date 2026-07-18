import cv2
import numpy as np
from PIL import Image

class PostProcessor:
    """
    Handles logical operations between masks and visual compositions.
    """
    
    @staticmethod
    def intersect_masks(mask1, mask2):
        """
        Computes the bitwise AND between two masks. 
        Resizes mask2 to match mask1 if dimensions differ.
        """
        if mask1.shape != mask2.shape:
            # Resize mask2 to match mask1 dimensions (Width, Height)
            mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Ensure both are binary uint8 (0 or 255)
        m1 = (mask1 > 127).astype(np.uint8) * 255
        m2 = (mask2 > 127).astype(np.uint8) * 255
        
        return cv2.bitwise_and(m1, m2)

    @staticmethod
    def apply_overlay(rgb_image, mask):
        """
        Applies a binary mask to an RGB image (Blackout everything outside the mask).
        """
        # Ensure mask is 3-channel for multiplication
        # 1. Convert mask to 0.0 or 1.0 (float32) for multiplication
        # THE FIX: We don't divide by 255 if we use the boolean test directly
        if len(mask.shape) == 2:
            binary_mask = (mask > 127).astype(np.float32) #/ 255.0
            mask_3d = np.stack([binary_mask] * 3, axis=-1)
        else:
            mask_3d = mask.astype(np.float32) #/ 255.0 if mask is already 0 or 255 \ # /!\ Carreful check this logic 

        # Multiply RGB by mask (0 or 1)
        masked_rgb = (rgb_image.astype(np.float32) * mask_3d).astype(np.uint8)
        return masked_rgb
    

    @staticmethod
    def get_final_colored_map(id_map_sam, mask_fv):
        """
        Logic: 
        1. Convert SAM ID Map to binary (Mask_SAM)
        2. Intersection = Mask_SAM AND Mask_FV
        3. Clean ID Map = Keep SAM IDs only inside the Intersection
        """
        # 1. SAM Binary Mask
        mask_sam = (id_map_sam > 0).astype(np.uint8) * 255
        
        # Resize FrontVeg mask if needed
        if mask_fv.shape != mask_sam.shape:
            mask_fv = cv2.resize(mask_fv, (mask_sam.shape[1], mask_sam.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # 2. Final Binary Intersection (The "True Foreground")
        final_binary_mask = cv2.bitwise_and(mask_sam, (mask_fv > 127).astype(np.uint8) * 255)
        
        # 3. Clean ID Map
        # We only keep the ID if it's inside our strict intersection
        final_id_map = np.where(final_binary_mask > 0, id_map_sam, 0).astype(np.uint16)
        
        return final_id_map, final_binary_mask


    @staticmethod
    def filter_id_map(id_map, frontveg_mask):
        """
        Keeps SAM3 instance IDs only where FrontVeg mask is white.
        """
        if id_map.shape != frontveg_mask.shape:
            frontveg_mask = cv2.resize(frontveg_mask, (id_map.shape[1], id_map.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # We keep the ID if frontveg_mask > 127, else we set it to 0
        filtered_id_map = np.where(frontveg_mask > 127, id_map, 0)
        return filtered_id_map.astype(np.uint16)

