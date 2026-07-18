import cv2
import numpy as np
import os

class Visualizer:
    @staticmethod
    def create_overlay(image_rgb, id_map, alpha=0.5):
        """Create an image using translucent colored masks."""
        # Convert PIL or RGB to BGR for OpenCV
        if image_rgb.shape[2] == 3:
            canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        else:
            canvas = image_rgb.copy()
            
        overlay = canvas.copy()
        unique_ids = np.unique(id_map)
        
        for obj_id in unique_ids:
            if obj_id == 0: continue # Fond
            
            # Deterministic random color based on ID
            np.random.seed(obj_id)
            color = np.random.randint(50, 255, size=3).tolist()
            
            mask = (id_map == obj_id)
            overlay[mask] = color
            
        # Image merging
        combined = cv2.addWeighted(canvas, 1 - alpha, overlay, alpha, 0)
        
        # Adding white outlines for a professional finish
        # Using the global binary mask for the outlines
        binary_mask = (id_map > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(combined, contours, -1, (255, 255, 255), 1)
        
        return combined

    @staticmethod
    def save_results(output_dir, img_name, id_map, original_pil):
        """Manages all save logic organized into subfolders."""
        img_stem = os.path.splitext(img_name)[0]
        img_np = np.array(original_pil)
        
        # 1. ID Mask (16-bit PNG)
        path_id = os.path.join(output_dir, 'masks_id', f"{img_stem}_id.png")
        cv2.imwrite(path_id, id_map.astype(np.uint16))
        
        # 2. Visible Mask (Binary 8-bit)
        path_vis = os.path.join(output_dir, 'masks_visible', f"{img_stem}_visible.png")
        visible_mask = (id_map > 0).astype(np.uint8) * 255
        cv2.imwrite(path_vis, visible_mask)
        
        # 3. Overlay colored visualization
        path_overlay = os.path.join(output_dir, 'overlay', f"visu_{img_name}")
        overlay_img = Visualizer.create_overlay(img_np, id_map)
        cv2.imwrite(path_overlay, overlay_img)