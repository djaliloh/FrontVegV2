import torch
import cv2
import numpy as np
import sys
import os

class DepthAnythingWrapper:
    def __init__(self, encoder='vitl', device=None, repo_path="external/Depth-Anything-V2"):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        if repo_path is None:
            raise ValueError("Depth repo path must be explicitly provided")     
        
        # Ajout dynamique du repo externe au path
        if repo_path not in sys.path:
            sys.path.append(repo_path)
            
        from depth_anything_v2.dpt import DepthAnythingV2

        self.configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        self.model = DepthAnythingV2(**self.configs[encoder])
        ckpt_path = f'checkpoints/depthanything_ckpts/depth_anything_v2_{encoder}.pth'  
        self.model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        self.model.to(self.device).eval()
        print(f"DepthAnythingV2 ({encoder}) loaded on {self.device}")

    def infer(self, bgr_image, input_size=518):
        """Prend une image CV2 (BGR) et retourne une map [0-255] uint8."""
        depth = self.model.infer_image(bgr_image, input_size)
        # Normalisation 0-255
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255.0
        return depth.astype(np.uint8)
    



# /home/adjalil/Working/FrontVegetation/checkpoints/depthanything_ckpts/depth_anything_v2_vitl.pth