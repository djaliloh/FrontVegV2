import torch
import numpy as np
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

class SAM3Predictor:
    def __init__(self, checkpoint_path, device="cuda", conf_threshold=0.35):
        self.device = torch.device(device)
        print(f"Loading SAM3 on {self.device}...")
        
        # Initialization of the original model and processor
        self.model = build_sam3_image_model(checkpoint_path=checkpoint_path, load_from_HF=False)
        self.processor = Sam3Processor(self.model, confidence_threshold=conf_threshold)

    def predict_crop(self, pil_crop, prompt):
        """Predicted on a single tile (crop)."""
        with torch.no_grad():
            inputs = self.processor.set_image(pil_crop)
            inputs = self.processor.set_text_prompt(state=inputs, prompt=prompt)
        
        if "masks" in inputs and inputs["masks"] is not None:
            return inputs["masks"].cpu().numpy(), inputs["scores"].cpu().numpy()
        return None, None