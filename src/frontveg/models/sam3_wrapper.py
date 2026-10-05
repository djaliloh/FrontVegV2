"""Thin wrapper around the SAM3 image model and its text-prompt processor."""

from __future__ import annotations

import logging

import torch

from frontveg.resources import add_to_sys_path

logger = logging.getLogger(__name__)


class SAM3Predictor:
    def __init__(self, checkpoint_path, repo_path=None, device=None, conf_threshold: float = 0.35):
        add_to_sys_path(repo_path)
        try:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ImportError as e:
            raise ImportError(
                "Could not import sam3. Install it with 'pip install -e external/sam3' "
                "or set FRONTVEG_SAM3_REPO."
            ) from e

        if device is None or (device == "cuda" and not torch.cuda.is_available()):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Half precision on Volta and newer GPUs; full precision elsewhere.
        use_fp16 = self.device.type == "cuda" and torch.cuda.get_device_capability(self.device)[0] >= 7
        self.dtype = torch.float16 if use_fp16 else torch.float32

        logger.info("Loading SAM3 on %s (%s)", self.device, self.dtype)
        self.model = build_sam3_image_model(
            checkpoint_path=str(checkpoint_path), load_from_HF=False, device=self.device.type
        )
        self.processor = Sam3Processor(
            self.model, device=self.device.type, confidence_threshold=conf_threshold
        )

    def predict_crop(self, pil_crop, prompt: str):
        """Segment ``prompt`` in one image or tile. Returns ``(masks, scores)`` or ``(None, None)``."""
        autocast = torch.autocast(
            device_type=self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"
        )
        with torch.no_grad(), autocast:
            state = self.processor.set_image(pil_crop)
            state = self.processor.set_text_prompt(state=state, prompt=prompt)

        if state.get("masks") is None:
            return None, None
        return state["masks"].cpu().numpy(), state["scores"].float().cpu().numpy()

    def release_cache(self) -> None:
        """Return cached GPU memory to the driver."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
