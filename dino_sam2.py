import glob
import os
import cv2
import json
import tqdm
import time
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import supervision as sv
import imageio.v2 as imageio
from datetime import datetime
import pycocotools.mask as mask_util
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict


def dino_sam2_segmentation(TEXT_PROMPT, IMG_PATH, BOX_THRESHOLD, TEXT_THRESHOLD, OUTPUT_DIR, img_compteur):
    """
    Author : ADOH
    """

    # Output directory
    subdir_name = os.path.basename(os.path.dirname(IMG_PATH)) 
    image_name = os.path.basename(IMG_PATH)     

    output_dir = Path(os.path.join(OUTPUT_DIR, "grounded_sam2_preds", subdir_name))
    output_dir.mkdir(parents=True, exist_ok=True)


    # setup the input image and text prompt for SAM 2 and Grounding DINO
    # VERY important: text queries need to be lowercased + end with a dot
    # Load image
    image_source, image = load_image(IMG_PATH)
    sam2_predictor.set_image(image_source)

    # **Detection : Grounding DINO Prediction**
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )

    # process the box prompt to compatible format for SAM 2 
    h, w, _ = image_source.shape
    if boxes is None or len(boxes) == 0:  # Aucune boîte détectée
        mask_dir = Path(os.path.join(OUTPUT_DIR, "masks", subdir_name))
        mask_dir.mkdir(parents=True, exist_ok=True)

        mask_empty = np.zeros((h, w), dtype=np.uint8)  # Image noire
        Image.fromarray(mask_empty).save(os.path.join(mask_dir, f"{image_name}"))
        # print(f"DEBUGG::: IMAGE NOIRE ENREGISTREE DANS 000: {mask_dir}")

        return []  # Quitte la fonction après avoir enregistré l'image noire

    # Si on a bien des boîtes, on continue normalement
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    

    torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # print("TF32 activé pour optimisation des calculs sur GPU Ada Lovelace")


    # **Segmentation : SAM2 prediction**
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )


    """
    Post-process the output of the model to get the masks, scores, and logits for visualization
    """
    # convert the shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)


    confidences = confidences.numpy().tolist()
    class_names = labels

    class_ids = np.array(list(range(len(class_names))))

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(class_names, confidences)
    ]

    """
    Visualize image with supervision API
    """
    img = cv2.imread(IMG_PATH)
    detections = sv.Detections(
        xyxy=input_boxes,  # (n, 4)
        mask=masks.astype(bool),  # (n, h, w)
        class_id=class_ids
    )

    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    # cv2.imwrite(os.path.join(output_dir, "groundingdino_annotated_image3.jpg"), annotated_frame)

    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imwrite(os.path.join(output_dir, f"{image_name}"), annotated_frame)

    """
    Save the mask 
    """
    mask_dir = Path(os.path.join(OUTPUT_DIR, "masks", subdir_name))
    mask_dir.mkdir(parents=True, exist_ok=True)

    
    if masks is not None and len(masks) != 0:
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for mask_ in masks:
            combined_mask = np.maximum(combined_mask, (mask_ * 255).astype(np.uint8))
        Image.fromarray(combined_mask).save(os.path.join(mask_dir, f"{image_name}"))

    else:  # Si masks est None ou une liste vide
        mask_empty = np.zeros((h, w), dtype=np.uint8)  # Image noire
        Image.fromarray(mask_empty).save(os.path.join(mask_dir, f"{image_name}"))
        # print(f"IMAGE NOIRE ENREGISTREE DANS 404: {mask_dir}")
    
    # **Empty GPU memory after each image**
    # del boxes, confidences, labels, masks, scores, logits  # Delete tensors
    torch.cuda.empty_cache()  # Empty GPU memory
    torch.cuda.ipc_collect()  # Clean CUDA



# Load model at once
SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"#"checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml" #"sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"  #GroundingDINO_SwinB_cfg.py
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"               #groundingdino_swinb_cogcoor.pth

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE USED: *{DEVICE}*")

# SAM Model
sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# Dino Model
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG, 
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)
