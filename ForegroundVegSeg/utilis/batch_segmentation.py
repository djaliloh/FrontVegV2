import argparse
import glob
import os
import tqdm
import torch
from pathlib import Path
from datetime import datetime

import sys
sys.path.append('/home/utilisateur/Bureau/segmentation')
from dino_sam2 import dino_sam2_segmentation


def batch_image_seg(args):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DEVICE USED: *{DEVICE}*")

    IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tiff', '.tif')

    # Timer
    start_time = datetime.now()

    prev_subdir = None
    sub_ix = 0

    for img_pth in tqdm.tqdm(glob.glob(os.path.join(args.input_path, "*", "*"))):
        sub_dirname = os.path.basename(os.path.dirname(img_pth))

        if not img_pth.lower().endswith(IMG_EXTENSIONS):
            continue

        if sub_dirname != prev_subdir:
            sub_ix = 0
            prev_subdir = sub_dirname

        sub_ix += 1
        total_images_in_subdir = len(os.listdir(os.path.dirname(img_pth)))

        for text_prompt in args.text_prompts:
            # print("--START--", text_prompt)
            prompt_clean = text_prompt.replace(" ", "_").replace(".", "").replace(",", "")
            OUTPUT_DIR_sd = Path(args.output_path) / f"{prompt_clean}"
            # print("**PROMPT**",prompt_clean)
            # print("**OUT DIR**", OUTPUT_DIR_sd)
            # exit(0)
            OUTPUT_DIR_sd.mkdir(parents=True, exist_ok=True)

            print(f"Image: {sub_ix}/{total_images_in_subdir} in {sub_dirname}")
            dino_sam2_segmentation(
                TEXT_PROMPT=text_prompt,
                IMG_PATH=img_pth,
                BOX_THRESHOLD=args.box_threshold,
                TEXT_THRESHOLD=args.text_threshold,
                OUTPUT_DIR=OUTPUT_DIR_sd,
                img_compteur=sub_ix
            )

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f"TEMPS ECOULÉ : {elapsed_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation batch with Grounding DINO + SAM2")

    parser.add_argument("--input_path", type=str, default="./dataset", required=True, help="path to image dataset")
    parser.add_argument("--output_path", type=str, default="dinoSam2-outputs", help="output folder")
    parser.add_argument("--text_prompts", type=str, nargs='+', default=["green foliage . fruits ."], help="Liste de prompts, séparés par des espaces points")
    parser.add_argument("--box_threshold", type=float, default=0.35, help="bbx threshold for DINO")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    # parser.add_argument("--sam2_checkpoint", type=str, default="./checkpoints/sam2.1_hiera_large.pt", help="Checkpoint du modèle SAM2")
    # parser.add_argument("--sam2_config", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml", help="Config du modèle SAM2")
    # parser.add_argument("--gdino_config", type=str, default="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py", help="Config de Grounding DINO")
    # parser.add_argument("--gdino_checkpoint", type=str, default="gdino_checkpoints/groundingdino_swint_ogc.pth", help="Checkpoint de Grounding DINO")

    args = parser.parse_args()
    batch_image_seg(args)



# txt_prompt = [
# "hypertrophied leaves",
# "swallen leaves",
# "curled leaves",
# "distorted foliage",
# "blistered leaves",
# "puffy leaves",
# "malformed leaves"
# ]

# # Disease prompts -> vigne
# txt_prompt = [
# "black rot",
# "Red discolored leaves",
# "Reddened foliage"
# ]