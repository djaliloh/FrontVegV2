import os
import cv2
import sys
import glob
import torch
import argparse
import matplotlib
import numpy as np

module_path = os.path.abspath("Depth-Anything-V2") # path of the Depth Anything
print(module_path)
if module_path not in sys.path:
    sys.path.append(module_path)

from depth_anything_v2.dpt import DepthAnythingV2

def load_model(encoder, device):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    model = DepthAnythingV2(**model_configs[encoder])
    checkpoint_path = f'Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth'
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    return model.to(device).eval()

def process_images(img_path, input_size, outdir, encoder, pred_only, grayscale, device):
    depth_anything = load_model(encoder, device)
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")
    
    # filenames = []
    # for subdir in glob.glob(os.path.join(img_path, "*")): 
           
    #     if os.path.isdir(subdir):  
    #         filenames += [f for f in glob.glob(os.path.join(subdir, '**/*'), recursive=True) if f.lower().endswith(valid_extensions)]

    # cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    # for k, filename in enumerate(filenames):
    #     print(f'Processing {k+1}/{len(filenames)}: {filename}')
    #     subdirname = os.path.basename(os.path.dirname(filename))

        # print("IMAGE",filename)
        # print("subdir", subdirname)

    images_by_subdir = {}

    for subdir in glob.glob(os.path.join(img_path, "*")):
        # print("ICC:: SUBDIR",subdir)
        # print("IMAGE PATH",img_path)
        # exit(0)
        if os.path.isdir(subdir):
            sub_images = [
                f for f in glob.glob(os.path.join(subdir, '**/*'), recursive=True)
                if f.lower().endswith(valid_extensions)
            ]
            if sub_images: 
                images_by_subdir[subdir] = sub_images

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    # Processing images
    for subdir, images in images_by_subdir.items():
        subdirname = os.path.basename(subdir)
        print(f"\n Traitement du dossier : {subdirname} ({len(images)} images)")

        for i, img_path in enumerate(images, start=1):
            print(f"{i}/{len(images)} : {img_path}")


            os.makedirs(os.path.join(outdir, "depth_maps", subdirname), exist_ok=True)

            raw_image = cv2.imread(img_path)
            depth = depth_anything.infer_image(raw_image, input_size)

            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)

            if grayscale:
                depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
            else:
                depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

            output_path = os.path.join(outdir, "depth_maps", subdirname, os.path.basename(img_path))
            if pred_only:
                cv2.imwrite(output_path, depth)
            else:
                split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
                combined_result = cv2.hconcat([raw_image, split_region, depth])
                cv2.imwrite(output_path, combined_result)

        print(f"Traitement du dossier {subdirname} terminé !")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exécute Depth Anything V2 sur un jeu d'images")
    parser.add_argument("--img_path", type=str, default="./dataset/", help="Chemin vers le dossier d'images")
    parser.add_argument("--input_size", type=int, default=518, help="Taille d'entrée du modèle")
    parser.add_argument("--outdir", type=str, default="./Depth-Anything-V2", help="Répertoire de sortie")
    parser.add_argument("--encoder", type=str, choices=["vits", "vitb", "vitl", "vitg"], default="vitl", help="Type d'encodeur")
    parser.add_argument("--pred_only", action="store_true", help="Si défini, sauvegarde uniquement la prédiction")
    parser.add_argument("--grayscale", action="store_true", help="Si défini, génère des cartes de profondeur en niveaux de gris")

    args = parser.parse_args()

    # Détection du device
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"You are using : *{device}*")

    process_images(args.img_path, args.input_size, args.outdir, args.encoder, args.pred_only, args.grayscale, device)


# run  --> python generate_depth.py --img_path ./dataset/ --input_size 518 --outdir ./GdSam2/Depth-Anything-V2 --encoder vitl --pred_only --grayscale
