import os
import re
import argparse
import numpy as np
import pandas as pd
import cv2 as cv
from PIL import Image
from tqdm import tqdm

def natural_sort_key(s):
    """Permit to sort strings containing numbers in a natural way (ex: 2 before 10)"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def load_binary_image(path):
    """Load an image, convert it to grayscale, and return a binary mask (boolean)"""
    img = Image.open(path).convert('L')
    arr = np.array(img, dtype=np.uint8)
    return arr != 0

def calculate_metrics(gt, mask):
    eps = 1e-6
    gt_flat = gt.flatten()
    mask_flat = mask.flatten()
    
    tp = np.sum(gt_flat & mask_flat)
    tn = np.sum(~gt_flat & ~mask_flat)
    fp = np.sum(~gt_flat & mask_flat)
    fn = np.sum(gt_flat & ~mask_flat)
    
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    fpr = fp / (fp + tn + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    
    return dice, precision, recall, fpr, iou

def image_confusion(GT, predict):
    """Generates the confusion image in BGR format in an extremely optimized and vectorized way."""
    G = GT.astype(bool)
    P = predict.astype(bool)
    
    # Channel R = P * 255, Channel G = G * 255, Channel B = (P & G) * 255
    # In BGR: B, G, R
    R = P * np.uint8(255)
    G_ch = G * np.uint8(255)
    B = (P & G) * np.uint8(255)
    
    return np.dstack((B, G_ch, R))

def main():
    parser = argparse.ArgumentParser(description="Calcul des métriques de segmentation et génération d'images de confusion.")
    parser.add_argument("--gt_dir", type=str, required=True, help="Dossier contenant les Ground Truths (masques sémantiques)")
    parser.add_argument("--mask_dir", type=str, required=True, help="Dossier contenant les masques prédits")
    parser.add_argument("--output_dir", type=str, default="metrics", help="Dossier de sortie pour les résultats (Excel et confusion)")
    parser.add_argument("--method", type=str, default="frontveg2", help="Nom de la méthode évaluée")
    parser.add_argument("--skip_confusion", action="store_true", help="Passer la génération des images de confusion")
    
    args = parser.parse_args()

    # Configuration des chemins
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Chargement initial et filtrage rigoureux (une seule fois)
    gt_files_raw = [f for f in os.listdir(args.gt_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    mask_files_raw = [f for f in os.listdir(args.mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

    if len(gt_files_raw) == 0:
        print(f"Error: No valid files found in gt_dir : {args.gt_dir}")
        return
    if len(mask_files_raw) == 0:
        print(f"Error: No valid files found in mask_dir : {args.mask_dir}")
        return

    # Dictionnaires nom sans extension → nom complet avec extension
    gt_dict = {os.path.splitext(f)[0]: f for f in gt_files_raw}
    mask_dict = {os.path.splitext(f)[0]: f for f in mask_files_raw}

    # Intersection triée de manière naturelle
    common_names = sorted(set(gt_dict.keys()) & set(mask_dict.keys()), key=natural_sort_key)
    
    if len(common_names) == 0:
        print("Error: No common files found between GT and Masks based on names.")
        return
        
    print(f"Found {len(common_names)} matching pairs to process.")

    method_results = []

    # 2. Boucle de calcul des métriques
    print("\n>>> Calculating metrics...")
    for idx, name in enumerate(common_names):
    
        gt_path = os.path.join(args.gt_dir, gt_dict[name])
        mask_path = os.path.join(args.mask_dir, mask_dict[name])

        print(f"{idx+1}/{len(common_names)} {os.path.basename(gt_path)} <--> {os.path.basename(mask_path)}")

        gt = load_binary_image(gt_path)
        mask = load_binary_image(mask_path)

        if gt.shape != mask.shape:
            mask = cv.resize(mask.astype(np.uint8),
                             (gt.shape[1], gt.shape[0]),
                             interpolation=cv.INTER_NEAREST).astype(bool)

        dice, precision, recall, fpr, iou = calculate_metrics(gt, mask)

        method_results.append({
            "File": gt_dict[name],
            "Mask_File": mask_dict[name],
            "Dice": dice,
            "Precision": precision,
            "Recall": recall,
            "FPR": fpr,
            "IoU": iou,
            "Threshold Method": args.method
        })

    # Calcul et sauvegarde des statistiques globales
    dice_scores = [res["Dice"] for res in method_results]
    iou_scores = [res["IoU"] for res in method_results]
    
    mean_std = {
        "Threshold Method": args.method,
        "Mean Dice": np.mean(dice_scores),
        "Std Dice": np.std(dice_scores),
        "Mean IoU": np.mean(iou_scores),
        "Std IoU": np.std(iou_scores)
    }

    excel_path = os.path.join(args.output_dir, f"results_{args.method}.xlsx")
    df_details = pd.DataFrame(method_results)
    df_summary = pd.DataFrame([mean_std])
    
    with pd.ExcelWriter(excel_path) as writer:
        df_details.to_excel(writer, sheet_name=args.method, index=False)
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
    
    print(f"Mean Dice: {mean_std['Mean Dice']:.4f} | Mean IoU: {mean_std['Mean IoU']:.4f}")
    print(f"Results saved in: {excel_path}")

    # 3. Génération des images de confusion (alignement préservé !)
    if not args.skip_confusion:
        print("\n>>> Generating confusion images...")
        confusion_dir = os.path.join(args.output_dir, "confusion_images_" + args.method)
        os.makedirs(confusion_dir, exist_ok=True)
        
        for name in tqdm(common_names):
            gt_path = os.path.join(args.gt_dir, gt_dict[name])
            mask_path = os.path.join(args.mask_dir, mask_dict[name])
            
            gt = load_binary_image(gt_path)
            mask = load_binary_image(mask_path)
            
            if gt.shape != mask.shape:
                mask = cv.resize(mask.astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv.INTER_NEAREST).astype(bool)
                
            conf_img = image_confusion(gt, mask)
            
            # Sauvegarde 
            output_file = os.path.join(confusion_dir, f"confusion_{args.method}_{name}.png")
            cv.imwrite(output_file, conf_img)
            
        print(f"Confusion images saved in: {confusion_dir}")

if __name__ == "__main__":
    main()


































































# import os
# import re
# import argparse
# import numpy as np
# import pandas as pd
# import cv2 as cv
# from PIL import Image
# from tqdm import tqdm

# def natural_sort_key(s):
#     """Permit to sort strings containing numbers in a natural way (ex: 2 before 10)"""
#     return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# def load_binary_image(path):
#     """Load an image, convert it to grayscale, and return a binary mask (boolean)"""
#     img = Image.open(path).convert('L')
#     arr = np.array(img, dtype=np.uint8)
#     return arr != 0

# def calculate_metrics(gt, mask):
#     eps = 1e-6
#     gt_flat = gt.flatten()
#     mask_flat = mask.flatten()
    
#     tp = np.sum(gt_flat & mask_flat)
#     tn = np.sum(~gt_flat & ~mask_flat)
#     fp = np.sum(~gt_flat & mask_flat)
#     fn = np.sum(gt_flat & ~mask_flat)
    
#     precision = tp / (tp + fp + eps)
#     recall = tp / (tp + fn + eps)
#     fpr = fp / (fp + tn + eps)
#     iou = tp / (tp + fp + fn + eps)
#     dice = (2 * tp) / (2 * tp + fp + fn + eps)
    
#     return dice, precision, recall, fpr, iou

# def image_confusion(GT, predict):
#     """Generates the confusion image in BGR format in an extremely optimized and vectorized way."""
#     G = GT.astype(bool)
#     P = predict.astype(bool)
    
#     # Channel R = P * 255, Channel G = G * 255, Channel B = (P & G) * 255
#     # In BGR: B, G, R
#     R = P * np.uint8(255)
#     G_ch = G * np.uint8(255)
#     B = (P & G) * np.uint8(255)
    
#     return np.dstack((B, G_ch, R))

# def main():
#     parser = argparse.ArgumentParser(description="Calcul des métriques de segmentation et génération d'images de confusion.")
#     parser.add_argument("--gt_dir", type=str, required=True, help="Dossier contenant les Ground Truths (masques sémantiques)")
#     parser.add_argument("--mask_dir", type=str, required=True, help="Dossier contenant les masques prédits")
#     parser.add_argument("--output_dir", type=str, default="metrics", help="Dossier de sortie pour les résultats (Excel et confusion)")
#     parser.add_argument("--method", type=str, default="frontveg2", help="Nom de la méthode évaluée")
#     parser.add_argument("--skip_confusion", action="store_true", help="Passer la génération des images de confusion")
    
#     args = parser.parse_args()

#     # Configuration des chemins
#     os.makedirs(args.output_dir, exist_ok=True)
    
#     gt_files = sorted([f for f in os.listdir(args.gt_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))], key=natural_sort_key)
#     mask_files = sorted([f for f in os.listdir(args.mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))], key=natural_sort_key)

#     if len(gt_files) == 0:
#         print(f"Error: No valid files found in gt_dir : {args.gt_dir}")
#         return
#     if len(mask_files) == 0:
#         print(f"Error: No valid files found in mask_dir : {args.mask_dir}")
#         return

#     if len(gt_files) != len(mask_files):
#         print(f"Warning: Number of GT ({len(gt_files)}) and Masks ({len(mask_files)}) is different!")

#     method_results = []
    
#     # 1. Calcul des métriques
#     print("\n>>> Calculating metrics...")
#     # for gt_file, mask_file in tqdm(zip(gt_files, mask_files), total=min(len(gt_files), len(mask_files))):
#     #     gt_path = os.path.join(args.gt_dir, gt_file)
#     #     mask_path = os.path.join(args.mask_dir, mask_file)

#     #     if os.path.splitext(gt_file)[0] != os.path.splitext(mask_file)[0]:
#     #         print(f"WARNING: {gt_file} != {mask_file}")
#     #         # continue
        
#     #     gt = load_binary_image(gt_path)
#     #     mask = load_binary_image(mask_path)
        
#     #     if gt.shape != mask.shape:
#     #         mask = cv.resize(mask.astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv.INTER_NEAREST).astype(bool)
            
#     #     dice, precision, recall, fpr, iou = calculate_metrics(gt, mask)


#     #     # print(f" [DEBUGGING] {gt_file} -> Dice={dice:.6f}, IoU={iou:.6f}, Precision={precision:.6f}, Recall={recall:.6f}")
        
#     #     method_results.append({
#     #         "File": gt_file,
#     #         "Mask_File": mask_file,
#     #         "Dice": dice,
#     #         "Precision": precision,
#     #         "Recall": recall,
#     #         "FPR": fpr,
#     #         "IoU": iou,
#     #         "Threshold Method": args.method
#     #     })

#     # # 1. Calcul des métriques
#     # print("\n>>> Calculating metrics...")

#     gt_files = [f for f in os.listdir(args.gt_dir)
#                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

#     mask_files = [f for f in os.listdir(args.mask_dir)
#                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

#     # dictionnaires nom → fichier
#     gt_dict = {os.path.splitext(f)[0]: f for f in gt_files}
#     mask_dict = {os.path.splitext(f)[0]: f for f in mask_files}

#     # intersection des noms communs
#     common_names = sorted(set(gt_dict.keys()) & set(mask_dict.keys()),
#                         key=natural_sort_key)

#     method_results = []

#     for name in tqdm(common_names):
#         gt_path = os.path.join(args.gt_dir, gt_dict[name])
#         mask_path = os.path.join(args.mask_dir, mask_dict[name])

#         print("\n[DEBUG] ",os.path.basename(gt_path),"<-->", os.path.basename(mask_path)) # for debugging
        
#         gt = load_binary_image(gt_path)
#         mask = load_binary_image(mask_path)

#         if gt.shape != mask.shape:
#             mask = cv.resize(mask.astype(np.uint8),
#                             (gt.shape[1], gt.shape[0]),
#                             interpolation=cv.INTER_NEAREST).astype(bool)

#         dice, precision, recall, fpr, iou = calculate_metrics(gt, mask)

#         method_results.append({
#             "File": gt_dict[name],
#             "Mask_File": mask_dict[name],
#             "Dice": dice,
#             "Precision": precision,
#             "Recall": recall,
#             "FPR": fpr,
#             "IoU": iou,
#             "Threshold Method": args.method
#         })

#     # Calcul des statistiques globales
#     dice_scores = [res["Dice"] for res in method_results]
#     iou_scores = [res["IoU"] for res in method_results]
    
#     mean_std = {
#         "Threshold Method": args.method,
#         "Mean Dice": np.mean(dice_scores),
#         "Std Dice": np.std(dice_scores),
#         "Mean IoU": np.mean(iou_scores),
#         "Std IoU": np.std(iou_scores)
#     }

#     # Sauvegarde Excel
#     excel_path = os.path.join(args.output_dir, f"results_{args.method}.xlsx")
#     df_details = pd.DataFrame(method_results)
#     df_summary = pd.DataFrame([mean_std])
    
#     with pd.ExcelWriter(excel_path) as writer:
#         df_details.to_excel(writer, sheet_name=args.method, index=False)
#         df_summary.to_excel(writer, sheet_name="Summary", index=False)
    
#     print(f"Mean Dice: {mean_std['Mean Dice']:.4f} | Mean IoU: {mean_std['Mean IoU']:.4f}")
#     print(f"Results saved in: {excel_path}")

#     # 2. Génération des images de confusion
#     if not args.skip_confusion:
#         print("\n>>> Generating confusion images...")
#         confusion_dir = os.path.join(args.output_dir, "confusion_images_"+ args.method)
#         os.makedirs(confusion_dir, exist_ok=True)
        
#         for idx, (gt_file, mask_file) in enumerate(tqdm(zip(gt_files, mask_files), total=min(len(gt_files), len(mask_files)))):
#             gt_path = os.path.join(args.gt_dir, gt_file)
#             mask_path = os.path.join(args.mask_dir, mask_file)
            
#             gt = load_binary_image(gt_path)
#             mask = load_binary_image(mask_path)
            
#             if gt.shape != mask.shape:
#                 mask = cv.resize(mask.astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv.INTER_NEAREST).astype(bool)
                
#             conf_img = image_confusion(gt, mask)
#             output_file = os.path.join(confusion_dir, f"confusion_{args.method}_{idx+1}.png")
#             cv.imwrite(output_file, conf_img)
            
#         print(f"Confusion images saved in: {confusion_dir}")

# if __name__ == "__main__":
#     main()



# # python run_metrics.py --gt_dir 
# # "F:\EXPERIMENTS-ECCV-annot_code\annotaion-labelbox\vine\blackrot\gts\semantic_masks" 
# # --mask_dir "F:\EXPERIMENTS-ECCV-annot_code\annotaion-labelbox\vine\blackrot\fvg_predictions_peak_dist_8_brownspot\final_masks\blackrot" 
# # --output_dir "metrics" --method "frontveg2"
