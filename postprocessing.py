import cv2
import numpy as np
import cv2.ximgproc as xip

depth = cv2.imread("/home/utilisateur/Bureau/segmentation/Depth-Anything-V2/depth_maps/pecher_moins_cloque/1CA92B13-06A8-4BA4-ABDF-16B4FB1513D1.png", cv2.IMREAD_GRAYSCALE)
# filtered = cv2.bilateralFilter(depth, d=9, sigmaColor=75, sigmaSpace=75)

# cv2.imwrite("filtered_bilateral.png", filtered)



# depth = cv2.imread("/home/utilisateur/Bureau/segmentation/Depth-Anything-V2/depth_maps/pecher_moins_cloque/1CA92B13-06A8-4BA4-ABDF-16B4FB1513D1.png", cv2.IMREAD_GRAYSCALE)
# guide = depth  # ou image RGB si disponible
# filtered = xip.guidedFilter(guide=guide, src=depth, radius=8, eps=0.01)

# cv2.imwrite("filtered_guided.png", filtered)


# median = cv2.medianBlur(depth, 5)
# mean = cv2.blur(depth, (5, 5))

# depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
# h = depth_norm.shape[0]
# gradient_mask = np.linspace(1.0, 0.5, h).reshape(-1, 1)  # de haut (1.0) vers bas (0.5)
# adjusted = (depth_norm.astype(np.float32) * gradient_mask).astype(np.uint8)



# Normalisation (optionnelle, si les valeurs ne sont pas sur 0–255)
depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Définir le seuil vertical (par exemple, les 30% du bas)
h = depth_norm.shape[0]
threshold_y = int(h * 0.7)

# Créer un masque pour la partie basse
mask = np.zeros_like(depth_norm, dtype=np.uint8)
mask[threshold_y:, :] = 1  # 1 en bas, 0 en haut

# Réduire la luminosité dans la zone basse
darkened = depth_norm.copy()
darkened[mask == 1] = (darkened[mask == 1] * 0.25).astype(np.uint8)  # facteur d'atténuation

cv2.imwrite("filtered_guided4.png", darkened)

