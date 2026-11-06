import numpy as np
import torch
from torchvision.ops import box_iou
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

class SAM3TiledInference:
    def __init__(
    self, 
    predictor, 
    tile_size=640, 
    overlap=0.50, 
    max_tile_ratio=0.55,
    bbox_iou_threshold=0.05,
    mask_iou_threshold=0.10
    ):
        self.predictor = predictor
        self.tile_size = tile_size
        self.stride = int(tile_size * (1 - overlap))
        self.max_tile_ratio = max_tile_ratio
        # GMA params
        self.bbox_iou_threshold = bbox_iou_threshold
        self.mask_iou_threshold = mask_iou_threshold 

    def _is_hallucination(self, mask, w_t, h_t):
        area = mask.sum()
        if area < 50: return True
        if area / (w_t * h_t) > self.max_tile_ratio: return True
        return False

    def process(self, pil_image, prompt):
        W, H = pil_image.size
        fragments = []

        # --- BOUCLE DE TILING ---
        for y in range(0, H, self.stride):
            for x in range(0, W, self.stride):
                x_end, y_end = min(x + self.tile_size, W), min(y + self.tile_size, H)
                w_t, h_t = x_end - x, y_end - y
                
                crop = pil_image.crop((x, y, x_end, y_end))
                masks, scores = self.predictor.predict_crop(crop, prompt)

                if masks is not None:
                    for i, m_raw in enumerate(masks):
                        m_bool = m_raw[0] > 0
                        if self._is_hallucination(m_bool, w_t, h_t): continue

                        # bbox extraction and fragment storage
                        ys, xs = np.where(m_bool)
                        if len(ys) == 0: continue
                        
                        ly1, lx1, ly2, lx2 = ys.min(), xs.min(), ys.max(), xs.max()
                        fragments.append({
                            'bbox': [lx1 + x, ly1 + y, lx2 + x, ly2 + y],
                            'mask': m_bool[ly1:ly2+1, lx1:lx2+1],  # Cropped mask (to save RAM)
                            'score': scores[i]
                        })
                torch.cuda.empty_cache()

        return self._merge_fragments(fragments, W, H)

    def _merge_fragments(self, fragments, W, H):
        if not fragments: return None, 0

        num_frags = len(fragments)
        
        # --- ÉTAPE 1 : GRAPHE DE CONNEXITÉ HYBRIDE (BOX PUIS MASK IOU) ---
        # On commence par calculer l'IoU rapide des boîtes englobantes
        boxes = torch.tensor([f['bbox'] for f in fragments], dtype=torch.float32)
        box_iou_matrix = box_iou(boxes, boxes).numpy()
        
        # Matrice d'adjacence initiale (chaque fragment est connecté à lui-même)
        adjacency = np.eye(num_frags, dtype=bool)
        
        # On filtre : on n'analyse au niveau du masque QUE si les bboxes se touchent
        pairs = np.argwhere(box_iou_matrix > self.bbox_iou_threshold)
        for i, j in pairs:
            if i >= j: continue # On évite les doublons (matrice symétrique)
            
            f1, f2 = fragments[i], fragments[j]
            b1, b2 = f1['bbox'], f2['bbox']
            
            # Calcul de la zone d'intersection des bboxes (coordonnées inclusives)
            ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
            ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
            
            if ix1 <= ix2 and iy1 <= iy2:
                # Extraction des sous-régions de masques (attention au +1 pour l'excluf de numpy)
                p1 = f1['mask'][iy1 - b1[1] : iy2 - b1[1] + 1, ix1 - b1[0] : ix2 - b1[0] + 1]
                p2 = f2['mask'][iy1 - b2[1] : iy2 - b2[1] + 1, ix1 - b2[0] : ix2 - b2[0] + 1]
                
                # Calcul du Mask IoU réel sur la zone commune
                intersection = np.logical_and(p1, p2).sum()
                if intersection > 0:
                    area1 = f1['mask'].sum()
                    area2 = f2['mask'].sum()
                    mask_iou = intersection / (area1 + area2 - intersection)
                    
                    # Seuil d'IoU réel pour valider qu'il s'agit du même objet coupé par la grille
                    if mask_iou > self.mask_iou_threshold: 
                        adjacency[i, j] = True
                        adjacency[j, i] = True

        # Résolution des composants connexes
        n_leaves, labels = connected_components(csgraph=csr_matrix(adjacency), directed=False)

        # --- ÉTAPES 2 & 3 : FUSION PAR MAXIMUM GLOBAL (RAM OPTIMISÉE) ---
        # Plus de dictionnaire par leaf, plus de stack (N, H, W). 
        # On travaille directement "en ligne" sur l'image finale.
        final_id_map = np.zeros((H, W), dtype=np.uint16)
        max_score_map = np.zeros((H, W), dtype=np.float32)

        for idx, frag in enumerate(fragments):
            leaf_id = int(labels[idx]) + 1
            score = float(frag['score'])
            x1, y1, x2, y2 = frag['bbox']
            patch = frag['mask']

            h_p, w_p = patch.shape
            y2c, x2c = min(y1 + h_p, H), min(x1 + w_p, W)
            ph, pw = y2c - y1, x2c - x1

            # Masque local ajusté aux bordures de l'image si nécessaire
            local_mask = patch[:ph, :pw]

            # On extrait la zone correspondante de notre carte de scores globale
            global_score_roi = max_score_map[y1:y2c, x1:x2c]

            # Un pixel est mis à jour SI il appartient au fragment ET si son score 
            # est strictement supérieur au score maximum enregistré à cet endroit.
            update_mask = local_mask & (score > global_score_roi)

            # Écriture directe (Winner-Takes-All instantané sans allocation de RAM)
            max_score_map[y1:y2c, x1:x2c][update_mask] = score
            final_id_map[y1:y2c, x1:x2c][update_mask] = leaf_id

        return final_id_map, n_leaves