import numpy as np
import torch
from torchvision.ops import box_iou
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

class SAM3TiledInference:
    def __init__(self, predictor, tile_size=640, overlap=0.50, max_tile_ratio=0.55):
        self.predictor = predictor
        self.tile_size = tile_size
        self.stride = int(tile_size * (1 - overlap))
        self.max_tile_ratio = max_tile_ratio

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

        # --- LOGIQUE DE GRAPHE  ---
        boxes = torch.tensor([f['bbox'] for f in fragments], dtype=torch.float32)
        iou_matrix = box_iou(boxes, boxes)
        adjacency = (iou_matrix > 0.15).numpy().astype(int) # Seuil IOU
        
        n_leaves, labels = connected_components(csgraph=csr_matrix(adjacency), directed=False)
        
        # Reconstruction de la map ID (16-bit)
        final_id_map = np.zeros((H, W), dtype=np.uint16)
        for idx, frag in enumerate(fragments):
            leaf_id = labels[idx] + 1
            x1, y1, x2, y2 = frag['bbox']
            patch = frag['mask']
            # On s'assure que le patch rentre dans la zone (clipping)
            h_p, w_p = patch.shape
            final_id_map[y1:y1+h_p, x1:x1+w_p][patch] = leaf_id
            
        return final_id_map, n_leaves

# =============================================================================
# import numpy as np
# import torch
# from scipy.ndimage import gaussian_filter
# from skimage.measure import label
# from skimage.morphology import remove_small_objects, binary_closing, disk


# class SAM3TiledInference:

#     def __init__(self, predictor, tile_size=640, overlap=0.5):
#         self.predictor = predictor
#         self.tile_size = tile_size
#         self.stride = int(tile_size * (1 - overlap))

#     def _is_hallucination(self, mask, w_t, h_t):
#         area = mask.sum()
#         if area < 50:
#             return True
#         if area / (w_t * h_t) > 0.6:
#             return True
#         return False

#     def process(self, pil_image, prompt):
#         W, H = pil_image.size

#         # --- ACCUMULATION MAPS ---
#         score_map = np.zeros((H, W), dtype=np.float32)
#         count_map = np.zeros((H, W), dtype=np.float32)

#         for y in range(0, H, self.stride):
#             for x in range(0, W, self.stride):

#                 x_end = min(x + self.tile_size, W)
#                 y_end = min(y + self.tile_size, H)

#                 crop = pil_image.crop((x, y, x_end, y_end))
#                 masks, scores = self.predictor.predict_crop(crop, prompt)

#                 if masks is None:
#                     continue

#                 for i, m_raw in enumerate(masks):
#                     m_bool = m_raw[0] > 0

#                     if self._is_hallucination(m_bool, x_end - x, y_end - y):
#                         continue

#                     score = scores[i]

#                     # projection globale
#                     h_p, w_p = m_bool.shape

#                     score_map[y:y+h_p, x:x+w_p] += m_bool * score
#                     count_map[y:y+h_p, x:x+w_p] += m_bool

#                 torch.cuda.empty_cache()

#         return self._finalize(score_map, count_map)

#     def _finalize(self, score_map, count_map):

#         # --- NORMALISATION ---
#         valid = count_map > 0
#         norm_map = np.zeros_like(score_map)
#         norm_map[valid] = score_map[valid] / count_map[valid]

#         # --- SMOOTH ---
#         norm_map = gaussian_filter(norm_map, sigma=1)

#         # --- THRESHOLD ---
#         binary = norm_map > 0.3

#         # --- CLEANING ---
#         binary = binary_closing(binary, disk(3))
#         binary = remove_small_objects(binary, min_size=200)

#         # --- FINAL INSTANCES ---
#         labeled = label(binary)

#         return labeled.astype(np.uint16), labeled.max()