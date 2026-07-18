import os
import glob
from PIL import Image
from frontveg.models.sam3_wrapper import SAM3Predictor
from frontveg.core.sam3_tiler import SAM3TiledInference
from frontveg.utils.visualization import Visualizer

def main():
    # --- CONFIGURATION (Normalement via YAML) ---
    CKPT = "checkpoints/sam3_ckpts/sam3.pt"
    INPUT_DIR = "data/arbo_test/test"
    OUTPUT_DIR = "outputs/sam3_test_pathlib"
    PROMPT = "leaf"
    
    # Init the folders for results
    for sub in ['masks_id', 'masks_visible', 'overlay']:
        os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)

    # --- INITIALIZATION ---
    # Load the model ONLY ONCE
    predictor = SAM3Predictor(checkpoint_path=CKPT)
    # We configure the tiler with the predictor and our tiling strategy
    tiler = SAM3TiledInference(predictor, tile_size=640, overlap=0.5)

    # --- EXECUTION ---
    # images = glob.glob(os.path.join(INPUT_DIR, "*.jpg *.tiff *.png *.tif")) # Ajoute les autres extensions si besoin
    # from pathlib import Path
    # ext = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    # images = [p for p in Path(INPUT_DIR).iterdir() if p.suffix.lower() in ext]
    from pathlib import Path
    images = Path(INPUT_DIR).glob("*.*") # Ajoute les autres extensions si besoin
    
    for img_path in images: #glob.glob(os.path.join(INPUT_DIR, "*")):
        name = os.path.basename(img_path)
        print(f"Processing {name}...")
        
        img = Image.open(img_path).convert("RGB")
        
        # L'inférence produit la map ID finale
        id_map, n_found = tiler.process(img, prompt=PROMPT)
        
        if id_map is not None:
            print(f"Found {n_found} elements.")
            Visualizer.save_results(OUTPUT_DIR, name, id_map, img)
        else:
            print("No elements detected.")

if __name__ == "__main__":
    main()



# pip install -e external/sam3
# pip install -e external/depthanything
# pip install -e external/dino

# PYTHONPATH=src python -m frontveg.main_sam3
# PYTHONPATH=src python -m frontveg.main_frontveg --input data/raw/arbo_test/test --output outputs/sam3_results --encoder vitl --plot