# FrontVeg-V2: Foreground-Aware Zero-Shot Plant Trait Segmentation

<img src="assets/logo.png" alt="Project Logo" width=3000>

[![Napari Hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/link/napari-frontveg)](https://www.napari-hub.org/plugins/napari-frontveg)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

We present **FrontVeg V2**, a high-precision computer vision pipeline designed for automated phenotyping in **viticulture** and **arboriculture**. By fusing **monocular depth estimation** with **vision-language segmentation priors (SAM3)**, it isolates the foreground canopy from cluttered agricultural backgrounds, enabling accurate leaf area estimation and fruit counting.

## Key Features

* **Geometric-Semantic Fusion:** Uses Depth-Anything-V2 to filter out background rows, solving the "background noise" problem in dense orchards.
* **Zero-Shot Adaptation:** Prompt-based segmentation (e.g., "leaf", "grapes", "diseased") without re-training.
* **Napari Plugin:** Interactive "Human-in-the-loop" interface for researchers.
* **Advanced Phenotyping:** Automated calculation of **Total Area**, **Instance Count**, **Mean Size**, and **Standard Deviation**.

## Installation

### 1. Install the Package

Install FrontVeg V2 and its plugin via PyPI:
```bash
pip install frontvegV2
```
*Note: This will install napari and the core runtime dependencies.*

### 2. Download Required Models

FrontVeg V2 relies on **SAM3** and **Depth-Anything-V2**, which require manual downloading of their checkpoints and source code. 

1. **Depth-Anything-V2**: 
   - Clone the repo: `git clone https://github.com/DepthAnything/Depth-Anything-V2.git`
   - Download the pre-trained weights (e.g., `depth_anything_v2_vitl.pth`).
2. **SAM3**: 
   - Clone the repo: `git clone https://github.com/facebookresearch/sam3.git`
   - Install it: `cd sam3 && pip install -e .`
   - Download the SAM3 weights (e.g., `sam3.pt`).

### 3. Configure Paths

By default, the plugin will look for models in `checkpoints/` and `external/` directories relative to your current working directory. 

If your models are stored elsewhere, set the following environment variables before launching napari:
```bash
export FRONTVEG_SAM3_CKPT="/path/to/checkpoints/sam3.pt"
export FRONTVEG_DEPTH_REPO="/path/to/Depth-Anything-V2"
export FRONTVEG_DEPTH_CKPT_DIR="/path/to/depth_checkpoints_folder"
```

## Usage in Napari

1. Launch Napari: `napari`
2. Open an RGB image of a row crop.
3. Select **Plugins > FrontVeg V2 Studio**.
4. Set your prompt (e.g., "leaf") and adjust parameters (Tile Overlap, Sigma, etc.).
5. Click **Run Complete Pipeline**.

## Citing FrontVeg V2

```bibtex
@inproceedings{adjalil2026frontveg,
  title={FrontVeg-SAM3: Foreground-Aware Zero-Shot Plant Trait Segmentation in Trellised Crops Using Side View Monocular RGB Images},
  author={A-D. Ousseini Hamza, et al.}, 
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)}, 
  year={2026}
}
```

## License
License is pending.
