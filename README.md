# FrontVeg V2: Foreground-Aware Zero-Shot Plant Trait Segmentation in Trellised Crops

<!-- ## Authors -->
**Abdoul Djalil Ousseini Hamza** ([ORCID](https://orcid.org/0009-0008-6172-4439)), **Herearii Metuarea** ([ORCID](https://orcid.org/0009-0008-2716-0617)), **Corentin Lothodé** ([ORCID](https://orcid.org/0000-0002-8209-317X)), **Morgane Roth** ([ORCID](https://orcid.org/0000-0002-5244-4215)), **Eric Duchêne** ([ORCID](https://orcid.org/0000-0003-2712-1892)), **Lionel Ley**, **David Alletru** ([ORCID](https://orcid.org/0009-0006-6238-6123)), and **David Rousseau**<sup>*</sup> ([ORCID](https://orcid.org/0000-0002-7935-1609))

*\* Project Supervisor* 

<!-- <img src="https://raw.githubusercontent.com/djaliloh/FrontVegV2/main/assets/logo.png" alt="Project Logo" style="max-width: 100%; height: auto;"> -->

<!-- ![Logo](https://raw.githubusercontent.com/djaliloh/FrontVegV2/main/assets/logo.png)  -->

<!-- <img src="assets/logo.png" alt="Project Logo" style="max-width: 100%; height: auto;"> -->

<!-- <img src="https://i.imgur.com/NcQWGjS.png" alt="Project Logo" style="max-width: 100%; height: auto;"> -->
<img src="https://raw.githubusercontent.com/djaliloh/Deep-learning-training/main/results/logo.png" alt="FrontVeg V2 Logo" style="max-width: 100%; height: auto;">

[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/frontvegv2)](https://napari-hub.org/plugins/frontvegv2.html) [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) 
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

<!-- uncomment this when the repos become public -->
<!-- [![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/djaliloh/FrontVegV2)](https://github.com/djaliloh/FrontVegV2) --> 



# FrontVeg V2

A napari plugin for automated plant organ segmentation and leaf area estimation in trellised crops using side view monocular RGB images.

<!-- > ⚠️ **Important:** SAM3 requires gated model weights from Hugging Face. Please follow the setup instructions below before running the plugin. -->  



<!-- [![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/frontveg)](https://napari-hub.org/plugins/frontveg) -->

---

## Prerequisites

1. **Create a virtual environment** 
   ```bash
   conda create -n <env_name> python=3.10 -y
   conda activate <env_name>
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

2. **Request SAM3 Weights Access:**
   Request access to the SAM3 checkpoint on [Hugging Face](https://huggingface.co/facebook/sam3). Once approved, download `sam3.pt`.

3. **Download Depth-Anything V2 Weights:**
   Download the Large model checkpoint (`depth_anything_v2_vitl.pth`) from the official [Depth-Anything V2 repository](https://github.com/DepthAnything/Depth-Anything-V2).

4. **Place your downloaded checkpoints inside the project folder:** 
   - Place sam3.pt into: checkpoints/sam3_ckpts/sam3.pt
   - Place depth checkpoints into: checkpoints/depthanything_ckpts/

---

## Installation

### Option 1: Recommended (Git Clone + Editable Install)

This option automatically sets up the relative paths for `external/` submodules and `checkpoints/`.

```bash
# 1. Clone the repository 
git clone https://github.com/djaliloh/FrontVegV2.git
cd FrontVegV2

# 2. Clone and install external models in the external/ folder:
cd external

# Depth-Anything V2
git clone https://github.com/DepthAnything/Depth-Anything-V2.git

# SAM3
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
cd ../..

# 3. Install in editable mode 
pip install -e .
```

### Option 2: Direct PyPI Install (Coming Soon) 

> ℹ️ **Note:** Standalone installation via `pip install` will be released soon. 
> In the meantime, please follow **Option 1** to use the plugin. 


<!-- ### Option 2: Direct PyPI Install + Environment Variables  -->

<!-- ```powershell
# If you installed the plugin directly via PyPI (pip install frontvegv2), you must specify the paths to your local SAM3 code repository and checkpoints using environment variables : 

# Windows (PowerShell):
$env:FRONTVEG_SAM3_REPO = "<path_to_sam3_repo>"
$env:FRONTVEG_SAM3_CKPT = "<path_to_sam3.pt>"
$env:FRONTVEG_DEPTH_REPO = "<path_to_depth_anything_v2_repo>"
$env:FRONTVEG_DEPTH_CKPT_DIR = "<path_to_depth_ckpts_folder>"

napari


# Linux / macOS (Bash):
export FRONTVEG_SAM3_REPO="<path_to_sam3_repo>"
export FRONTVEG_SAM3_CKPT="<path_to_sam3.pt>"
export FRONTVEG_DEPTH_REPO="<path_to_depth_anything_v2_repo>"
export FRONTVEG_DEPTH_CKPT_DIR="<path_to_depth_ckpts_folder>"

napari
``` -->



## Usage in Napari

1. Launch Napari: `napari`
2. Open an RGB image of a row crop.
3. Select **Plugins > FrontVeg V2 Studio**.
4. Set your prompt (e.g., "leaf") and adjust parameters (Tile Overlap, Sigma, etc.).
5. Click **Run Complete Pipeline**.


<!-- ## Citing FrontVeg V2

```bibtex
@article{djaliloh2026frontvegv2,
  title={FrontVeg-V2: Foreground-Aware Zero-Shot Plant Trait Segmentation in Trellised Crops Using Side View Monocular RGB Images},
  author={A-D. Ousseini Hamza, et al.}, 
  journal={SoftwareX}, 
  year={2026},
  url={https://github.com/djaliloh/FrontVeg2} 
}
```  -->

## Troubleshooting

- Missing Module 'triton': On Windows, install the Windows-compatible Triton build: pip install triton-windows.
- RuntimeError: mat1 and mat2 must have the same dtype: If running on GPUs older than NVIDIA Ampere (e.g., GTX 10xx, RTX 20xx), ensure autocast is set to float16 or float32 instead of bfloat16.
- Missing Checkpoints Error: Verify that sam3.pt exists at the expected path or set FRONTVEG_SAM3_CKPT manually.


<!-- ## License
License is pending. -->

## Acknowledgements

This work was supported by the French **Programme et Équipements Prioritaires de Recherche (PEPR) AgroEcoNum** ([pepr-agroeconum.fr](https://www.pepr-agroeconum.fr/)), the **France 2030** program through the Agence Nationale de la Recherche (ANR), and the European Union’s **Horizon Europe** research and innovation programme (Grant No. 101094587, **PHENET**). 

This research used computing resources from the **GLiCID Computing Facility** (Ligerien Group for Intensive Distributed Computing, [doi:10.60487/glicid](https://doi.org/10.60487/glicid), Pays de la Loire, France), as well as HPC and storage resources provided by **GENCI at IDRIS** on the Jean Zay supercomputer’s H100 partition (Grant 2025-AD010115553R1).

The authors would like to thank **Sirine Gharbi**, **Oumaima Karia**, **Justin Langlois**, **Paul**, and **Thomas** for their valuable assistance in annotating the image dataset used in this study.

## Contact
Imhorphen team, bioimaging research group,\\
IRHS, UMR 1345, INRAE,\\ 
42 Rue Georges Morel, 49070 Beaucouzé, France

- David Rousseau - Professor, david.rousseau@univ-angers.fr
- Corentin Lothode - Researcher Engineer, corentin.lothode@inrae.fr
- Herearii Metuarea - PhD student, herearii.metuarea@univ-angers.fr
- Abdoul Djalil Ousseini Hamza - Engineer, abdoul-djalil.ousseini-hamza@inrae.fr
