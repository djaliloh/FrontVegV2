# Installation

This guide installs FrontVeg V2 with its two external models, **Depth-Anything-V2** and **SAM3**, in about 15 minutes (downloads excluded).

## Requirements

| | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.10 or 3.11 |
| GPU | Not required for depth estimation | NVIDIA GPU with CUDA 12.x and 8 GB VRAM or more |
| Disk space | 6 GB (environment and checkpoints) | 10 GB |
| Tools | Git, Conda (or `venv`) | |

> [!NOTE]
> SAM3 is designed for NVIDIA GPUs. On a CPU-only machine the depth stage runs, but SAM3 segmentation is very slow or may fail.

## Step 1. Create a Python environment

```bash
conda create -n frontveg python=3.10 -y
conda activate frontveg
```

Install PyTorch for your CUDA version (see [pytorch.org](https://pytorch.org/get-started/locally/)), for example CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Step 2. Clone the repository with its submodules

SAM3 and Depth-Anything-V2 are Git submodules pinned to tested versions.

```bash
git clone --recursive https://github.com/djaliloh/FrontVegV2.git
cd FrontVegV2
```

> [!TIP]
> Already cloned without `--recursive`? Run `git submodule update --init`.

## Step 3. Install SAM3

```bash
pip install -e external/sam3
```

Depth-Anything-V2 needs no installation: FrontVeg V2 loads it from `external/Depth-Anything-V2`.

## Step 4. Download the checkpoints

**4.1 Depth-Anything-V2 Large** (1.3 GB, public):

```bash
curl -L -o checkpoints/depthanything_ckpts/depth_anything_v2_vitl.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth
```

**4.2 SAM3** (gated):

1. Request access on the [facebook/sam3](https://huggingface.co/facebook/sam3) model page.
2. Once approved, log in and download `sam3.pt`:

```bash
pip install -U huggingface_hub
huggingface-cli login
huggingface-cli download facebook/sam3 sam3.pt --local-dir checkpoints/sam3_ckpts
```

## Step 5. Install FrontVeg V2

```bash
pip install -e .
```

Optional extras:

| Extra | Command | Adds |
|---|---|---|
| `raw` | `pip install -e ".[raw]"` | Camera RAW files (CR3, CR2, NEF, ARW, DNG) |
| `dev` | `pip install -e ".[dev]"` | Tests, linting, documentation tools |

## Step 6. Check the installation

Your folder should look like this:

```text
FrontVegV2/
├── checkpoints/
│   ├── depthanything_ckpts/
│   │   └── depth_anything_v2_vitl.pth
│   └── sam3_ckpts/
│       └── sam3.pt
├── external/
│   ├── Depth-Anything-V2/
│   └── sam3/
└── src/
```

Then run:

```bash
python -c "from frontveg.resources import resolve_paths; print(resolve_paths().missing('vitl') or 'All resources found')"
frontveg --help
```

The first command prints `All resources found`, or the list of missing files with the path where FrontVeg V2 looked for them.

## Alternative: install from PyPI

If you only need the napari plugin, you can install the published package and keep the models in any folder:

```bash
pip install frontvegV2
```

Then tell FrontVeg V2 where the `external/` and `checkpoints/` folders are, either with one variable:

```bash
# Linux / macOS
export FRONTVEG_ROOT=/path/to/folder/with/external/and/checkpoints
# Windows (PowerShell)
$env:FRONTVEG_ROOT = "C:\path\to\folder\with\external\and\checkpoints"
```

or with one variable per resource:

| Variable | Points to |
|---|---|
| `FRONTVEG_DEPTH_REPO` | `Depth-Anything-V2` repository |
| `FRONTVEG_DEPTH_CKPT_DIR` | Folder holding `depth_anything_v2_vitl.pth` |
| `FRONTVEG_SAM3_REPO` | `sam3` repository (not needed if SAM3 is pip-installed) |
| `FRONTVEG_SAM3_CKPT` | `sam3.pt` file |

Without these variables, FrontVeg V2 looks for a folder containing both `external/` and `checkpoints/`: next to the installed package, then in the current folder and its parents, then in common locations of your home folder (`Desktop`, `Documents`, `Downloads`, ...).

Next: [use the napari plugin](napari-guide.md) or [process folders from the command line](batch-processing.md).
