# Batch processing

The command-line tools process whole folders of images with the same pipeline as the napari plugin. They are installed with the package.

| Command | Pipeline |
|---|---|
| `frontveg` | Full pipeline: FrontVeg foreground + SAM3 instances + fusion |
| `frontveg-depth` | FrontVeg foreground masks only (no SAM3, no GPU required) |
| `frontveg-sam3` | SAM3 tiled segmentation only (no foreground filtering) |

From a source checkout without installation, `python main_final.py` is equivalent to `frontveg`.

## Step 1. Organize the images

Put the images in one folder, or in one subfolder per acquisition (plot, date, camera...). Each subfolder is processed separately and its depth maps share the same normalization.

```text
data/
├── plot_A/
│   ├── IMG_0001.JPG
│   └── IMG_0002.JPG
└── plot_B/
    └── IMG_0101.CR3
```

Supported formats: PNG, JPEG, TIFF, BMP and, with `pip install "frontvegV2[raw]"`, CR3, CR2, NEF, ARW and DNG.

## Step 2. Run

```bash
frontveg --input data --output results --prompt leaf
```

Useful options:

```bash
# Grapes, more tile overlap, verbose logs
frontveg --input data --output results --prompt grape --overlap 0.6 -v

# Foreground masks only, with histogram plots
frontveg-depth --input data --output results_depth --plot
```

Progress is logged with timestamps. The command ends with the number of processed images and lists the failed ones; its exit code is `0` on success, `1` if some images failed and `2` if resources are missing.

## Step 3. Read the outputs

```text
results/
├── final_masks/<folder>/<image>.png                         # Binary mask of the final objects
├── colored_finalid_overlay_input_rgb/<folder>/<image>.png   # Colored objects over the input image
├── sam3_overlay/<folder>/<image>.png                        # Colored objects over the foreground only
├── summary/<folder>.csv                                     # One row per image
└── temp_frontveg/
    ├── depth/<folder>/<image>.png                           # Normalized depth map
    ├── masks/<folder>/<image>.png                           # FrontVeg foreground mask
    └── graphics/<folder>/hist_<image>.png                   # Histogram analysis (unless --no_plots)
```

All outputs are lossless PNG files named after the input image, whatever its format. `summary/<folder>.csv` contains `image`, `instances`, `total_area_px` and `mean_area_px`.

## Options

Run `frontveg --help` for the full list.

| Option | Default | Description |
|---|---|---|
| `--input` | required | Image folder |
| `--output` | `output_final` | Output folder |
| `--prompt` | `leaf` | SAM3 text prompt |
| `--sigma` | `1.1` | Histogram smoothing |
| `--peak_dist` | `8` | Minimal distance between histogram peaks (bins) |
| `--peak_height` | `0.5` | Minimal height of histogram peaks |
| `--smoothed` / `--no-smoothed` | smoothed | Smooth the histogram before peak detection |
| `--encoder` | `vitl` | Depth-Anything-V2 encoder (`vits`, `vitb`, `vitl`, `vitg`) |
| `--tile_size` | `640` | SAM3 tile side, in pixels |
| `--overlap` | `0.5` | Overlap between tiles, in `[0, 1)` |
| `--conf_threshold` | `0.35` | SAM3 confidence threshold |
| `--device` | auto | `cuda` or `cpu` |
| `--depth_repo_path`, `--depth_ckpt_path`, `--sam3_repo_path`, `--sam3_ckpt` | auto | Override the [resource locations](installation.md#alternative-install-from-pypi) |
| `--no_plots` | off | Skip the histogram plots |
| `-v`, `--verbose` | off | Debug messages |

## Use from Python

```python
from frontveg.config import FrontVegConfig
from frontveg.master_pipeline import MasterPipeline
from frontveg.utils.image_loader import ImageLoader

pipeline = MasterPipeline(FrontVegConfig(prompt="grape", overlap=0.6))  # Loads the models once

image = ImageLoader.load_image("data/plot_A/IMG_0001.JPG")  # RGB uint8 array
result = pipeline.segment(image)

print(len(result.instance_areas), "grapes")
result.id_map  # uint16 labels, 0 = background
result.mask  # uint8 binary mask (0/255)
result.foreground  # FrontVeg foreground mask
```

`segment()` accepts `prompt`, `sigma`, `peak_dist` and `overlap` overrides for a single call; `run_full_stack(input_dir, output_dir)` reproduces the `frontveg` command.
