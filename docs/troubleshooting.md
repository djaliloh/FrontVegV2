# Troubleshooting

## Installation

**`Missing FrontVeg V2 resources: ...`**
The message lists each missing file and where it was looked for. Check the [folder layout](installation.md#step-6-check-the-installation), or point FrontVeg V2 to your files with `FRONTVEG_ROOT` or the per-resource [environment variables](installation.md#alternative-install-from-pypi).

**`Depth-Anything-V2 repository: .../external/Depth-Anything-V2`** although the folder exists
The submodule folder is empty. Run `git submodule update --init`.

**`Could not import sam3`**
Install SAM3 in the active environment: `pip install -e external/sam3`.

**`No module named 'triton'`**
Install the Triton build for your platform: `pip install triton-windows` on Windows, `pip install triton` on Linux.

**`Reading .CR3 files requires rawpy`**
Install the RAW extra: `pip install "frontvegV2[raw]"`.

## Running

**`RuntimeError: mat1 and mat2 must have the same dtype`**
Mixed precision is not supported by your GPU. FrontVeg V2 uses float16 on NVIDIA GPUs of compute capability 7.0 or newer (Volta and later) and float32 otherwise; update PyTorch and the NVIDIA driver if the error persists.

**`CUDA out of memory`**
Close other GPU applications and run again. Images are processed one at a time, so the batch size is not a factor.

**SAM3 is very slow**
Check that PyTorch sees the GPU: `python -c "import torch; print(torch.cuda.is_available())"`. If it prints `False`, reinstall PyTorch for your CUDA version.

**Warning *"No depth valley found"***
The depth histogram has fewer than three modes, so FrontVeg used the middle of the depth range. See the [tuning guide](parameters.md#tuning-guide).

**Nothing detected**
Try a more generic prompt (`leaf`, `fruit`), check the foreground overlay, and lower `--conf_threshold`.

## napari

**The plugin is missing from the Plugins menu**
Make sure napari runs in the environment where FrontVeg V2 is installed (`pip show frontvegV2`), then restart napari.

**Black window or OpenGL errors on Linux**
Preload the system OpenGL library: `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGL.so.1 napari`.

**Where are my files?**
In the **Output folder** shown in the panel (default `~/FrontVegV2_results`). The notification after the run gives the exact path.

Still stuck? [Open an issue](https://github.com/djaliloh/FrontVegV2/issues) with the full error message, your OS, GPU and `pip list` output.
