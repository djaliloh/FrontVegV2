"""FrontVeg V2 napari plugin: one-click organ segmentation of the front crop row."""

# No "from __future__ import annotations": magicgui resolves the widget annotations at runtime.
import csv
import logging
import re
import time
from pathlib import Path

import napari
import numpy as np
from magicgui import magicgui
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_error, show_info, show_warning

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path.home() / "FrontVegV2_results"

SETUP_HELP = (
    "See the installation guide: clone the external repositories into 'external/', "
    "download the checkpoints into 'checkpoints/', or set FRONTVEG_ROOT to your "
    "FrontVeg V2 folder."
)

# Heavy models are loaded once per napari session.
_STATE: dict = {}


def get_pipeline():
    """Load (once) and return the shared MasterPipeline."""
    if "pipeline" not in _STATE:
        from frontveg.master_pipeline import MasterPipeline

        _STATE["pipeline"] = MasterPipeline()
    return _STATE["pipeline"]


def safe_name(text: str) -> str:
    """File-system safe version of a layer name or prompt."""
    return re.sub(r"[^\w.-]+", "_", str(text)).strip("_") or "image"


def layer_to_rgb(layer) -> np.ndarray:
    """RGB uint8 array of a 2D napari image layer (multiscale: highest resolution)."""
    from frontveg.utils.image_loader import to_rgb_uint8

    data = np.asarray(layer.data[0] if layer.multiscale else layer.data)
    if data.ndim != (3 if layer.rgb else 2):
        raise ValueError(f"Select a single 2D image, got an array of shape {data.shape}.")
    return to_rgb_uint8(data)


def save_statistics(path: Path, areas: dict[int, int], prompt, sigma, peak_dist, overlap) -> None:
    """One row per instance, plus the run parameters."""
    total = sum(areas.values())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "ID_Instance",
                "Count",
                "Total_Area_Pixels",
                "Surface_Pixels",
                "Prompt",
                "Sigma",
                "Peak_Dist",
                "Tile_Overlap",
            ]
        )
        for inst_id, area in areas.items():
            writer.writerow([inst_id, len(areas), total, area, prompt, sigma, peak_dist, overlap])


def save_masks(folder: Path, base: str, result) -> list[Path]:
    """Binary mask and color-coded instance mask as PNG."""
    from frontveg.utils.visualization import instance_palette, write_image

    binary = write_image(folder / f"{base}_mask_binary.png", result.mask)
    colored = instance_palette(int(result.id_map.max()) + 1)[result.id_map]
    return [binary, write_image(folder / f"{base}_mask_colored.png", colored)]


def _upsert_layer(viewer, data, name: str, **kwargs) -> None:
    """Replace the data of an existing labels layer, or add a new one."""
    if name in viewer.layers:
        viewer.layers[name].data = data
    else:
        viewer.add_labels(data, name=name, **kwargs)


def make_frontveg_widget():
    @thread_worker
    def run(image_rgb, prompt, sigma, peak_dist, overlap):
        if "pipeline" not in _STATE:
            yield "Loading Depth-Anything-V2 and SAM3 (first run only)..."
        pipeline = get_pipeline()
        yield f"Segmenting '{prompt}'..."
        return pipeline.segment(
            image_rgb, prompt=prompt, sigma=sigma, peak_dist=peak_dist, overlap=overlap
        )

    @magicgui(
        call_button="Run Complete Pipeline",
        prompt={"label": "Object to segment", "widget_type": "LineEdit"},
        tile_overlap={"label": "Tile overlap (SAM3)", "min": 0.1, "max": 0.9, "step": 0.05},
        sigma={"label": "Sigma (depth)", "min": 0.1, "max": 5.0, "step": 0.1},
        peak_dist={"label": "Peak distance (depth)", "min": 1, "max": 20, "step": 1},
        show_foreground={"label": "Show FrontVeg foreground"},
        auto_save={"label": "Save CSV (statistics)"},
        save_mask={"label": "Save masks (PNG)"},
        output_dir={"label": "Output folder", "mode": "d"},
    )
    def widget(
        viewer: napari.Viewer,
        image: napari.layers.Image,
        prompt: str = "leaf",
        tile_overlap: float = 0.5,
        sigma: float = 1.1,
        peak_dist: int = 8,
        show_foreground: bool = False,
        auto_save: bool = False,
        save_mask: bool = False,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
    ) -> None:
        if image is None:
            show_error("Open an RGB image first.")
            return
        prompt = prompt.strip()
        if not prompt:
            show_error("Enter the object to segment, e.g. 'leaf'.")
            return
        try:
            image_rgb = layer_to_rgb(image)
        except ValueError as e:
            show_error(str(e))
            return

        start = time.time()
        worker = run(image_rgb, prompt, sigma, peak_dist, tile_overlap)

        def on_result(result):
            if result.depth is not None and result.depth.used_fallback:
                show_warning("No depth valley found: FrontVeg used the default threshold.")
            if show_foreground:
                _upsert_layer(viewer, result.foreground // 255, "FrontVeg foreground", opacity=0.4)
            areas = result.instance_areas
            if not areas:
                show_info(f"No '{prompt}' detected in the foreground.")
                return
            _upsert_layer(viewer, result.id_map, f"Result_{prompt}", opacity=0.8)

            values = np.fromiter(areas.values(), dtype=float)
            show_info(
                f"{prompt}: {len(areas)} instance(s), mean area {values.mean():.0f} px "
                f"(std {values.std():.0f} px), {time.time() - start:.1f} s"
            )

            base = f"{safe_name(image.name)}_{safe_name(prompt)}"
            if auto_save or save_mask:
                output_dir.mkdir(parents=True, exist_ok=True)
            if auto_save:
                csv_path = output_dir / f"{base}_stats.csv"
                save_statistics(csv_path, areas, prompt, sigma, peak_dist, tile_overlap)
                show_info(f"Statistics saved to {csv_path}")
            if save_mask:
                save_masks(output_dir, base, result)
                show_info(f"Masks saved to {output_dir}")

        def on_error(error):
            logger.error("FrontVeg V2 failed", exc_info=error)
            from frontveg.master_pipeline import MissingResourceError

            hint = f"\n\n{SETUP_HELP}" if isinstance(error, MissingResourceError) else ""
            show_error(f"FrontVeg V2 failed: {error}{hint}")

        def on_finished():
            widget.call_button.enabled = True
            widget.call_button.text = "Run Complete Pipeline"

        worker.yielded.connect(show_info)
        worker.returned.connect(on_result)
        worker.errored.connect(on_error)
        worker.finished.connect(on_finished)
        widget.call_button.enabled = False
        widget.call_button.text = "Running..."
        worker.start()

    return widget
