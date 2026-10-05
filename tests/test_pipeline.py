import numpy as np
from PIL import Image

from frontveg.config import FrontVegConfig
from frontveg.core.pipeline import FALLBACK_THRESHOLD, FrontVegPipeline


def make_pipeline():
    return FrontVegPipeline(FrontVegConfig(), depth_repo_path="repo", depth_ckpt_path="ckpt")


def test_predict_keeps_front_row(fake_depth_module):
    mask = make_pipeline().predict(np.zeros((60, 90, 3), np.uint8))
    assert mask.dtype == np.uint8
    assert mask[:, 65:].mean() > 250  # Front row (closest) is foreground
    assert mask[:, :25].max() == 0


def test_fallback_threshold_when_no_valley(fake_depth_module):
    result = make_pipeline().threshold_depth(np.full((20, 20), 200, np.uint8), 255)
    assert result.used_fallback and result.threshold == FALLBACK_THRESHOLD


def test_process_folder_writes_png_for_every_format(fake_depth_module, tmp_path):
    src = tmp_path / "plot_1"
    src.mkdir()
    for name in ["a.jpg", "b.png"]:
        Image.fromarray(np.full((60, 90, 3), 100, np.uint8)).save(src / name)
    (src / "broken.png").write_bytes(b"x")  # Skipped with a warning

    masks = make_pipeline().process_folder(src, tmp_path / "out", plot_graphics=True)
    assert sorted(p.name for p in masks.values()) == ["a.png", "b.png"]
    assert all(p.exists() for p in masks.values())
    assert (tmp_path / "out" / "depth" / "plot_1" / "a.png").exists()
    assert (tmp_path / "out" / "graphics" / "plot_1" / "hist_a.png").exists()
