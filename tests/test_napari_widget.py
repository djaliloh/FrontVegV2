import numpy as np
import pytest

napari = pytest.importorskip("napari")

import napari_frontveg  # noqa: E402
from frontveg.master_pipeline import SegmentationResult  # noqa: E402


class FakePipeline:
    def __init__(self):
        self.calls = []

    def segment(self, image_rgb, **kwargs):
        self.calls.append(kwargs)
        ids = np.zeros(image_rgb.shape[:2], np.uint16)
        ids[5:15, 5:15] = 3
        ids[20:25, 20:30] = 8
        mask = (ids > 0).astype(np.uint8) * 255
        return SegmentationResult(ids, mask, np.full_like(mask, 255))


@pytest.fixture
def fake_pipeline(monkeypatch):
    pipeline = FakePipeline()
    monkeypatch.setitem(napari_frontveg._STATE, "pipeline", pipeline)
    return pipeline


def test_widget_adds_labels_and_saves_stats(make_napari_viewer, qtbot, tmp_path, fake_pipeline):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((40, 40, 3), np.uint8), name="plot 1/a", rgb=True)
    widget = napari_frontveg.make_frontveg_widget()
    viewer.window.add_dock_widget(widget)
    widget.output_dir.value = tmp_path
    widget.auto_save.value = True
    widget.sigma.value = 2.0

    widget()
    qtbot.waitUntil(lambda: "Result_leaf" in viewer.layers, timeout=10000)
    qtbot.waitUntil(lambda: widget.call_button.enabled, timeout=10000)

    assert fake_pipeline.calls[-1]["sigma"] == 2.0
    rows = (tmp_path / "plot_1_a_leaf_stats.csv").read_text().splitlines()
    assert rows[1].startswith("3,2,150,100,leaf")
    assert rows[2].startswith("8,2,150,50,leaf")


def test_safe_name():
    assert napari_frontveg.safe_name("leaf / stem: *") == "leaf_stem"
    assert napari_frontveg.safe_name("///") == "image"
