import numpy as np
from conftest import FakePredictor, OriginImage

from frontveg.core.nms_processor import NMSProcessor
from frontveg.core.sam3_full_inference import SAM3FullImageInference
from frontveg.core.sam3_standard_tiler import SAM3StandardTiledInference
from frontveg.core.sam3_tiler import SAM3TiledInference

OBJECTS = [(250, 100, 400, 160), (30, 30, 80, 80)]  # One leaf crosses a tile border


def test_tiled_merges_object_cut_by_tiles():
    predictor = FakePredictor(OBJECTS)
    id_map, n = SAM3TiledInference(predictor, tile_size=320).process(OriginImage(600, 300), "leaf")
    assert n == 2
    assert len(np.unique(id_map[100:160, 250:400])) == 1
    assert predictor.released == 1


def test_tiled_returns_none_without_detection():
    tiler = SAM3TiledInference(FakePredictor([]), tile_size=320)
    assert tiler.process(OriginImage(600, 300), "leaf") == (None, 0)


def test_tiled_overlap_can_change_per_call():
    predictor = FakePredictor([])
    tiler = SAM3TiledInference(predictor, tile_size=320, overlap=0.5)
    tiler.process(OriginImage(960, 320), "leaf")
    n_default = len(predictor.calls)
    predictor.calls.clear()
    tiler.process(OriginImage(960, 320), "leaf", overlap=0.75)
    assert len(predictor.calls) > n_default


def test_standard_tiler():
    tiler = SAM3StandardTiledInference(FakePredictor(OBJECTS), tile_size=320)
    id_map, n = tiler.process(OriginImage(600, 300), "leaf")
    assert n >= 2 and id_map.shape == (300, 600)


def test_full_image_inference():
    id_map, n = SAM3FullImageInference(FakePredictor(OBJECTS)).process(
        OriginImage(600, 300), "leaf"
    )
    assert n == 2 and id_map.max() == 2


def test_nms_processor_removes_small_and_giant_instances():
    id_map = np.zeros((100, 100), np.uint16)
    id_map[:80, :80] = 1  # 64 % of the image
    id_map[90:95, 90:95] = 2  # 25 px
    id_map[85:100, 0:20] = 3  # 300 px, kept
    cleaned = NMSProcessor(min_area=200, max_area_ratio=0.55).suppress(id_map)
    assert set(np.unique(cleaned)) == {0, 3}
