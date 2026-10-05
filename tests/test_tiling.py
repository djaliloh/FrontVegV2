import numpy as np
import pytest

from frontveg.core.tiling import (
    box_iou_matrix,
    count_instances,
    extract_fragments,
    paint_winner_takes_all,
    stride_from_overlap,
    tile_origins,
)


@pytest.mark.parametrize(
    ("length", "tile", "stride", "expected"),
    [
        (960, 640, 320, [0, 320]),  # 320 + 640 reaches the border: no tile at 640
        (1000, 640, 320, [0, 320, 640]),  # Previously also 960, a 40 px redundant tile
        (500, 640, 320, [0]),  # Image smaller than one tile
        (640, 640, 320, [0]),
    ],
)
def test_tile_origins_stop_at_border(length, tile, stride, expected):
    origins = tile_origins(length, tile, stride)
    assert origins == expected
    assert origins[-1] + tile >= length


@pytest.mark.parametrize("overlap", [-0.1, 1.0, 1.5])
def test_stride_rejects_invalid_overlap(overlap):
    with pytest.raises(ValueError):
        stride_from_overlap(640, overlap)


def test_stride_from_overlap():
    assert stride_from_overlap(640, 0.5) == 320
    assert stride_from_overlap(640, 0.0) == 640


def test_box_iou_matrix():
    iou = box_iou_matrix([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    assert iou[0, 0] == pytest.approx(1.0)
    assert iou[0, 1] == pytest.approx(50 / 150)
    assert iou[0, 2] == 0
    np.testing.assert_allclose(iou, iou.T)


def test_extract_fragments_filters_and_offsets():
    masks = np.zeros((3, 1, 100, 100), dtype=np.float32)
    masks[0, 0, 10:20, 30:50] = 1  # Valid
    masks[1, 0, 0:5, 0:5] = 1  # Too small (25 px)
    masks[2, 0] = 1  # Covers the whole tile
    frags = extract_fragments(masks, [0.9, 0.8, 0.7], (200, 100), (100, 100), 50, 0.55)
    assert len(frags) == 1
    assert frags[0]["bbox"] == [230, 110, 249, 119]
    assert frags[0]["mask"].shape == (10, 20)


def test_paint_keeps_highest_score():
    low = {"bbox": [0, 0, 9, 9], "mask": np.ones((10, 10), bool), "score": 0.5}
    high = {"bbox": [5, 5, 14, 14], "mask": np.ones((10, 10), bool), "score": 0.9}
    id_map = paint_winner_takes_all([low, high], [1, 2], 20, 20)
    assert id_map[7, 7] == 2
    assert id_map[0, 0] == 1
    assert count_instances(id_map) == 2


def test_paint_clips_to_image():
    frag = {"bbox": [15, 15, 24, 24], "mask": np.ones((10, 10), bool), "score": 0.9}
    id_map = paint_winner_takes_all([frag], [1], 20, 20)
    assert id_map[15:, 15:].all()
