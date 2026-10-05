import numpy as np
import pytest

from frontveg.core.hist_processor import HistogramProcessor
from frontveg.core.post_processor import PostProcessor
from frontveg.core.valley_processor import ValleyProcessor


def gaussian_histogram(modes):
    x = np.arange(256)
    return sum(h * np.exp(-((x - m) ** 2) / (2 * 8**2)) for m, h in modes).astype(np.float32)


THREE_MODES = gaussian_histogram([(40, 800), (120, 500), (210, 900)])


def test_valley_between_second_and_third_modes():
    result = ValleyProcessor(sigma=1.1, peak_dist=8).find_optimal_threshold(THREE_MODES)
    assert result is not None
    assert len(result["peaks"]) == 3
    assert 140 < result["optimal_threshold"] < 190


def test_valley_returns_none_with_two_modes():
    hist = gaussian_histogram([(60, 500), (180, 500)])
    assert ValleyProcessor().find_optimal_threshold(hist) is None


def test_valley_overrides_do_not_change_instance():
    processor = ValleyProcessor(sigma=1.1, peak_dist=8)
    processor.find_optimal_threshold(THREE_MODES, sigma=3.0, peak_dist=2)
    assert (processor.sigma, processor.peak_dist) == (1.1, 8)


def test_histogram_ignores_zero_depth():
    depth = np.array([[0, 0, 255, 255]], dtype=np.uint8)
    clean, hist = HistogramProcessor().normalize_and_hist(depth, 255)
    assert hist.sum() == 2 and hist[255] == 2
    np.testing.assert_array_equal(clean, depth)


def test_histogram_handles_zero_reference():
    clean, hist = HistogramProcessor().normalize_and_hist(np.zeros((4, 4), np.uint8), 0)
    assert hist.sum() == 0 and clean.max() == 0


def test_fusion_keeps_ids_inside_foreground_only():
    ids = np.zeros((10, 10), np.uint16)
    ids[:, :5] = 4
    ids[:, 5:] = 9
    foreground = np.zeros((5, 5), np.uint8)  # Smaller: resized to the id map
    foreground[:, :3] = 255
    final_ids, mask = PostProcessor.get_final_colored_map(ids, foreground)
    assert set(np.unique(final_ids)) == {0, 4, 9}
    assert final_ids[:, 6:].max() == 0
    assert mask.dtype == np.uint8 and set(np.unique(mask)) == {0, 255}


@pytest.mark.parametrize("shape", [(10, 10), (5, 5)])
def test_apply_overlay_blacks_out_background(shape):
    rgb = np.full((10, 10, 3), 200, np.uint8)
    mask = np.zeros(shape, np.uint8)
    mask[: shape[0] // 2] = 255
    out = PostProcessor.apply_overlay(rgb, mask)
    assert out[0, 0].tolist() == [200, 200, 200]
    assert out[-1, -1].tolist() == [0, 0, 0]
