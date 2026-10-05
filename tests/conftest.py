import sys
import types

import numpy as np
import pytest


class FakePredictor:
    """SAM3 stand-in that segments fixed rectangles given in image coordinates."""

    def __init__(self, objects, score=0.9):
        self.objects = objects  # (x1, y1, x2, y2), exclusive ends
        self.score = score
        self.calls = []
        self.released = 0

    def predict_crop(self, crop, prompt):
        x0, y0 = getattr(crop, "frontveg_origin", (0, 0))
        w, h = crop.size
        self.calls.append((x0, y0, w, h))
        masks = []
        for x1, y1, x2, y2 in self.objects:
            m = np.zeros((h, w), dtype=bool)
            m[max(0, y1 - y0) : max(0, y2 - y0), max(0, x1 - x0) : max(0, x2 - x0)] = True
            if m.any():
                masks.append(m[None])
        if not masks:
            return None, None
        return np.stack(masks), np.full(len(masks), self.score, dtype=np.float32)

    def release_cache(self):
        self.released += 1


class OriginImage:
    """Minimal PIL-like image whose crops remember their origin."""

    def __init__(self, width, height):
        self.size = (width, height)

    def crop(self, box):
        x1, y1, x2, y2 = box
        tile = OriginImage(x2 - x1, y2 - y1)
        tile.frontveg_origin = (x1, y1)
        return tile


@pytest.fixture
def fake_depth_module(monkeypatch):
    """Replace the torch-based depth wrapper with a deterministic one."""

    class FakeDepth:
        def __init__(self, encoder, checkpoint_path, repo_path):
            pass

        def infer(self, rgb_image, input_size=518):
            # Three depth bands: background, middle row, front row (closest).
            h, w = rgb_image.shape[:2]
            depth = np.zeros((h, w), dtype=np.int32)
            depth[:, : w // 3] = 60
            depth[:, w // 3 : 2 * w // 3] = 120
            depth[:, 2 * w // 3 :] = 220
            noise = np.random.default_rng(0).integers(0, 12, size=(h, w))
            return np.clip(depth + noise, 1, 255).astype(np.uint8)

    module = types.ModuleType("frontveg.models.depth_wrapper")
    module.DepthAnythingWrapper = FakeDepth
    monkeypatch.setitem(sys.modules, "frontveg.models.depth_wrapper", module)
    return FakeDepth
