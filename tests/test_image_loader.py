import cv2
import numpy as np
import pytest
from PIL import Image

from frontveg.utils.image_loader import (
    ImageLoader,
    ImageLoadError,
    convert_output_filename,
    get_image_files,
    to_rgb_uint8,
)


@pytest.mark.parametrize("ext", [".png", ".jpg", ".tif", ".bmp"])
def test_load_image_returns_rgb(tmp_path, ext):
    path = tmp_path / f"red{ext}"
    Image.fromarray(np.full((8, 8, 3), [255, 0, 0], np.uint8)).save(path)

    rgb = ImageLoader.load_image(path)
    assert rgb.shape == (8, 8, 3) and rgb.dtype == np.uint8
    assert rgb[4, 4, 0] > 200 and rgb[4, 4, 2] < 50  # Red stays in channel 0
    assert ImageLoader.load_image(path, return_format="bgr")[4, 4, 2] > 200


def test_load_grayscale_and_16bit(tmp_path):
    gray = tmp_path / "gray.png"
    Image.fromarray(np.full((4, 4), 128, np.uint8)).save(gray)
    assert ImageLoader.load_image(gray).shape == (4, 4, 3)

    deep = tmp_path / "deep.png"
    cv2.imwrite(str(deep), np.full((4, 4), 65535, np.uint16))
    assert ImageLoader.load_image(deep).max() == 255


def test_load_missing_or_corrupt_file_raises(tmp_path):
    with pytest.raises(ImageLoadError):
        ImageLoader.load_image(tmp_path / "missing.png")
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(ImageLoadError):
        ImageLoader.load_image(bad)


@pytest.mark.parametrize(
    "array",
    [
        np.full((2, 2), 255, np.uint8),
        np.full((2, 2, 4), 255, np.uint8),
        np.full((2, 2, 3), 1.0, np.float32),
        np.full((2, 2, 3), 65535, np.uint16),
        np.ones((2, 2), bool),
    ],
)
def test_to_rgb_uint8(array):
    rgb = to_rgb_uint8(array)
    assert rgb.shape == (2, 2, 3) and rgb.dtype == np.uint8 and rgb.max() == 255


def test_to_rgb_uint8_rejects_stacks():
    with pytest.raises(ValueError):
        to_rgb_uint8(np.zeros((2, 4, 4, 3), np.uint8))


@pytest.mark.parametrize("name", ["a.CR3", "a.jpg", "a.JPEG", "a.png", "a.tif"])
def test_output_filename_is_always_png(name):
    assert convert_output_filename(name) == "a.png"


def test_get_image_files_is_case_insensitive(tmp_path):
    for name in ["b.JPG", "a.png", "c.Tif", "notes.txt"]:
        (tmp_path / name).touch()
    assert [p.name for p in get_image_files(tmp_path)] == ["a.png", "b.JPG", "c.Tif"]
