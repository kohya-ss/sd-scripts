"""Tests for the optional-OpenCV fallback (library/cv2_compat.py + library/_cv2_stub).

The equivalence tests need the real opencv-python and are skipped without it.
The registration test runs in a subprocess with ``cv2`` blocked so it works
either way.
"""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from library import _cv2_stub as stub

real_cv2 = pytest.importorskip("cv2")
if getattr(real_cv2, "_IS_CV2_STUB", False):
    real_cv2 = None

needs_opencv = pytest.mark.skipif(real_cv2 is None, reason="real opencv-python not installed")


def _maxdiff(a: np.ndarray, b: np.ndarray) -> float:
    assert a.shape == b.shape, (a.shape, b.shape)
    assert a.dtype == b.dtype, (a.dtype, b.dtype)
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())


def test_stub_is_registered_when_opencv_is_missing():
    code = textwrap.dedent(
        """
        import importlib.abc, sys
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name == "cv2" or name.startswith("cv2."):
                    raise ImportError("blocked")
        sys.meta_path.insert(0, Block())
        import library.utils  # imports cv2_compat before cv2
        import cv2
        from library import cv2_compat
        assert cv2_compat.HAS_OPENCV is False
        assert getattr(cv2, "_IS_CV2_STUB", False) is True
        import numpy as np
        img = np.zeros((4, 6, 3), dtype=np.uint8)
        assert cv2.resize(img, (3, 2), interpolation=cv2.INTER_AREA).shape == (2, 3, 3)
        print("stub-ok")
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    assert "stub-ok" in result.stdout


def test_require_opencv_raises_without_opencv(monkeypatch):
    from library import cv2_compat

    monkeypatch.setattr(cv2_compat, "HAS_OPENCV", False)
    with pytest.raises(SystemExit):
        cv2_compat.require_opencv("tools/canny.py")
    monkeypatch.setattr(cv2_compat, "HAS_OPENCV", True)
    cv2_compat.require_opencv("tools/canny.py")  # no-op


@needs_opencv
@pytest.mark.parametrize(
    "code_name, channels, tol",
    [
        ("COLOR_BGR2RGB", 3, 0),
        ("COLOR_RGB2BGR", 3, 0),
        ("COLOR_BGRA2RGBA", 4, 0),
        ("COLOR_RGBA2BGRA", 4, 0),
        ("COLOR_BGR2GRAY", 3, 1),
        ("COLOR_RGB2GRAY", 3, 1),
        ("COLOR_BGR2HSV", 3, 1),
        ("COLOR_RGB2HSV", 3, 1),
    ],
)
def test_cvtcolor_matches_opencv(code_name, channels, tol):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(37, 53, channels), dtype=np.uint8)
    # include pure hues and greys, where hue wrap-around / zero saturation matter
    img[0, :3] = [[255, 0, 0], [0, 255, 0], [0, 0, 255]] if channels == 3 else [[255, 0, 0, 7], [0, 255, 0, 7], [0, 0, 255, 7]]
    img[1, :2, :3] = [[0, 0, 0], [255, 255, 255]]
    expected = real_cv2.cvtColor(img, getattr(real_cv2, code_name))
    got = stub.cvtColor(img, getattr(stub, code_name))
    assert _maxdiff(expected, got) <= tol


@needs_opencv
def test_hsv_to_bgr_matches_opencv():
    rng = np.random.default_rng(1)
    hsv = np.stack(
        [
            rng.integers(0, 180, (37, 53), dtype=np.uint8),
            rng.integers(0, 256, (37, 53), dtype=np.uint8),
            rng.integers(0, 256, (37, 53), dtype=np.uint8),
        ],
        axis=-1,
    )
    assert _maxdiff(real_cv2.cvtColor(hsv, real_cv2.COLOR_HSV2BGR), stub.cvtColor(hsv, stub.COLOR_HSV2BGR)) <= 2
    assert _maxdiff(real_cv2.cvtColor(hsv, real_cv2.COLOR_HSV2RGB), stub.cvtColor(hsv, stub.COLOR_HSV2RGB)) <= 2


@needs_opencv
def test_gray2bgr_matches_opencv():
    gray = np.random.default_rng(2).integers(0, 256, (11, 13), dtype=np.uint8)
    assert _maxdiff(real_cv2.cvtColor(gray, real_cv2.COLOR_GRAY2BGR), stub.cvtColor(gray, stub.COLOR_GRAY2BGR)) == 0


@needs_opencv
@pytest.mark.parametrize("interp", ["INTER_AREA", "INTER_LINEAR"])
def test_resize_area_linear_match_opencv_random_sizes(interp):
    """AREA and LINEAR are re-implemented in NumPy and must match OpenCV up to rounding,
    for shrinking, enlarging and mixed scaling, and for 1/3/4-channel images."""
    rng = np.random.default_rng(3)
    for _ in range(150):
        h, w = int(rng.integers(1, 80)), int(rng.integers(1, 80))
        nh, nw = int(rng.integers(1, 90)), int(rng.integers(1, 90))
        c = int(rng.choice([0, 1, 3, 4]))
        img = rng.integers(0, 256, size=(h, w) if c == 0 else (h, w, c), dtype=np.uint8)
        expected = real_cv2.resize(img, (nw, nh), interpolation=getattr(real_cv2, interp))
        got = stub.resize(img, (nw, nh), interpolation=getattr(stub, interp))
        assert _maxdiff(expected, got) <= 1, (img.shape, (nh, nw))


@needs_opencv
def test_resize_area_large_image_exact():
    rng = np.random.default_rng(4)
    img = rng.integers(0, 256, size=(1536, 2048, 3), dtype=np.uint8)
    expected = real_cv2.resize(img, (1024, 768), interpolation=real_cv2.INTER_AREA)
    got = stub.resize(img, (1024, 768), interpolation=stub.INTER_AREA)
    assert _maxdiff(expected, got) <= 1


def test_resize_rgba_channels_are_independent():
    """Pillow premultiplies alpha when resizing RGBA; the stub must not (OpenCV does not)."""
    rng = np.random.default_rng(5)
    rgb = rng.integers(0, 256, size=(40, 60, 3), dtype=np.uint8)
    alpha_a = np.full((40, 60, 1), 255, dtype=np.uint8)
    alpha_b = rng.integers(0, 256, size=(40, 60, 1), dtype=np.uint8)
    for interp in (stub.INTER_AREA, stub.INTER_LINEAR, stub.INTER_CUBIC, stub.INTER_LANCZOS4):
        out_a = stub.resize(np.concatenate([rgb, alpha_a], axis=-1), (30, 20), interpolation=interp)
        out_b = stub.resize(np.concatenate([rgb, alpha_b], axis=-1), (30, 20), interpolation=interp)
        assert np.array_equal(out_a[..., :3], out_b[..., :3])


def test_resize_shapes_and_dtypes():
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    assert stub.resize(img, (5, 4), interpolation=stub.INTER_AREA).shape == (4, 5, 3)
    assert stub.resize(np.zeros((10, 20), np.uint8), (5, 4), interpolation=stub.INTER_LINEAR).shape == (4, 5)
    # single-channel with trailing dim comes back 2-D, like OpenCV
    assert stub.resize(np.zeros((10, 20, 1), np.uint8), (5, 4), interpolation=stub.INTER_AREA).shape == (4, 5)
    f32 = np.random.default_rng(6).random((16, 16), dtype=np.float32)
    out = stub.resize(f32, (64, 64), interpolation=stub.INTER_CUBIC)
    assert out.shape == (64, 64) and out.dtype == np.float32
    # same size returns a copy, never the input itself
    same = stub.resize(img, (20, 10), interpolation=stub.INTER_AREA)
    assert same is not img and same.shape == img.shape
    # fx / fy form
    assert stub.resize(img, None, fx=0.5, fy=0.5, interpolation=stub.INTER_AREA).shape == (5, 10, 3)


def test_imwrite_roundtrip(tmp_path):
    from PIL import Image

    bgr = np.random.default_rng(8).integers(0, 256, size=(8, 9, 3), dtype=np.uint8)
    path = tmp_path / "x.png"
    assert stub.imwrite(str(path), bgr)
    rgb = np.asarray(Image.open(path).convert("RGB"))
    assert np.array_equal(rgb, bgr[..., ::-1])
