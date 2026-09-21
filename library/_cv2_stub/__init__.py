"""
Pure-Python / NumPy / Pillow fallback for a subset of the OpenCV (cv2) API.

This stub is automatically registered as the ``cv2`` module in ``sys.modules``
by :mod:`library.cv2_compat` when the real ``opencv-python`` package is not
installed, so that the rest of the codebase can keep using ``import cv2``
without modification.

Only the API surface actually used by the training / core-library code paths
is implemented. In particular:

- ``cvtColor`` supports BGR/RGB/BGRA/RGBA/HSV/GRAY conversions used in the
  codebase (the HSV variant follows OpenCV's H=[0,180) convention).
- ``resize`` reproduces OpenCV's ``INTER_AREA`` and ``INTER_LINEAR`` sampling
  in NumPy (these are the modes the dataset pipeline uses by default, so
  results match a real OpenCV install up to rounding). ``INTER_CUBIC``,
  ``INTER_LANCZOS4`` and ``INTER_NEAREST*`` go through Pillow's closest
  ``PIL.Image.Resampling`` filter and differ slightly from OpenCV.
- ``imshow`` displays the image through Pillow's default image viewer
  (``PIL.Image.show``). ``waitKey`` blocks on ``input()`` so that the caller
  can page through images one at a time; it returns ``ord(s[0])`` of the
  first character typed (or ``13`` for an empty Enter) so existing checks
  like ``k == ord("e")`` continue to work. ``destroyAllWindows`` is a no-op.
- ``imwrite`` is implemented via Pillow (BGR -> RGB).

Tools that rely on richer OpenCV functionality (Canny, warpAffine, face
detection, imdecode/imencode, etc.) check :data:`library.cv2_compat.HAS_OPENCV`
and bail out with a clear error message when real OpenCV is missing.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

__version__ = "0.0.0-stub"

# ---------------------------------------------------------------------------
# Constants (unique integers; values do not need to match real OpenCV)
# ---------------------------------------------------------------------------

# Color conversion codes
COLOR_BGR2RGB = 4
COLOR_RGB2BGR = 5
COLOR_BGRA2RGBA = 6
COLOR_RGBA2BGRA = 7
COLOR_BGR2GRAY = 10
COLOR_RGB2GRAY = 11
COLOR_GRAY2BGR = 12
COLOR_GRAY2RGB = 13
COLOR_BGR2HSV = 40
COLOR_HSV2BGR = 41
COLOR_RGB2HSV = 42
COLOR_HSV2RGB = 43

# Interpolation flags
INTER_NEAREST = 100
INTER_LINEAR = 101
INTER_CUBIC = 102
INTER_AREA = 103
INTER_LANCZOS4 = 104
INTER_NEAREST_EXACT = 105

# Image read flags
IMREAD_UNCHANGED = -1
IMREAD_GRAYSCALE = 0
IMREAD_COLOR = 1

# Border types (only values actually referenced are needed)
BORDER_CONSTANT = 200
BORDER_REFLECT = 201
BORDER_REPLICATE = 202

# imwrite params
IMWRITE_JPEG_QUALITY = 300


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Interpolation modes that go through Pillow. INTER_AREA and INTER_LINEAR are
# implemented in NumPy instead (see _area_resample_rows / _linear_resample_rows)
# because Pillow's filters are antialiased and do not reproduce OpenCV's output.
_INTER_TO_PIL = {
    INTER_NEAREST: Image.Resampling.NEAREST,
    INTER_NEAREST_EXACT: Image.Resampling.NEAREST,
    INTER_CUBIC: Image.Resampling.BICUBIC,
    INTER_LANCZOS4: Image.Resampling.LANCZOS,
}


def _resize_single_channel(ch: np.ndarray, dsize: Tuple[int, int], pil_interp) -> np.ndarray:
    return np.asarray(Image.fromarray(ch).resize(dsize, resample=pil_interp))


def _area_resample_rows(x: np.ndarray, new_len: int) -> np.ndarray:
    """Exact pixel-area resampling along axis 0 (OpenCV INTER_AREA when shrinking).

    Output row j is the average of the input over its footprint
    ``[j * scale, (j + 1) * scale)``, which is exactly what OpenCV computes
    when both axes shrink. Only used for ``scale > 1``.
    """
    n = x.shape[0]
    if new_len == n:
        return x
    scale = 1.0 / (new_len / n)  # same rounding as OpenCV's scale = 1 / inv_scale
    dt = _work_dtype(x, n)

    # The footprint of output row j is: a fractional first row idx_j, between
    # k-1 and k whole rows (k = floor(scale)), and a fractional last row
    # idx_{j+1}. The whole rows are summed with a fixed-width gather (rows
    # idx_j+1 .. idx_j+k) followed by a plain sum, which is much faster in
    # NumPy than cumsum or reduceat. Whenever the gather window runs past the
    # whole rows it lands on row idx_{j+1} (`extra` times), which is taken
    # out again through the last-row weight.
    bounds = np.arange(new_len + 1, dtype=np.float64) * scale
    bounds[-1] = n
    idx = np.minimum(np.floor(bounds).astype(np.int64), n - 1)
    frac = bounds - idx
    k = int(np.floor(scale))

    rows = np.minimum(idx[:-1, None] + 1 + np.arange(k, dtype=np.int64)[None, :], n - 1)
    gathered = np.take(x, rows.ravel(), axis=0)
    sums = gathered.reshape((new_len, k) + x.shape[1:]).sum(axis=1, dtype=dt)

    whole = idx[1:] - idx[:-1] - 1  # whole rows actually inside the footprint (k-1 or k)
    extra = k - whole  # how many gathered rows were row idx_{j+1} instead
    w_first = _column((1.0 - frac[:-1]).astype(dt), x.ndim)
    w_last = _column((frac[1:] - extra).astype(dt), x.ndim)
    sums += w_first * np.take(x, idx[:-1], axis=0)
    sums += w_last * np.take(x, idx[1:], axis=0)
    sums /= scale
    return sums


def _area_2tap_resample_rows(x: np.ndarray, new_len: int) -> np.ndarray:
    """OpenCV's INTER_AREA fallback used when the image is not shrunk on both axes.

    OpenCV only performs true area averaging when ``scale >= 1`` for both
    axes. Otherwise it runs its generic two-tap resampler with the
    ``area_mode`` sample positions (``sx = floor(dx * scale)``, weight taken
    from the overlap of the destination pixel with the next source pixel).
    For an enlarging axis this equals exact area averaging; for a shrinking
    axis it is a coarse two-tap approximation. Reproduced here so that mixed
    up/down scaling matches OpenCV up to rounding.
    """
    n = x.shape[0]
    if new_len == n:
        return x
    # Mirror OpenCV's arithmetic exactly (inv_scale = dst/src as double,
    # scale = 1/inv_scale, weight computed in double then truncated to float)
    # so that floating-point ties at exact pixel boundaries resolve the same way.
    inv_scale = new_len / n
    scale = 1.0 / inv_scale
    d = np.arange(new_len, dtype=np.float64)
    lo = np.floor(d * scale).astype(np.int64)
    w = ((d + 1.0) - (lo + 1) * inv_scale).astype(np.float32)
    w = np.where(w <= 0, np.float32(0), w - np.floor(w))
    w[lo >= n - 1] = 0.0
    lo[lo >= n - 1] = n - 1
    return _two_tap_rows(x, lo, w)


def _linear_resample_rows(x: np.ndarray, new_len: int) -> np.ndarray:
    """Two-tap bilinear resampling along axis 0 (OpenCV INTER_LINEAR semantics).

    Sample positions use half-pixel centres and are clamped at the borders the
    same way OpenCV does; there is no antialiasing when downscaling.
    """
    n = x.shape[0]
    if new_len == n:
        return x
    # Same arithmetic as OpenCV: position in double, truncated to float
    # before the floor, so boundary ties resolve identically.
    scale = 1.0 / (new_len / n)
    pos = ((np.arange(new_len, dtype=np.float64) + 0.5) * scale - 0.5).astype(np.float32)
    lo = np.floor(pos).astype(np.int64)
    w = pos - lo
    w[lo < 0] = 0.0
    lo[lo < 0] = 0
    w[lo >= n - 1] = 0.0
    lo[lo >= n - 1] = n - 1
    return _two_tap_rows(x, lo, w)


def _column(v: np.ndarray, ndim: int) -> np.ndarray:
    """Reshape a 1-D per-row weight vector so it broadcasts along axis 0."""
    return v.reshape((-1,) + (1,) * (ndim - 1))


def _work_dtype(x: np.ndarray, n: int) -> type:
    # float32 keeps sums of uint8 data exact as long as the total stays below
    # 2**24 (i.e. fewer than ~65k pixels along the axis); otherwise float64.
    if x.dtype == np.uint8 and n * 255 < (1 << 24):
        return np.float32
    return np.float64


def _two_tap_rows(x: np.ndarray, lo: np.ndarray, w: np.ndarray) -> np.ndarray:
    """out = x[lo] * (1 - w) + x[lo + 1] * w along axis 0 (lo + 1 clamped)."""
    n = x.shape[0]
    hi = np.minimum(lo + 1, n - 1)
    dt = np.float64 if x.dtype == np.float64 else np.float32
    w = _column(w.astype(dt), x.ndim)
    return np.take(x, lo, axis=0) * (1 - w) + np.take(x, hi, axis=0) * w


def _resample_2d(src: np.ndarray, new_h: int, new_w: int, fn) -> np.ndarray:
    """Apply a row resampler to both axes: rows first on the contiguous
    input, then on a contiguous transposed copy of the (already smaller)
    intermediate so the column pass also reduces over a contiguous block."""
    out = fn(src, new_h)
    out = np.ascontiguousarray(np.swapaxes(out, 0, 1))
    out = fn(out, new_w)
    return np.ascontiguousarray(np.swapaxes(out, 0, 1))


def _cast_like(out: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        out += 0.5
        np.floor(out, out=out)
        np.clip(out, info.min, info.max, out=out)
        return out.astype(dtype)
    return out.astype(dtype, copy=False)


def _swap_rgb_bgr_3ch(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img[..., ::-1])


def _swap_rgb_bgr_4ch(img: np.ndarray) -> np.ndarray:
    # Reverse the first three channels, keep the alpha.
    out = np.empty_like(img)
    out[..., 0] = img[..., 2]
    out[..., 1] = img[..., 1]
    out[..., 2] = img[..., 0]
    out[..., 3] = img[..., 3]
    return out


def _to_gray(img: np.ndarray, source_order: str) -> np.ndarray:
    # ITU-R BT.601 luma coefficients, matching OpenCV.
    if source_order == "BGR":
        b, g, r = img[..., 0], img[..., 1], img[..., 2]
    else:
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
    gray = 0.114 * b.astype(np.float32) + 0.587 * g.astype(np.float32) + 0.299 * r.astype(np.float32)
    return np.clip(gray + 0.5, 0, 255).astype(img.dtype if img.dtype == np.uint8 else np.uint8)


def _bgr_to_hsv_uint8(img: np.ndarray) -> np.ndarray:
    """OpenCV-compatible BGR -> HSV (uint8, H in [0,180))."""
    if img.dtype != np.uint8:
        raise NotImplementedError("HSV conversion stub only supports uint8 images")

    f = img.astype(np.float32)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    v = np.maximum(np.maximum(b, g), r)
    m = np.minimum(np.minimum(b, g), r)
    diff = v - m

    s = np.where(v > 0, (diff / np.maximum(v, 1e-8)) * 255.0, 0.0)

    h = np.zeros_like(v)
    # Avoid division by zero; where diff == 0, hue stays 0.
    safe = diff > 0
    # v == r
    mask_r = safe & (v == r)
    mask_g = safe & (v == g) & ~mask_r
    mask_b = safe & (v == b) & ~mask_r & ~mask_g

    h[mask_r] = (60.0 * (g[mask_r] - b[mask_r]) / diff[mask_r])
    h[mask_g] = (60.0 * (2.0 + (b[mask_g] - r[mask_g]) / diff[mask_g]))
    h[mask_b] = (60.0 * (4.0 + (r[mask_b] - g[mask_b]) / diff[mask_b]))
    h = np.where(h < 0, h + 360.0, h)
    # Map [0,360) -> [0,180) to match OpenCV uint8 convention. Hues that
    # round up to 180 wrap to 0 (they are the same colour), as in OpenCV.
    h = np.floor(h / 2.0 + 0.5) % 180.0

    out = np.stack(
        [
            h.astype(np.uint8),
            np.clip(s + 0.5, 0, 255).astype(np.uint8),
            np.clip(v + 0.5, 0, 255).astype(np.uint8),
        ],
        axis=-1,
    )
    return out


def _hsv_to_bgr_uint8(img: np.ndarray) -> np.ndarray:
    """OpenCV-compatible HSV -> BGR (uint8, H in [0,180))."""
    if img.dtype != np.uint8:
        raise NotImplementedError("HSV conversion stub only supports uint8 images")

    f = img.astype(np.float32)
    h = f[..., 0] * 2.0  # back to [0,360)
    s = f[..., 1] / 255.0
    v = f[..., 2]

    c = v * s
    hh = h / 60.0
    x = c * (1.0 - np.abs((hh % 2.0) - 1.0))
    m = v - c

    zeros = np.zeros_like(h)
    r = np.select(
        [hh < 1, hh < 2, hh < 3, hh < 4, hh < 5, hh < 6],
        [c, x, zeros, zeros, x, c],
        default=0,
    )
    g = np.select(
        [hh < 1, hh < 2, hh < 3, hh < 4, hh < 5, hh < 6],
        [x, c, c, x, zeros, zeros],
        default=0,
    )
    b = np.select(
        [hh < 1, hh < 2, hh < 3, hh < 4, hh < 5, hh < 6],
        [zeros, zeros, x, c, c, x],
        default=0,
    )

    out = np.stack(
        [
            np.clip(b + m + 0.5, 0, 255).astype(np.uint8),
            np.clip(g + m + 0.5, 0, 255).astype(np.uint8),
            np.clip(r + m + 0.5, 0, 255).astype(np.uint8),
        ],
        axis=-1,
    )
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cvtColor(src: np.ndarray, code: int) -> np.ndarray:
    """Subset of OpenCV's ``cv2.cvtColor`` implemented with NumPy."""
    if src is None:
        raise ValueError("cvtColor: src is None")

    if code in (COLOR_BGR2RGB, COLOR_RGB2BGR):
        if src.ndim != 3 or src.shape[2] != 3:
            raise ValueError(f"cvtColor: expected 3-channel image, got shape {src.shape}")
        return _swap_rgb_bgr_3ch(src)

    if code in (COLOR_BGRA2RGBA, COLOR_RGBA2BGRA):
        if src.ndim != 3 or src.shape[2] != 4:
            raise ValueError(f"cvtColor: expected 4-channel image, got shape {src.shape}")
        return _swap_rgb_bgr_4ch(src)

    if code == COLOR_BGR2GRAY:
        return _to_gray(src, "BGR")
    if code == COLOR_RGB2GRAY:
        return _to_gray(src, "RGB")
    if code in (COLOR_GRAY2BGR, COLOR_GRAY2RGB):
        if src.ndim == 2:
            return np.stack([src, src, src], axis=-1)
        if src.ndim == 3 and src.shape[2] == 1:
            return np.concatenate([src, src, src], axis=-1)
        raise ValueError(f"cvtColor: GRAY2BGR expected single-channel image, got shape {src.shape}")

    if code == COLOR_BGR2HSV:
        return _bgr_to_hsv_uint8(src)
    if code == COLOR_HSV2BGR:
        return _hsv_to_bgr_uint8(src)
    if code == COLOR_RGB2HSV:
        return _bgr_to_hsv_uint8(_swap_rgb_bgr_3ch(src))
    if code == COLOR_HSV2RGB:
        return _swap_rgb_bgr_3ch(_hsv_to_bgr_uint8(src))

    raise NotImplementedError(
        f"cv2 stub: cvtColor code {code} is not implemented. "
        "Install opencv-python to use this conversion."
    )


def resize(
    src: np.ndarray,
    dsize: Tuple[int, int],
    dst: Optional[np.ndarray] = None,
    fx: float = 0,
    fy: float = 0,
    interpolation: int = INTER_LINEAR,
) -> np.ndarray:
    """Subset of OpenCV's ``cv2.resize`` (NumPy for AREA/LINEAR, Pillow otherwise)."""
    if src is None:
        raise ValueError("resize: src is None")
    if src.ndim == 3 and src.shape[2] == 1:
        src = src[..., 0]  # OpenCV returns a 2-D array for single-channel input

    if dsize is None or dsize == (0, 0):
        if fx <= 0 or fy <= 0:
            raise ValueError("resize: either dsize or fx/fy must be provided")
        dsize = (int(round(src.shape[1] * fx)), int(round(src.shape[0] * fy)))

    new_w, new_h = int(dsize[0]), int(dsize[1])

    if interpolation in (INTER_AREA, INTER_LINEAR):
        # NumPy implementations that reproduce OpenCV's sampling (see helpers).
        if (new_h, new_w) == src.shape[:2]:
            return src.copy()
        if interpolation == INTER_LINEAR:
            fn = _linear_resample_rows
        elif new_h <= src.shape[0] and new_w <= src.shape[1]:
            fn = _area_resample_rows
        else:
            fn = _area_2tap_resample_rows
        out = _resample_2d(src, new_h, new_w, fn)
        return _cast_like(out, src.dtype)

    pil_interp = _INTER_TO_PIL.get(interpolation, Image.Resampling.BICUBIC)
    dsize = (new_w, new_h)

    if src.ndim == 2:
        return _resize_single_channel(src, dsize, pil_interp)

    if src.ndim == 3 and src.shape[2] == 3 and src.dtype == np.uint8:
        pil = Image.fromarray(src, mode="RGB")
        return np.asarray(pil.resize(dsize, resample=pil_interp))

    # Everything else (single channel with trailing dim, RGBA, non-uint8) is
    # resized one channel at a time. In particular RGBA must NOT go through
    # Pillow's "RGBA" mode: Pillow premultiplies by alpha while resampling,
    # which alters the colour channels wherever alpha < 255, whereas OpenCV
    # treats the four channels independently.
    channels = [_resize_single_channel(src[..., c], dsize, pil_interp) for c in range(src.shape[2])]
    return np.stack(channels, axis=-1)


def imshow(winname: str, mat: np.ndarray) -> None:
    """Display ``mat`` via Pillow's default image viewer.

    OpenCV's ``imshow`` takes BGR(A) input, so the channel order is flipped
    before handing the array to Pillow. Callers reaching this stub have
    explicitly opted in to image display (e.g. ``--debug_dataset`` or the
    interactive viewer in ``gen_img.py``), so surfacing the image is the
    expected behavior; the accompanying ``waitKey`` call paces the display.
    """
    if mat is None:
        return

    if mat.ndim == 2:
        pil = Image.fromarray(mat)
    elif mat.ndim == 3 and mat.shape[2] == 1:
        pil = Image.fromarray(mat[..., 0])
    elif mat.ndim == 3 and mat.shape[2] == 3:
        pil = Image.fromarray(_swap_rgb_bgr_3ch(mat))
    elif mat.ndim == 3 and mat.shape[2] == 4:
        pil = Image.fromarray(_swap_rgb_bgr_4ch(mat), mode="RGBA")
    else:
        logger.warning("cv2 stub: imshow got unsupported shape %s; skipping.", mat.shape)
        return

    try:
        pil.show(title=winname)
    except Exception as e:  # noqa: BLE001
        logger.error("cv2 stub: imshow failed to display '%s': %s", winname, e)


def waitKey(delay: int = 0) -> int:
    """Block on ``input()`` in place of OpenCV's key-wait.

    The ``delay`` argument is ignored: in this codebase ``waitKey`` is only
    used from interactive / debug code paths that want to pause between
    images. Returns ``ord(s[0])`` of the first typed character so that
    existing checks such as ``k == ord("e")`` / ``k == ord("s")`` keep
    working; an empty Enter returns ``13`` (carriage return). ``q`` is
    remapped to ``27`` (ESC) since ESC cannot be entered through
    ``input()``, preserving ``k == 27`` branches in callers.
    """
    try:
        s = input("cv2 stub: press Enter to continue (or type 'q' to quit / 'e' / 's'): ")
    except EOFError:
        return -1
    if not s:
        return 13
    # 'q' stands in for the ESC key (keycode 27) since ESC cannot be entered
    # through input(); callers that branch on ``k == 27`` (e.g. the outer
    # epoch loop in debug_dataset) keep working.
    if s[0] == "q":
        return 27
    return ord(s[0])


def destroyAllWindows() -> None:  # pragma: no cover - no-op
    return None


def imwrite(filename: str, img: np.ndarray, params=None) -> bool:
    """Minimal ``cv2.imwrite`` replacement. Assumes BGR / BGRA input."""
    if img is None:
        return False

    if img.ndim == 2:
        pil = Image.fromarray(img)
    elif img.ndim == 3 and img.shape[2] == 3:
        pil = Image.fromarray(_swap_rgb_bgr_3ch(img))
    elif img.ndim == 3 and img.shape[2] == 4:
        pil = Image.fromarray(_swap_rgb_bgr_4ch(img), mode="RGBA")
    else:
        raise ValueError(f"imwrite: unsupported image shape {img.shape}")

    try:
        pil.save(filename)
        return True
    except Exception as e:  # noqa: BLE001
        logger.error("cv2 stub: imwrite failed for %s: %s", filename, e)
        return False


# Sentinel that :mod:`library.cv2_compat` reads to decide whether the caller
# is running against the real OpenCV or this stub.
_IS_CV2_STUB = True
