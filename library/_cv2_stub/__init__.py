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
- ``resize`` is implemented via Pillow, mapping ``INTER_*`` constants to
  their closest ``PIL.Image.Resampling`` equivalents.
- ``imshow`` / ``waitKey`` / ``destroyAllWindows`` are no-ops (debug-only).
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

_INTER_TO_PIL = {
    INTER_NEAREST: Image.Resampling.NEAREST,
    INTER_NEAREST_EXACT: Image.Resampling.NEAREST,
    INTER_LINEAR: Image.Resampling.BILINEAR,
    INTER_CUBIC: Image.Resampling.BICUBIC,
    # INTER_AREA has no exact PIL analogue; HAMMING is a reasonable
    # approximation for downsampling and BOX-like averaging.
    INTER_AREA: Image.Resampling.HAMMING,
    INTER_LANCZOS4: Image.Resampling.LANCZOS,
}


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
    # Map [0,360) -> [0,180) to match OpenCV uint8 convention.
    h = h / 2.0

    out = np.stack(
        [
            np.clip(h + 0.5, 0, 179).astype(np.uint8),
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
    """Subset of OpenCV's ``cv2.resize`` implemented via Pillow."""
    if src is None:
        raise ValueError("resize: src is None")

    if dsize is None or dsize == (0, 0):
        if fx <= 0 or fy <= 0:
            raise ValueError("resize: either dsize or fx/fy must be provided")
        dsize = (int(round(src.shape[1] * fx)), int(round(src.shape[0] * fy)))

    pil_interp = _INTER_TO_PIL.get(interpolation, Image.Resampling.BILINEAR)

    # Pillow cannot handle every dtype/shape cv2 does; fall back to a
    # per-channel resize for the cases we care about (uint8 HxW, HxWx{1,3,4}).
    if src.ndim == 2:
        pil = Image.fromarray(src)
        resized = pil.resize(dsize, resample=pil_interp)
        return np.asarray(resized)

    if src.ndim == 3 and src.shape[2] in (1, 3, 4) and src.dtype == np.uint8:
        if src.shape[2] == 1:
            pil = Image.fromarray(src[..., 0])
            resized = pil.resize(dsize, resample=pil_interp)
            return np.asarray(resized)[..., None]
        mode = "RGB" if src.shape[2] == 3 else "RGBA"
        pil = Image.fromarray(src, mode=mode)
        resized = pil.resize(dsize, resample=pil_interp)
        return np.asarray(resized)

    # Generic fallback: resize each channel independently.
    channels = [
        np.asarray(Image.fromarray(src[..., c]).resize(dsize, resample=pil_interp))
        for c in range(src.shape[2])
    ]
    return np.stack(channels, axis=-1)


def imshow(winname: str, mat: np.ndarray) -> None:  # pragma: no cover - no-op
    if not getattr(imshow, "_warned", False):
        logger.warning(
            "cv2 stub: imshow is not supported without opencv-python; window '%s' will not be shown.",
            winname,
        )
        imshow._warned = True  # type: ignore[attr-defined]


def waitKey(delay: int = 0) -> int:  # pragma: no cover - no-op
    return -1


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
