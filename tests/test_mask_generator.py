"""
Unit tests for library/mask_generator.py.

These cover the procedural mask functions that drive --train_inpainting:
shape/size/mode invariants, seed determinism, binary-value invariant, coverage
bounds, and combine_masks logic. visualize_masks.py covers the qualitative side
(eyeballing variety); these tests cover the contractual side.
"""

import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from library.mask_generator import (
    cloud_mask,
    polygon_mask,
    shape_mask,
    wobbly_ellipse_mask,
    combine_masks,
    random_mask,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coverage(mask: Image.Image) -> float:
    arr = np.array(mask, dtype=np.uint8)
    return float((arr >= 128).sum()) / arr.size


def _is_binary(mask: Image.Image) -> bool:
    arr = np.array(mask, dtype=np.uint8)
    unique = set(np.unique(arr).tolist())
    return unique.issubset({0, 255})


# ---------------------------------------------------------------------------
# Shape / mode / size invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("size", [(64, 64), (128, 96), (200, 256)])
@pytest.mark.parametrize(
    "fn",
    [
        lambda w, h, s: cloud_mask(w, h, seed=s),
        lambda w, h, s: polygon_mask(w, h, seed=s),
        lambda w, h, s: shape_mask(w, h, seed=s),
        lambda w, h, s: wobbly_ellipse_mask(w, h, seed=s),
        lambda w, h, s: random_mask(w, h, seed=s),
    ],
    ids=["cloud", "polygon", "shape", "wobbly_ellipse", "random"],
)
def test_mask_shape_and_mode(size, fn):
    w, h = size
    m = fn(w, h, 0)
    assert isinstance(m, Image.Image)
    assert m.mode == "L"
    assert m.size == (w, h)


@pytest.mark.parametrize(
    "fn",
    [
        lambda s: cloud_mask(64, 64, seed=s),
        lambda s: polygon_mask(64, 64, seed=s),
        lambda s: shape_mask(64, 64, seed=s),
        lambda s: wobbly_ellipse_mask(64, 64, seed=s),
        lambda s: random_mask(64, 64, seed=s),
    ],
    ids=["cloud", "polygon", "shape", "wobbly_ellipse", "random"],
)
def test_mask_is_binary(fn):
    # Each generator must produce strictly {0, 255} values.
    assert _is_binary(fn(42))


# ---------------------------------------------------------------------------
# Seed determinism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fn",
    [
        lambda s: cloud_mask(96, 96, seed=s),
        lambda s: polygon_mask(96, 96, seed=s),
        lambda s: shape_mask(96, 96, seed=s),
        lambda s: wobbly_ellipse_mask(96, 96, seed=s),
        lambda s: random_mask(96, 96, seed=s),
    ],
    ids=["cloud", "polygon", "shape", "wobbly_ellipse", "random"],
)
def test_seed_reproducible(fn):
    a = np.array(fn(123))
    b = np.array(fn(123))
    assert np.array_equal(a, b)


@pytest.mark.parametrize(
    "fn",
    [
        lambda s: cloud_mask(96, 96, seed=s),
        lambda s: polygon_mask(96, 96, seed=s),
        lambda s: shape_mask(96, 96, seed=s),
        lambda s: wobbly_ellipse_mask(96, 96, seed=s),
        lambda s: random_mask(96, 96, seed=s),
    ],
    ids=["cloud", "polygon", "shape", "wobbly_ellipse", "random"],
)
def test_seed_diversity(fn):
    # Different seeds should very likely give different masks. We sample many
    # seeds so a stuck generator can't pass by coincidence.
    samples = {np.array(fn(s)).tobytes() for s in range(8)}
    assert len(samples) > 1


# ---------------------------------------------------------------------------
# Coverage bounds
# ---------------------------------------------------------------------------

def test_cloud_threshold_controls_coverage():
    # threshold is approximate but should track monotonically.
    low = _coverage(cloud_mask(128, 128, threshold=0.1, seed=0))
    high = _coverage(cloud_mask(128, 128, threshold=0.3, seed=0))
    assert low < high
    # threshold ~ target fraction of masked area; allow generous slack.
    assert 0.05 <= low <= 0.20
    assert 0.20 <= high <= 0.45


def test_polygon_coverage_within_bounds():
    # Single polygon, fixed seed; coverage should fall roughly within bounds.
    # The bounds describe each polygon's area target; the rasterised result
    # can over-/under-shoot a little, so we use loose envelopes.
    samples = [
        _coverage(polygon_mask(128, 128, n_points=6, irregularity=0.0,
                               min_coverage=0.15, max_coverage=0.35,
                               n_polygons=1, seed=s))
        for s in range(16)
    ]
    mean = float(np.mean(samples))
    assert 0.05 < mean < 0.5


def test_wobbly_ellipse_coverage_tracks_target():
    # coverage parameter should approximately control the masked fraction.
    samples_low = [_coverage(wobbly_ellipse_mask(128, 128, coverage=0.15,
                                                 wobble_scale=0.1, seed=s))
                   for s in range(8)]
    samples_high = [_coverage(wobbly_ellipse_mask(128, 128, coverage=0.4,
                                                  wobble_scale=0.1, seed=s))
                    for s in range(8)]
    assert np.mean(samples_low) < np.mean(samples_high)


# ---------------------------------------------------------------------------
# combine_masks
# ---------------------------------------------------------------------------

def test_combine_masks_union_is_or():
    a = Image.fromarray(np.array([[0, 255], [0, 0]], dtype=np.uint8), mode="L")
    b = Image.fromarray(np.array([[0, 0],   [255, 0]], dtype=np.uint8), mode="L")
    out = np.array(combine_masks(a, b, mode="union"))
    assert np.array_equal(out, np.array([[0, 255], [255, 0]], dtype=np.uint8))


def test_combine_masks_intersection_is_and():
    a = Image.fromarray(np.array([[255, 255], [0, 0]], dtype=np.uint8), mode="L")
    b = Image.fromarray(np.array([[255, 0],   [0, 0]], dtype=np.uint8), mode="L")
    out = np.array(combine_masks(a, b, mode="intersection"))
    assert np.array_equal(out, np.array([[255, 0], [0, 0]], dtype=np.uint8))


def test_combine_masks_three_inputs_union():
    a = Image.fromarray(np.array([[255, 0]], dtype=np.uint8), mode="L")
    b = Image.fromarray(np.array([[0, 0]],   dtype=np.uint8), mode="L")
    c = Image.fromarray(np.array([[0, 255]], dtype=np.uint8), mode="L")
    out = np.array(combine_masks(a, b, c, mode="union"))
    assert np.array_equal(out, np.array([[255, 255]], dtype=np.uint8))


# ---------------------------------------------------------------------------
# random_mask sanity
# ---------------------------------------------------------------------------

def test_random_mask_nonempty():
    # Across a handful of seeds, at least one produced mask must have non-zero
    # coverage (a generator that always returns all-zero would silently break
    # training without a crash).
    coverages = [_coverage(random_mask(128, 128, seed=s)) for s in range(8)]
    assert max(coverages) > 0.01
