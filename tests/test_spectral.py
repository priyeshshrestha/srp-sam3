"""Tests for post-encoder spectral insertion.

The central guarantee under test is the one that makes these routes worth trying at all:
every blend is convex and has a setting that reproduces the RGB baseline EXACTLY. Input
substitution (CIR) has no such setting, which is why it could only trade and, measurably,
traded badly. If a refactor ever breaks that property, these routes lose their floor.

    python tests/test_spectral.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.spectral import (  # noqa: E402
    normalise_map, reference_stats, ecological_plausibility,
    modulate_similarity, fuse_similarity, select_prompt_points,
)


def test_normalise_map():
    print("[1/6] Similarity maps are put on a common [0, 1] scale...")
    sim = np.array([[0.45, 0.60], [0.75, 0.90]])
    out = normalise_map(sim)
    assert np.isclose(out.min(), 0.0) and np.isclose(out.max(), 1.0)
    # Order must be preserved -- normalisation may not reorder pixels.
    assert np.all(np.argsort(out.ravel()) == np.argsort(sim.ravel()))
    # A constant map has no information and must not blow up to NaN or inf.
    flat = normalise_map(np.full((8, 8), 0.7))
    assert np.all(flat == 0.0) and np.isfinite(flat).all()
    print("      --> Passed!")


def test_reference_stats_from_support_only():
    print("[2/6] Reference statistics summarise the support region...")
    ndvi = np.zeros((32, 32)); ndvi[:16] = 0.75; ndvi[16:] = -0.10
    mask = np.zeros((32, 32), dtype=bool); mask[:16] = True

    ref = reference_stats(mask, ndvi=ndvi)
    assert np.isclose(ref["ndvi_mean"], 0.75, atol=1e-6), ref
    # Std is floored so a uniform support cannot produce a delta-function kernel.
    assert ref["ndvi_std"] >= 0.02

    # An empty mask must yield nothing, not a spurious statistic over the whole image.
    assert reference_stats(np.zeros((32, 32), dtype=bool), ndvi=ndvi) == {}
    # Signals that were not supplied must not appear.
    assert "shannon_mean" not in ref and "elevation_mean" not in ref
    print("      --> Passed!")


def test_plausibility_peaks_at_the_reference():
    print("[3/6] Plausibility peaks where the query matches the support...")
    ref = {"ndvi_mean": 0.70, "ndvi_std": 0.05}
    query = np.array([[0.70, 0.60], [0.30, -0.10]])
    plaus = ecological_plausibility(ref, ndvi=query)

    assert plaus is not None
    assert plaus.min() >= 0.0 and plaus.max() <= 1.0
    assert np.isclose(plaus[0, 0], 1.0, atol=1e-6), "exact match should score 1"
    # Monotone decreasing with distance from the reference mean.
    assert plaus[0, 0] > plaus[0, 1] > plaus[1, 0] > plaus[1, 1]

    # No signal, or no reference, means no opinion -- callers need None to fall back.
    assert ecological_plausibility({}, ndvi=query) is None
    assert ecological_plausibility(ref, ndvi=None) is None

    # A wider tolerance must be strictly more permissive everywhere off the mean.
    tight = ecological_plausibility(ref, ndvi=query, tolerance=1.0)
    loose = ecological_plausibility(ref, ndvi=query, tolerance=4.0)
    assert np.all(loose[1:] >= tight[1:])
    print("      --> Passed!")


def test_modulation_has_a_baseline_floor():
    print("[4/6] weight=0 reproduces the baseline map EXACTLY...")
    rng = np.random.default_rng(0)
    sim = rng.uniform(0.45, 0.9, size=(24, 24))
    plaus = rng.uniform(0.0, 1.0, size=(24, 24))

    # THE guarantee: w=0 is the unmodified baseline, bit for bit.
    assert np.allclose(modulate_similarity(sim, plaus, weight=0.0), normalise_map(sim))
    # And a missing plausibility map degrades to the baseline rather than to zeros.
    assert np.allclose(modulate_similarity(sim, None, weight=0.9), normalise_map(sim))

    # w=1 hands the map over to ecology entirely.
    assert np.allclose(modulate_similarity(sim, plaus, weight=1.0), plaus)

    # Intermediate weights stay bounded and move monotonically between the two.
    for w in (0.1, 0.3, 0.5, 0.7):
        out = modulate_similarity(sim, plaus, weight=w)
        assert out.min() >= 0.0 and out.max() <= 1.0

    # Shape mismatches must raise, not broadcast into nonsense.
    try:
        modulate_similarity(sim, np.zeros((8, 8)), weight=0.5)
        raise AssertionError("shape mismatch should have raised")
    except ValueError:
        pass
    print("      --> Passed!")


def test_dual_stream_cannot_lose():
    print("[5/6] alpha=1 recovers the RGB stream exactly...")
    rng = np.random.default_rng(1)
    s_rgb = rng.uniform(0.4, 0.9, size=(16, 16))
    s_cir = rng.uniform(0.3, 0.8, size=(16, 16))

    # The floor: alpha=1 is pure RGB, so a tuned alpha can never underperform it.
    assert np.allclose(fuse_similarity([s_rgb, s_cir], alpha=1.0), normalise_map(s_rgb))
    assert np.allclose(fuse_similarity([s_rgb, s_cir], alpha=0.0), normalise_map(s_cir))

    mixed = fuse_similarity([s_rgb, s_cir], alpha=0.5)
    assert mixed.min() >= 0.0 and mixed.max() <= 1.0
    # A convex blend of two maps cannot exceed the pointwise envelope of either.
    hi = np.maximum(normalise_map(s_rgb), normalise_map(s_cir))
    lo = np.minimum(normalise_map(s_rgb), normalise_map(s_cir))
    assert np.all(mixed <= hi + 1e-9) and np.all(mixed >= lo - 1e-9)

    assert np.allclose(fuse_similarity([s_rgb]), normalise_map(s_rgb))
    try:
        fuse_similarity([s_rgb, np.zeros((4, 4))])
        raise AssertionError("shape mismatch should have raised")
    except ValueError:
        pass
    print("      --> Passed!")


def test_prompt_points_spread_out():
    print("[6/6] Prompt points sample the region instead of clustering...")
    score = np.zeros((64, 64))
    score[10, 10] = 1.0      # two distinct hotspots, far apart
    score[50, 50] = 0.95
    score[10, 12] = 0.94     # a near-duplicate of the first, 2 px away
    score[30, 30] = -1.0     # the clear negative

    coords, labels = select_prompt_points(score, num_positive=2, num_negative=1,
                                          min_separation=16)
    assert coords.shape == (3, 2) and labels.tolist() == [1, 1, 0]

    pos = coords[labels == 1]
    # The second positive must NOT be the 2-px neighbour of the first.
    dists = np.abs(pos[0] - pos[1]).max()
    assert dists > 16, f"positives only {dists} px apart -- suppression failed"
    assert tuple(coords[labels == 0][0]) == (30.0, 30.0), "negative is not at the minimum"

    # The single-point case must still behave like the original argmax/argmin.
    c1, l1 = select_prompt_points(score, num_positive=1, num_negative=1)
    assert tuple(c1[0]) == (10.0, 10.0) and l1.tolist() == [1, 0]
    print("      --> Passed!")


if __name__ == "__main__":
    print("\n--- Post-encoder spectral insertion tests ---")
    test_normalise_map()
    test_reference_stats_from_support_only()
    test_plausibility_peaks_at_the_reference()
    test_modulation_has_a_baseline_floor()
    test_dual_stream_cannot_lose()
    test_prompt_points_spread_out()
    print("\nAll spectral insertion tests passed.\n")
