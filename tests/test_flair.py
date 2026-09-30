"""Tests for the FLAIR loader.

TIFF I/O is stubbed out so these run anywhere, including on a machine with neither
rasterio nor tifffile installed. What is under test is the part that can silently produce
wrong science -- band ordering, NDVI arithmetic, composite assembly, elevation scaling and
domain parsing -- not whether a third-party library can open a file.

    python tests/test_flair.py
"""

import os
import sys
import shutil
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import datasets.flair as flair_mod  # noqa: E402
from datasets.flair import (  # noqa: E402
    FLAIRDataset, FLAIR_CLASSES, FOLDS, BASELINE_13, VEGETATION_CLASSES,
    split_by_domain, domain_of, BLUE, GREEN, RED, NIR, ELEV,
)

SIZE = 64  # smaller than a real 512 patch; nothing under test depends on the size


def build_fake_tree(root, domains=("D004_2021", "D007_2020")):
    """Create the directory shape and empty stub files the loader globs for."""
    made = []
    for dom in domains:
        for roi in ("Z1_UA", "Z2_UB"):
            img_dir = os.path.join(root, dom, roi, "img")
            msk_dir = os.path.join(root, dom, roi, "msk")
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(msk_dir, exist_ok=True)
            for n in range(3):
                img = os.path.join(img_dir, f"IMG_{len(made):06d}.tif")
                msk = os.path.join(msk_dir, f"MSK_{len(made):06d}.tif")
                open(img, "wb").close()
                open(msk, "wb").close()
                made.append(img)
    return made


def install_fake_reader(elevation_metres=True):
    """Replace _read_tif with a deterministic synthetic FLAIR patch generator.

    Returns the known-correct band values so tests can assert exact arithmetic.
    """
    blue, green, red, nir = 40.0, 90.0, 60.0, 200.0   # 8-bit DNs; NIR brightest, as vegetation
    elev = 12.0 if elevation_metres else 0.4

    def fake_read(path):
        if "MSK" in os.path.basename(path):
            mask = np.full((1, SIZE, SIZE), 7, dtype=np.uint8)   # deciduous
            mask[:, : SIZE // 2, :] = 1                          # building on the top half
            return mask
        patch = np.zeros((5, SIZE, SIZE), dtype=np.float32)
        patch[BLUE], patch[GREEN], patch[RED], patch[NIR] = blue, green, red, nir
        patch[ELEV] = elev
        return patch

    flair_mod._read_tif = fake_read
    return {"blue": blue, "green": green, "red": red, "nir": nir, "elev": elev}


def _reader_with_elevation(elev_m):
    """A fake reader identical to install_fake_reader's except for canopy height."""
    blue, green, red, nir = 40.0, 90.0, 60.0, 200.0

    def fake_read(path):
        if "MSK" in os.path.basename(path):
            mask = np.full((1, SIZE, SIZE), 7, dtype=np.uint8)
            mask[:, : SIZE // 2, :] = 1
            return mask
        patch = np.zeros((5, SIZE, SIZE), dtype=np.float32)
        patch[BLUE], patch[GREEN], patch[RED], patch[NIR] = blue, green, red, nir
        patch[ELEV] = elev_m
        return patch

    return fake_read


def test_discovery_and_pairing():
    print("[1/7] Discovery, IMG/MSK pairing and domain parsing...")
    root = tempfile.mkdtemp()
    try:
        made = build_fake_tree(root)
        install_fake_reader()
        ds = FLAIRDataset(root=root)
        assert len(ds) == len(made), f"expected {len(made)} patches, found {len(ds)}"
        assert set(ds.domains) == {"D004_2021", "D007_2020"}, f"domains were {set(ds.domains)}"
        assert domain_of("/x/D123_2019/Z1_UA/img/IMG_000001.tif") == "D123_2019"

        # Filtering by domain must actually filter.
        subset = FLAIRDataset(root=root, domains=["D004_2021"])
        assert len(subset) == len(made) // 2
        assert set(subset.domains) == {"D004_2021"}
        print("      --> Passed!")
    finally:
        shutil.rmtree(root)


def test_ndvi_arithmetic():
    print("[2/7] NDVI is computed from the right two bands...")
    root = tempfile.mkdtemp()
    try:
        build_fake_tree(root)
        bands = install_fake_reader()
        ds = FLAIRDataset(root=root)
        ndvi = ds[0]["ndvi"]

        nir01, red01 = bands["nir"] / 255.0, bands["red"] / 255.0
        expected = (nir01 - red01) / (nir01 + red01)
        assert torch.allclose(ndvi, torch.full_like(ndvi, expected), atol=1e-4), \
            f"NDVI {ndvi.mean():.4f} != expected {expected:.4f} -- wrong bands?"

        # Guard the band order itself: swapping NIR and Red must flip the sign.
        flipped = (red01 - nir01) / (nir01 + red01)
        assert not np.isclose(expected, flipped), "degenerate test fixture"
        assert expected > 0, "NIR should exceed Red on a vegetated fixture"
        print(f"      --> Passed! (NDVI = {expected:.4f})")
    finally:
        shutil.rmtree(root)


def test_composites():
    print("[3/7] Every composite is (3, H, W) float32 in [0, 1]...")
    root = tempfile.mkdtemp()
    try:
        build_fake_tree(root)
        bands = install_fake_reader()
        seen = {}
        for name in FLAIRDataset.COMPOSITES:
            ds = FLAIRDataset(root=root, composite=name)
            img = ds[0]["image"]
            assert img.shape == (3, SIZE, SIZE), f"{name}: shape {tuple(img.shape)}"
            assert img.dtype == torch.float32, f"{name}: dtype {img.dtype}"
            assert img.min() >= 0.0 and img.max() <= 1.0, \
                f"{name}: range [{img.min():.3f}, {img.max():.3f}] escapes [0, 1]"
            seen[name] = tuple(round(float(img[c].mean()), 4) for c in range(3))

        # rgb and cir must differ, or the composite switch is doing nothing.
        assert seen["rgb"] != seen["cir"], "rgb and cir composites are identical"

        # cir is (NIR, Red, Green): its first channel must be the brightest band.
        assert seen["cir"][0] > seen["cir"][1], "cir channel 0 is not NIR"
        assert np.isclose(seen["cir"][0], bands["nir"] / 255.0, atol=1e-3)
        assert np.isclose(seen["rgb"][0], bands["red"] / 255.0, atol=1e-3)
        print("      --> Passed!")
    finally:
        shutil.rmtree(root)


def test_elevation_free_composites_ignore_the_dsm():
    """The EU UAV target carries RGB + NIR and no elevation.

    Anything on the deliverable path must therefore be computable without the nDSM. This
    asserts it mechanically rather than by inspection: change the elevation channel wildly
    and every ELEVATION_FREE composite must be bit-identical, while every
    ELEVATION_DEPENDENT one must change. A composite that quietly picked up the DSM would
    produce a headline number that cannot transfer.
    """
    print("[4/7] Elevation-free composites are unaffected by the nDSM...")
    root = tempfile.mkdtemp()
    try:
        build_fake_tree(root)

        install_fake_reader(elevation_metres=True)          # nDSM = 12 m
        low = {name: FLAIRDataset(root=root, composite=name)[0]["image"].clone()
               for name in FLAIRDataset.COMPOSITES}

        # Same scene, radically different canopy height.
        flair_mod._read_tif = _reader_with_elevation(28.0)
        high = {name: FLAIRDataset(root=root, composite=name)[0]["image"]
                for name in FLAIRDataset.COMPOSITES}

        for name in FLAIRDataset.ELEVATION_FREE:
            assert torch.allclose(low[name], high[name]), \
                f"composite {name!r} is listed as elevation-free but changed with the nDSM"
        for name in FLAIRDataset.ELEVATION_DEPENDENT:
            assert not torch.allclose(low[name], high[name]), \
                f"composite {name!r} is listed as elevation-dependent but ignored the nDSM"

        # Every composite must be accounted for in exactly one of the two lists.
        assert set(FLAIRDataset.ELEVATION_FREE) | set(FLAIRDataset.ELEVATION_DEPENDENT) \
            == set(FLAIRDataset.COMPOSITES)
        assert not set(FLAIRDataset.ELEVATION_FREE) & set(FLAIRDataset.ELEVATION_DEPENDENT)

        # Shannon over NDVI must likewise be independent of height.
        install_fake_reader(elevation_metres=True)
        h_low = FLAIRDataset(root=root, shannon_source="ndvi")[0]["shannon"].clone()
        flair_mod._read_tif = _reader_with_elevation(28.0)
        h_high = FLAIRDataset(root=root, shannon_source="ndvi")[0]["shannon"]
        assert torch.allclose(h_low, h_high), "shannon_source='ndvi' leaked elevation"
        print("      --> Passed!")
    finally:
        shutil.rmtree(root)


def test_elevation_scaling():
    print("[5/7] Elevation is clipped and normalised, not treated as a DN...")
    root = tempfile.mkdtemp()
    try:
        build_fake_tree(root)
        bands = install_fake_reader(elevation_metres=True)
        ds = FLAIRDataset(root=root, elevation_clip_m=30.0)
        elev = ds[0]["elevation"]
        expected = bands["elev"] / 30.0
        assert torch.allclose(elev, torch.full_like(elev, expected), atol=1e-5), \
            f"elevation {elev.mean():.4f} != {expected:.4f}"

        # A shorter clip must raise the normalised value -- proving the clip is live.
        ds_short = FLAIRDataset(root=root, elevation_clip_m=12.0)
        assert ds_short[0]["elevation"].mean() > elev.mean()

        # Crucially, elevation must NOT have been divided by 255 with the optical bands.
        assert elev.mean() > 0.1, "elevation looks like it was scaled as an 8-bit DN"
        print("      --> Passed!")
    finally:
        shutil.rmtree(root)


def test_shannon_source():
    print("[6/7] Shannon can be computed over NDVI or over height...")
    root = tempfile.mkdtemp()
    try:
        build_fake_tree(root)
        install_fake_reader()
        for source in ("ndvi", "elevation"):
            ds = FLAIRDataset(root=root, shannon_source=source)
            h = ds[0]["shannon"]
            assert h.shape == (1, SIZE, SIZE)
            assert h.min() >= 0.0 and h.max() <= 1.0, f"{source}: H outside [0, 1]"
        # A constant fixture has zero local variation, so entropy must be ~0.
        assert FLAIRDataset(root=root)[0]["shannon"].max() < 1e-3, \
            "constant input produced non-zero entropy"

        try:
            FLAIRDataset(root=root, shannon_source="nonsense")
            raise AssertionError("bad shannon_source should have raised")
        except ValueError:
            pass
        print("      --> Passed!")
    finally:
        shutil.rmtree(root)


def test_splits_and_nomenclature():
    print("[7/7] Domain splits and the class nomenclature...")
    root = tempfile.mkdtemp()
    try:
        build_fake_tree(root)
        install_fake_reader()
        ds = FLAIRDataset(root=root)
        train, held = split_by_domain(ds, holdout=["D007_2020"])
        assert len(train) + len(held) == len(ds)
        assert set(train) & set(held) == set(), "domain split overlaps"
        assert all(ds.domains[i] == "D007_2020" for i in held)
        assert all(ds.domains[i] != "D007_2020" for i in train)

        assert len(FLAIR_CLASSES) == 19, f"expected 19 classes, got {len(FLAIR_CLASSES)}"
        assert min(FLAIR_CLASSES) == 1 and max(FLAIR_CLASSES) == 19, "class IDs are 1-based"
        assert 0 not in FLAIR_CLASSES, "FLAIR has no class 0"
        assert all(c in FLAIR_CLASSES for c in BASELINE_13)
        assert all(c in FLAIR_CLASSES for c in VEGETATION_CLASSES)
        for fold, classes in FOLDS.items():
            assert all(c in FLAIR_CLASSES for c in classes), f"fold {fold} has an unknown class"

        mask = ds[0]["mask"]
        assert mask.dtype == torch.int64 and mask.min() >= 1
        print("      --> Passed!")
    finally:
        shutil.rmtree(root)


if __name__ == "__main__":
    print("\n--- FLAIR loader tests (TIFF I/O stubbed) ---")
    test_discovery_and_pairing()
    test_ndvi_arithmetic()
    test_composites()
    test_elevation_free_composites_ignore_the_dsm()
    test_elevation_scaling()
    test_shannon_source()
    test_splits_and_nomenclature()
    print("\nAll FLAIR loader tests passed.\n")
