"""RGB-only FLAIR access for the SAM 1 / SAM 2 / SAM 3 comparison.

Deliberately small: no NDVI, no Shannon, no composites. A patch is an (H, W, 3) uint8 RGB
image and an (H, W) mask with class IDs 1..19.

THE BAND ORDER IS NOT ASSUMED. The first three channels of a FLAIR IMG patch are either
(B, G, R) or (R, G, B) depending on which source you trust -- the IGN Hugging Face dataset card
says Red, Green, Blue, NIR, nDSM; this repo's earlier loader assumed Blue, Green, Red. A wrong
choice feeds every pretrained model a blue/red-swapped image, so `band_order` is a required
argument with no default. Decide it with `scripts/check_bands.py` on the real files.

All paths handed around here are RELATIVE to the dataset root, so the same episodes file
works on the cluster and on a laptop.
"""

import glob
import json
import os
import re
import sqlite3
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from datasets.flair import _read_tif, domain_of, mask_path_for

BAND_ORDERS = ("rgb", "bgr")

# FLAIR-HUB labels (AERIAL_LABEL-COSIA) are numbered 0..18 in a DIFFERENT ORDER from FLAIR #1's
# 1..19. Source: the class table in github.com/IGNF/FLAIR-HUB (README, "flair_LC"). Everything in
# this repo speaks FLAIR #1 ids (datasets/flair.py FLAIR_CLASSES), so HUB labels are translated on
# read. Without this, e.g. deciduous (HUB 12) would be read as plowed land (#1 12).
HUB_TO_FLAIR1 = {
    0: 1,    # building
    1: 18,   # greenhouse
    2: 13,   # swimming pool
    3: 3,    # impervious surface
    4: 2,    # pervious surface
    5: 4,    # bare soil
    6: 5,    # water
    7: 14,   # snow
    8: 10,   # herbaceous vegetation
    9: 11,   # agricultural land
    10: 12,  # plowed land
    11: 9,   # vineyard
    12: 7,   # deciduous
    13: 6,   # coniferous
    14: 8,   # brushwood
    15: 15,  # clear cut
    16: 17,  # ligneous
    17: 16,  # mixed
    18: 19,  # undefined -> other
}
_HUB_LUT = np.array([HUB_TO_FLAIR1[i] for i in range(19)], dtype=np.int64)


def read_label(msk_path: str) -> np.ndarray:
    """(H, W) int64 class map in FLAIR #1 ids, whichever FLAIR release the file comes from."""
    raw = _read_tif(msk_path)[0].astype(np.int64)
    if "AERIAL_LABEL-COSIA" in os.path.basename(msk_path):
        if raw.min() < 0 or raw.max() > 18:
            raise ValueError(f"{msk_path}: FLAIR-HUB label values must be 0..18, got {raw.min()}..{raw.max()}")
        return _HUB_LUT[raw]
    return raw


def optical_to_rgb_uint8(raw: np.ndarray, band_order: str) -> np.ndarray:
    """(C>=3, H, W) raw patch -> (H, W, 3) uint8 RGB, honouring the declared band order."""
    if band_order not in BAND_ORDERS:
        raise ValueError(f"band_order must be one of {BAND_ORDERS}, got {band_order!r}")
    if raw.shape[0] < 3:
        raise ValueError(f"expected at least 3 channels, got {raw.shape[0]}")
    first3 = np.asarray(raw[:3], dtype=np.float32)
    if band_order == "bgr":
        first3 = first3[::-1]
    if first3.max() <= 1.0:          # already normalised to [0, 1]
        first3 = first3 * 255.0
    rgb = np.clip(first3, 0, 255).round().astype(np.uint8)
    return np.ascontiguousarray(np.transpose(rgb, (1, 2, 0)))


def discover_patches(root: str) -> List[str]:
    """Root-relative paths of every IMG_*.tif that has a matching mask, sorted."""
    imgs = sorted(glob.glob(os.path.join(root, "**", "IMG_*.tif"), recursive=True))
    imgs += sorted(glob.glob(os.path.join(root, "**", "*_AERIAL_RGBI_*.tif"), recursive=True))   # FLAIR-HUB
    out = []
    for img in imgs:
        if mask_path_for(img) is not None:
            out.append(os.path.relpath(img, root))
    if not out:
        raise FileNotFoundError(f"No FLAIR patches with labels under {root!r} (looked for IMG_*.tif + MSK_*.tif "
                                f"and FLAIR-HUB *_AERIAL_RGBI_*.tif + *_AERIAL_LABEL-COSIA_*.tif)")
    return out


def load_patch(root: str, rel_img: str, band_order: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (rgb uint8 (H, W, 3), mask int64 (H, W))."""
    img_path = os.path.join(root, rel_img)
    msk_path = mask_path_for(img_path)
    if msk_path is None:
        raise FileNotFoundError(f"no mask for {img_path}")
    rgb = optical_to_rgb_uint8(_read_tif(img_path), band_order)
    return rgb, read_label(msk_path)


def build_class_counts(root: str, rel_paths: Sequence[str],
                       cache: Optional[str] = None, workers: int = 1) -> Dict[str, Dict[int, int]]:
    """{rel_path: {class_id: pixel_count}} from one pass over the masks (cached to JSON)."""
    if cache and os.path.exists(cache):
        with open(cache) as fh:
            raw = json.load(fh)
        if set(raw) >= set(rel_paths):
            return {p: {int(c): n for c, n in raw[p].items()} for p in rel_paths}

    def one(rel):
        mask = read_label(mask_path_for(os.path.join(root, rel)))
        ids, n = np.unique(mask, return_counts=True)
        return {int(c): int(k) for c, k in zip(ids.tolist(), n.tolist())}

    counts: Dict[str, Dict[int, int]] = {}
    if workers > 1:                                    # reading labels is I/O bound: threads are enough
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(workers) as pool:
            for i, (rel, c) in enumerate(zip(rel_paths, pool.map(one, rel_paths))):
                counts[rel] = c
                if (i + 1) % 2000 == 0:
                    print(f"  indexed {i + 1}/{len(rel_paths)} masks", flush=True)
    else:
        for i, rel in enumerate(rel_paths):
            counts[rel] = one(rel)
            if (i + 1) % 1000 == 0:
                print(f"  indexed {i + 1}/{len(rel_paths)} masks", flush=True)
    if cache:
        os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
        with open(cache, "w") as fh:
            json.dump(counts, fh)
    return counts


def domains_of(rel_paths: Sequence[str]) -> Dict[str, str]:
    return {p: domain_of(p) for p in rel_paths}


# ------------------------------------------------------------------------ acquisition dates (FLAIR-HUB)
_HUB_NAME = re.compile(r"(D\d{3,6}-\d{4})_AERIAL_RGBI_(.+)_(\d+-\d+)\.tif$")


def hub_patch_id(rel: str) -> Optional[str]:
    """FLAIR-HUB patch id ('D069-2020_<roi>_<x-y>') of an AERIAL_RGBI patch path; None for other layouts."""
    m = _HUB_NAME.search(os.path.basename(rel))
    return f"{m.group(1)}_{m.group(2)}_{m.group(3)}" if m else None


def read_aerial_dates(gpkg: str) -> Dict[str, str]:
    """{patch_id: 'YYYYMMDD'} from a FLAIR-HUB *AERIAL_MTD_DATES.gpkg (per-domain or GLOBAL). Read-only SQLite."""
    con = sqlite3.connect(f"file:{os.path.abspath(gpkg)}?mode=ro", uri=True)
    try:
        tables = [r[0] for r in con.execute("select table_name from gpkg_contents where data_type='features'")]
        out: Dict[str, str] = {}
        for tb in tables:
            cols = {r[1] for r in con.execute(f'pragma table_info("{tb}")')}
            if {"patch_id", "date"} <= cols:
                out.update({pid: str(d) for pid, d in con.execute(f'select patch_id, date from "{tb}"')})
        return out
    finally:
        con.close()
