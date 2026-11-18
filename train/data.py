"""FLAIR patches as RGB or RGB+NIR tensors, the train/test split, and the training schedule.

Same rules as the few-shot benchmark: paths are relative to the data root, band order is required
(it only decides which of channels 0..2 is red; NIR is channel 3 in both FLAIR releases), and labels
are FLAIR #1 ids 1..19 whichever release the file comes from.

Pairing: the RGB and RGBI runs of one seed see the SAME images in the SAME order with the SAME
augmentations and the SAME prompted classes. Every random choice for training item k comes from
`np.random.default_rng([seed, k])`, so nothing depends on the model, the channel count, the number
of data-loader workers, or on where a resumed run restarted.
"""

import hashlib
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.flair import _read_tif, domain_of, mask_path_for
from datasets.flair_rgb import BAND_ORDERS, read_label
from train.sam3_rgbi import CLASSES

# CHANNELS: bands in the INPUT tensor. PATCH_CHANNELS: channels of the model's first (patch-embedding) layer.
# rgb_ndvi (VegSAM-style) reads all 4 bands but the ViT sees only R,G,B; NIR reaches the model ONLY through
# the NDVI adapter. rgbi_ndvi = raw NIR channel AND the adapter.
CHANNELS = {"rgb": 3, "rgbi": 4, "rgbi_ndvi": 4, "rgb_ndvi": 4}
PATCH_CHANNELS = {"rgb": 3, "rgbi": 4, "rgbi_ndvi": 4, "rgb_ndvi": 3}


def uses_ndvi(channels: str) -> bool:
    return channels.endswith("_ndvi")



MIN_PRESENT_FRAC = 0.005          # a class "is present" for prompt sampling if it covers >= 0.5% of the patch


def load_rgbi(root: str, rel: str, band_order: str) -> Tuple[np.ndarray, np.ndarray]:
    """(4, H, W) uint8 as R, G, B, NIR  and  (H, W) int64 FLAIR #1 labels."""
    if band_order not in BAND_ORDERS:
        raise ValueError(f"band_order must be one of {BAND_ORDERS}, got {band_order!r}")
    path = os.path.join(root, rel)
    raw = _read_tif(path)
    if raw.shape[0] < 4:
        raise ValueError(f"{rel}: {raw.shape[0]} channels; RGB+NIR training needs >= 4 (R, G, B, NIR)")
    x = np.asarray(raw[:4], dtype=np.float32)
    if band_order == "bgr":
        x = x[[2, 1, 0, 3]]
    if x.max() <= 1.0:
        x = x * 255.0
    x = np.clip(x, 0, 255).round().astype(np.uint8)
    return x, read_label(mask_path_for(path))


# ------------------------------------------------------------------------------------ split
def read_hub_split_csvs(csv_dir: str) -> Optional[Dict[str, set]]:
    """{'train': {rel paths}, 'test': {rel paths}} from the official FLAIR-HUB split CSVs
    (FLAIR-HUB_TRAIN.csv / FLAIR-HUB_TEST.csv, ';'-separated, AERIAL_RGBI column = '../<dom>_AERIAL_RGBI/...').
    The VALID split is not used (no model selection is done). None if the CSVs are absent."""
    import csv
    files = {"train": "FLAIR-HUB_TRAIN.csv", "test": "FLAIR-HUB_TEST.csv"}
    if not csv_dir or not all(os.path.exists(os.path.join(csv_dir, f)) for f in files.values()):
        return None
    out = {}
    for key, f in files.items():
        with open(os.path.join(csv_dir, f), newline="") as fh:
            out[key] = {os.path.normpath(row["AERIAL_RGBI"].replace("../", "", 1))
                        for row in csv.DictReader(fh, delimiter=";")}
    return out


def make_split(rel_paths: Sequence[str], mode: str = "auto", test_frac: float = 0.2, seed: int = 0,
               csv_dir: Optional[str] = None) -> Dict:
    """Train/test split. Test regions should be unseen in training.

    hub_csv   the OFFICIAL FLAIR-HUB split (FLAIR-HUB_TRAIN/TEST.csv in `csv_dir`), restricted to the
              patches present on disk
    official  paths with a directory component containing 'test' are the test set (the layout of
              the official FLAIR #1 release: flair_aerial_train/ vs flair_1_aerial_test/)
    domain    a seeded random `test_frac` of the domains is held out (whole domains, never patches)
    auto      hub_csv if the CSVs exist, else official if a 'test' directory exists, else domain
    """
    rel_paths = sorted(rel_paths)
    hub = read_hub_split_csvs(csv_dir) if mode in ("auto", "hub_csv") else None
    is_test = [any("test" in part.lower() for part in os.path.normpath(p).split(os.sep)[:-1]) for p in rel_paths]
    if mode == "auto":
        mode = "hub_csv" if hub else ("official" if any(is_test) and not all(is_test) else "domain")
    if mode == "hub_csv":
        if hub is None:
            raise ValueError(f"SPLIT=hub_csv needs FLAIR-HUB_TRAIN.csv and FLAIR-HUB_TEST.csv in {csv_dir!r}")
        test = [p for p in rel_paths if os.path.normpath(p) in hub["test"]]
        train = [p for p in rel_paths if os.path.normpath(p) in hub["train"]]
        if not test or not train:
            raise ValueError(f"the split CSVs match {len(train)} train / {len(test)} test patches on disk; "
                             f"is FLAIR_DATA_DIR the folder that holds the <domain>_AERIAL_RGBI folders?")
    elif mode == "official":
        test = [p for p, t in zip(rel_paths, is_test) if t]
        train = [p for p, t in zip(rel_paths, is_test) if not t]
        if not test or not train:
            raise ValueError("SPLIT=official needs some (not all) patches under a directory named '*test*'")
    elif mode == "domain":
        doms = sorted({domain_of(p) for p in rel_paths})
        if len(doms) < 2:
            raise ValueError(f"only {len(doms)} domain(s); a held-out-domain split needs >= 2")
        n_test = min(len(doms) - 1, max(1, int(round(test_frac * len(doms)))))
        test_doms = set(np.random.default_rng(seed).permutation(doms)[:n_test].tolist())
        test = [p for p in rel_paths if domain_of(p) in test_doms]
        train = [p for p in rel_paths if domain_of(p) not in test_doms]
    else:
        raise ValueError(f"split mode must be auto | hub_csv | official | domain, got {mode!r}")
    tr_d, te_d = {domain_of(p) for p in train}, {domain_of(p) for p in test}
    split = {"mode": mode, "test_frac": test_frac if mode == "domain" else None, "seed": seed,
             "n_unassigned": len(rel_paths) - len(train) - len(test),
             "train": train, "test": test, "train_domains": sorted(tr_d), "test_domains": sorted(te_d),
             "domain_overlap": sorted(tr_d & te_d)}
    split["sha"] = split_hash(split)
    return split


def split_hash(split: Dict) -> str:
    h = hashlib.sha256()
    for key in ("train", "test"):
        h.update(key.encode())
        for p in split[key]:
            h.update(p.encode())
    return h.hexdigest()[:16]


def load_split(path: str) -> Dict:
    with open(path) as fh:
        split = json.load(fh)
    if split_hash(split) != split.get("sha"):
        raise ValueError(f"{path}: contents do not match its recorded sha (edited by hand?)")
    return split


def test_subset(split: Dict, limit: Optional[int]) -> List[str]:
    """All test patches, or a fixed (seed 0) random `limit` of them: identical for every run."""
    test = list(split["test"])
    if not limit or limit >= len(test):
        return test
    idx = np.sort(np.random.default_rng(0).permutation(len(test))[:limit])
    return [test[i] for i in idx]


# ---------------------------------------------------------------------------------- datasets
def to_model_input(x_uint8: np.ndarray, channels: str) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x_uint8[:CHANNELS[channels]]))


def sample_prompts(label: np.ndarray, n_prompts: int, rng: np.random.Generator,
                   priority_idx=None) -> np.ndarray:
    """Class indices (0..11) to prompt: up to n_prompts-1 present classes, the rest absent ones, so
    the model learns both 'segment it' and 'it is not here'. Falls back gracefully at the extremes.
    With `priority_idx`, present PRIORITY classes are prompted before other present classes."""
    frac = np.array([(label == c).mean() for c in CLASSES])
    present = np.flatnonzero(frac >= MIN_PRESENT_FRAC)
    absent = np.flatnonzero(frac < MIN_PRESENT_FRAC)
    n_pos = min(len(present), max(1, n_prompts - 1)) if len(present) else 0
    if priority_idx is not None and n_pos:
        is_p = np.isin(present, list(priority_idx))
        pos = np.concatenate([rng.permutation(present[is_p]), rng.permutation(present[~is_p])])[:n_pos]
    else:
        pos = rng.choice(present, n_pos, replace=False) if n_pos else np.array([], dtype=int)
    n_neg = n_prompts - n_pos
    pool = absent if len(absent) >= n_neg else np.setdiff1d(np.arange(len(CLASSES)), pos)
    neg = rng.choice(pool, n_neg, replace=False)
    return np.concatenate([pos, neg]).astype(np.int64)


def augment(x: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    """Aerial images have no 'up': random 90-degree rotation and flip."""
    k = int(rng.integers(4))
    x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
    if rng.random() < 0.5:
        x, y = x[:, :, ::-1], y[:, ::-1]
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


# ---------------------------------------------------------------------- focus on priority classes
def focus_weights(counts: Dict[str, Dict[int, int]], rel_paths: Sequence[str], priority: Sequence[int],
                  floor: float) -> Dict[str, float]:
    """Sampling weight of a training patch = floor + (fraction of its pixels in the priority classes).
    So a patch full of priority classes is drawn (1 + floor) / floor times as often as one with none."""
    out = {}
    for rel in rel_paths:
        c = counts[rel]
        total = max(1, sum(c.values()))
        out[rel] = float(floor) + sum(c.get(int(k), 0) for k in priority) / total
    return out


def focus_hash(focus: Dict) -> str:
    h = hashlib.sha256(json.dumps([sorted(focus["priority"]), focus["floor"], focus["split_sha"]]).encode())
    for rel in sorted(focus["weights"]):
        h.update(f"{rel}:{focus['weights'][rel]:.6f}".encode())
    return h.hexdigest()[:16]


def load_focus(path: str, split: Dict) -> Dict:
    """Focus file -> dict with 'priority', 'floor', 'sha' and 'w' (weights aligned with split['train'])."""
    with open(path) as fh:
        focus = json.load(fh)
    if focus_hash(focus) != focus.get("sha"):
        raise ValueError(f"{path}: contents do not match its recorded sha (edited by hand?)")
    if focus["split_sha"] != split["sha"]:
        raise ValueError(f"{path} was made for split {focus['split_sha']}, not {split['sha']}")
    focus["w"] = [focus["weights"][r] for r in split["train"]]
    return focus


class TrainSchedule(Dataset):
    """Item k of the whole training run (k = step * batch + slot), deterministic in (seed, k).

    With `weights` (one per patch, e.g. from focus_weights) patches are drawn WITH replacement in
    proportion to their weight, and `priority_idx` classes are prompted first; otherwise every patch is
    seen once per pass in a random order. Either way the schedule depends only on (seed, weights), never
    on the model, so the arms of a comparison see identical data."""

    def __init__(self, root, rel_paths, band_order, channels, seed, n_items, n_prompts=3, start_item=0,
                 weights=None, priority_idx=None):
        self.root, self.rel, self.band_order, self.channels = root, list(rel_paths), band_order, channels
        self.seed, self.n_prompts, self.start = seed, n_prompts, start_item
        self.n_items = n_items
        self.priority_idx = None if priority_idx is None else [int(i) for i in priority_idx]
        g = np.random.default_rng([seed, 999_983])
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            self.order = g.choice(len(self.rel), size=n_items + 1, replace=True, p=w / w.sum())
        else:
            n_epochs = n_items // len(self.rel) + 1
            self.order = np.concatenate([g.permutation(len(self.rel)) for _ in range(n_epochs)])

    def __len__(self):
        return self.n_items - self.start

    def __getitem__(self, i):
        k = self.start + i
        rng = np.random.default_rng([self.seed, k])
        x, y = load_rgbi(self.root, self.rel[self.order[k]], self.band_order)
        x, y = augment(x, y, rng)
        cls = sample_prompts(y, self.n_prompts, rng, self.priority_idx)
        target = np.stack([(y == CLASSES[c]) for c in cls]).astype(np.float32)
        valid = (y >= 1) & (y <= 19)
        return to_model_input(x, self.channels), torch.from_numpy(cls), torch.from_numpy(target), torch.from_numpy(valid)


class TestSet(Dataset):
    def __init__(self, root, rel_paths, band_order, channels):
        self.root, self.rel, self.band_order, self.channels = root, list(rel_paths), band_order, channels

    def __len__(self):
        return len(self.rel)

    def __getitem__(self, i):
        x, y = load_rgbi(self.root, self.rel[i], self.band_order)
        return to_model_input(x, self.channels), torch.from_numpy(y), i
