"""Training-data focus on the classes that matter for biodiversity: per-patch sampling weights.

    python scripts/make_focus.py --root $FLAIR_DATA_DIR --split outputs/train/split.json \
        --priority 5 6 7 8 10 --floor 0.1 --out outputs/train/focus.json

Weight of a TRAINING patch = floor + (fraction of its pixels in the priority classes). Training then draws
patches with replacement in proportion to the weight and prompts present priority classes first. The
TEST set is never touched (still the complete official set), and every class keeps being trained and
evaluated: patches with few priority pixels are just drawn less often (a patch with none is drawn
floor / (1 + floor) as often as one that is all priority).

Class ids are FLAIR #1: 5 water, 6 coniferous, 7 deciduous, 8 brushwood, 9 vineyard, 10 herbaceous
vegetation, 11 agricultural land, 12 plowed land. Reads the label of every training patch once
(threaded, cached); re-running with the same settings reuses the file, different settings are refused
unless --force (finished runs are tied to the focus file they were trained with).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402

from datasets.flair import FLAIR_CLASSES  # noqa: E402
from datasets.flair_rgb import build_class_counts  # noqa: E402
from train.data import focus_hash, focus_weights, load_split  # noqa: E402
from train.sam3_rgbi import CLASSES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--priority", type=int, nargs="+", required=True)
    ap.add_argument("--floor", type=float, default=0.1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--counts_cache", default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    bad = [c for c in a.priority if c not in CLASSES]
    if bad:
        sys.exit(f"ERROR: priority classes must be among the 12 evaluated classes {CLASSES[0]}..{CLASSES[-1]}, got {bad}")
    if a.floor <= 0:
        sys.exit("ERROR: --floor must be > 0 (every training patch must stay possible to draw)")
    split = load_split(a.split)
    counts = build_class_counts(a.root, split["train"], cache=a.counts_cache, workers=a.workers)
    w = focus_weights(counts, split["train"], a.priority, a.floor)
    focus = {"priority": sorted(a.priority), "floor": a.floor, "split_sha": split["sha"], "weights": w}
    focus["sha"] = focus_hash(focus)

    if os.path.exists(a.out) and not a.force:
        old = json.load(open(a.out))
        if old.get("sha") == focus["sha"]:
            print(f"  reusing {a.out} (identical focus, sha {old['sha']})")
        else:
            sys.exit(f"ERROR: {a.out} holds a DIFFERENT focus (sha {old.get('sha')} vs {focus['sha']}); runs trained "
                     f"on it would be mixed with new ones. Use another output folder or pass --force.")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(focus, fh)
        print(f"  wrote {a.out}")

    # What the weighting does, so the numbers can be quoted: share of priority pixels seen in training.
    rels = split["train"]
    wt = np.array([w[r] for r in rels])
    frac = np.array([sum(counts[r].get(c, 0) for c in a.priority) / max(1, sum(counts[r].values())) for r in rels])
    names = ", ".join(f"{c} {FLAIR_CLASSES[c]}" for c in sorted(a.priority))
    print(f"  priority classes: {names}   |   floor {a.floor}   |   sha {focus['sha']}")
    print(f"  priority pixels per training patch: uniform sampling {100 * frac.mean():.1f} %  ->  "
          f"focused sampling {100 * float((wt * frac).sum() / wt.sum()):.1f} %")
    print(f"  patches with >= 50 % priority pixels: {100 * (frac >= 0.5).mean():.1f} % of the pool, "
          f"{100 * float(wt[frac >= 0.5].sum() / wt.sum()):.1f} % of the draws")


if __name__ == "__main__":
    main()
