"""Decide empirically whether the first three FLAIR channels are (R,G,B) or (B,G,R).

Run this on the real files BEFORE make_episodes.py. Two independent pieces of evidence:

  1. Class statistics. Bare soil, plowed land and (mostly) buildings/impervious surfaces in
     France are reddish/beige: Red > Blue. Water and swimming pools are the reverse. If channel 0
     is brighter than channel 2 on soil-like classes, channel 0 is Red (-> order 'rgb').
  2. A montage with BOTH readings side by side, for a human to judge. Soil should look brown or
     beige, vegetation green, water blue-green or dark. Blue-grey soil means the wrong order.

The statistic is a heuristic, not proof; the montage is the tiebreaker.

    python scripts/check_bands.py --root ./data --out outputs/band_check.png
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.flair import FLAIR_CLASSES, _read_tif, mask_path_for  # noqa: E402
from datasets.flair_rgb import discover_patches, optical_to_rgb_uint8, read_label  # noqa: E402
from evaluation.bands import decide_band_order  # noqa: E402

SOIL_LIKE = (4, 12)      # bare soil, plowed land
WATER_LIKE = (5, 13)     # water, swimming pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--samples", type=int, default=200, help="patches to scan")
    ap.add_argument("--out", default="outputs/band_check.png")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write_order", default=None,
                    help="write 'rgb' or 'bgr' to this file ONLY when the statistics are conclusive")
    args = ap.parse_args()

    rels = discover_patches(args.root)
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(rels), size=min(args.samples, len(rels)), replace=False)

    sums, counts = {}, {}
    montage_items = []
    for i in pick:
        raw = _read_tif(os.path.join(args.root, rels[i])).astype(np.float32)
        mask = read_label(mask_path_for(os.path.join(args.root, rels[i])))
        for c in np.unique(mask):
            sel = mask == c
            sums[int(c)] = sums.get(int(c), 0.0) + raw[:3][:, sel].sum(axis=1)
            counts[int(c)] = counts.get(int(c), 0) + int(sel.sum())
        present = set(np.unique(mask).tolist())
        if len(montage_items) < 6 and present & set(SOIL_LIKE + WATER_LIKE):
            montage_items.append(raw)

    print(f"\nMean of channels 0/1/2 per class ({len(pick)} patches)\n")
    print(f"  {'class':<24}{'ch0':>8}{'ch1':>8}{'ch2':>8}   ch0-ch2")
    for c in sorted(sums):
        m = sums[c] / counts[c]
        print(f"  {FLAIR_CLASSES.get(c, c):<24}{m[0]:>8.1f}{m[1]:>8.1f}{m[2]:>8.1f}   {m[0]-m[2]:>+7.1f}")

    def group_diff(classes):
        rows = [sums[c] / counts[c] for c in classes if c in sums]
        return float(np.mean([r[0] - r[2] for r in rows])) if rows else None

    soil_d, water_d = group_diff(SOIL_LIKE), group_diff(WATER_LIKE)
    if soil_d is not None:
        print(f"\n  soil-like classes : ch0-ch2 = {soil_d:+.1f}   (Red > Blue expected -> positive means 'rgb')")
    if water_d is not None:
        print(f"  water-like classes: ch0-ch2 = {water_d:+.1f}   (Blue >= Red expected -> negative means 'rgb')")
    order = decide_band_order(soil_d, water_d)
    if order:
        print(f"\n  Statistics agree: band_order = '{order}'. Still glance at the montage to confirm.")
        if args.write_order:
            os.makedirs(os.path.dirname(os.path.abspath(args.write_order)), exist_ok=True)
            with open(args.write_order, "w") as fh:
                fh.write(order + "\n")
    else:
        print("\n  Statistics missing, weak or contradictory: NOT deciding automatically.")
        print("  Open the montage and choose; then pass --band_order / set BAND_ORDER=rgb|bgr.")

    if montage_items:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(montage_items)
        fig, ax = plt.subplots(2, n, figsize=(3 * n, 6.4), squeeze=False)
        for j, raw in enumerate(montage_items):
            for r, order in enumerate(("rgb", "bgr")):
                ax[r][j].imshow(optical_to_rgb_uint8(raw, order))
                ax[r][j].axis("off")
                if j == 0:
                    ax[r][j].set_title(f"read as '{order}'", loc="left", fontsize=12, fontweight="bold")
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        fig.savefig(args.out, dpi=110, bbox_inches="tight")
        print(f"\n  Montage written to {args.out}  (top row: 'rgb', bottom row: 'bgr')")


if __name__ == "__main__":
    main()
