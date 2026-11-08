"""Write the train/test split for the training experiment (held-out WHOLE domains).

    python scripts/make_split.py --root $FLAIR_DATA_DIR --out outputs/train/split.json [--mode auto]

Re-running with the same settings reuses the existing file. Different settings are refused unless
--force, because every finished run is tied to the split it was trained and tested on.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.flair import _read_tif  # noqa: E402
from datasets.flair_rgb import discover_patches  # noqa: E402
from train.data import load_split, make_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="auto", choices=["auto", "hub_csv", "official", "domain"])
    ap.add_argument("--csv_dir", default=None, help="folder with the official FLAIR-HUB split CSVs "
                    "[default: <root>/splits]")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    rels = discover_patches(a.root)
    ch = _read_tif(os.path.join(a.root, rels[0])).shape[0]
    if ch < 4:
        sys.exit(f"ERROR: {rels[0]} has {ch} channels; RGB+NIR training needs R, G, B, NIR (>= 4)")
    csv_dir = a.csv_dir or os.path.join(a.root, "splits")
    split = make_split(rels, mode=a.mode, test_frac=a.test_frac, seed=a.seed, csv_dir=csv_dir)

    if os.path.exists(a.out) and not a.force:
        old = load_split(a.out)
        if old["sha"] == split["sha"]:
            print(f"  reusing {a.out} (identical split, sha {old['sha']})")
            split = old
        else:
            sys.exit(f"ERROR: {a.out} holds a DIFFERENT split (sha {old['sha']} vs {split['sha']}); runs trained "
                     f"on it would be mixed with new ones. Move the old outputs away or pass --force.")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(split, fh, indent=1)
        print(f"  wrote {a.out}")

    print(f"  split mode: {split['mode']}  |  train {len(split['train'])} patches / {len(split['train_domains'])} domains"
          f"  |  test {len(split['test'])} patches / {len(split['test_domains'])} domains  |  sha {split['sha']}")
    print(f"  test domains: {', '.join(split['test_domains'])}")
    if split.get("n_unassigned"):
        print(f"  {split['n_unassigned']} patches on disk are in neither split (e.g. the official VALID set): unused")
    if split["domain_overlap"]:
        print(f"  WARNING: {len(split['domain_overlap'])} domain(s) occur in BOTH train and test "
              f"({', '.join(split['domain_overlap'][:5])}...): the test is not fully out-of-domain.")


if __name__ == "__main__":
    main()
