"""Download the FLAIR-HUB aerial RGBI images + COSIA labels needed for the training experiment.

    python scripts/download_flair_hub.py --splits_dir ~/srp_phase2/data/splits --dest ~/flair_hub --budget_gb 100
    python scripts/download_flair_hub.py ... --dry_run          # show what would be downloaded, download nothing

Only two of FLAIR-HUB's modalities are fetched (AERIAL_RGBI and AERIAL_LABEL-COSIA; ~243 GB for all 74
domains, vs ~750 GB for everything). Selection, using the OFFICIAL split CSVs (FLAIR-HUB_TRAIN/VALID/TEST.csv):
  1. every domain that has TEST patches (the official test set must be complete)
  2. then training domains in a fixed (seed 0) random order while the total stays within --budget_gb
The split CSVs are copied to <dest>/splits, where make_split.py finds them.

Resumable: each domain zip is downloaded, unpacked into <dest>, deleted, and recorded in
<dest>/.downloaded; re-running skips finished domains. Public dataset, no token needed.
Source: huggingface.co/datasets/IGNF/FLAIR-HUB (Open Licence 2.0 / etalab-2.0: attribute IGN).
"""
import argparse
import csv
import json
import os
import random
import shutil
import sys
import zipfile

REPO = "IGNF/FLAIR-HUB"
KINDS = ("AERIAL_RGBI", "AERIAL_LABEL-COSIA")
SPLIT_FILES = {"train": "FLAIR-HUB_TRAIN.csv", "valid": "FLAIR-HUB_VALID.csv", "test": "FLAIR-HUB_TEST.csv"}


def domains_in_csv(path):
    """{domain: n_patches} from a FLAIR-HUB split CSV (patch_id = '<D012-2019>_<roi>_<x-y>')."""
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh, delimiter=";"):
            dom = row["patch_id"].split("_", 1)[0]
            out[dom] = out.get(dom, 0) + 1
    return out


def select_domains(per_split, size_gb, budget_gb):
    """All test domains, then training domains in a fixed (seed 0) random order within the budget.
    Returns (test_domains, train_domains, total_gb)."""
    test = sorted(d for d in per_split["test"] if d in size_gb)
    pool = sorted(d for d in per_split["train"] if d in size_gb and d not in test)
    random.Random(0).shuffle(pool)
    total = sum(size_gb[d] for d in test)
    if total > budget_gb:
        raise ValueError(f"the official test domains alone need {total:.1f} GB > budget {budget_gb} GB")
    train = []
    for d in pool:
        if total + size_gb[d] <= budget_gb:
            train.append(d)
            total += size_gb[d]
    return test, train, total


def remote_sizes():
    from huggingface_hub import HfApi
    sizes = {}
    for item in HfApi().list_repo_tree(REPO, repo_type="dataset", path_in_repo="data"):
        name = item.path.split("/")[-1]
        for kind in KINDS:
            if name.endswith(f"_{kind}.zip"):
                sizes.setdefault(name[: -len(f"_{kind}.zip")], {})[kind] = (item.path, item.size or 0)
    return sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", required=True, help="folder with FLAIR-HUB_TRAIN/VALID/TEST.csv")
    ap.add_argument("--dest", required=True)
    ap.add_argument("--budget_gb", type=float, default=100.0)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    for f in SPLIT_FILES.values():
        if not os.path.exists(os.path.join(a.splits_dir, f)):
            sys.exit(f"ERROR: {f} not found in {a.splits_dir}")
    per_split = {k: domains_in_csv(os.path.join(a.splits_dir, f)) for k, f in SPLIT_FILES.items()}
    sizes = remote_sizes()

    def gb(dom):
        return sum(s for _, s in sizes.get(dom, {}).values()) / 1e9

    missing = [d for k in ("train", "test") for d in per_split[k] if set(sizes.get(d, {})) != set(KINDS)]
    if missing:
        print(f"  WARNING: {len(missing)} domain(s) in the CSVs have no RGBI/label zip on the hub: {missing[:5]}")
    try:
        test, train_chosen, total = select_domains(per_split, {d: gb(d) for d in sizes}, a.budget_gb)
    except ValueError as e:
        sys.exit(f"ERROR: {e}")
    chosen = test + train_chosen
    n_train = sum(per_split["train"].get(d, 0) for d in chosen)
    n_test = sum(per_split["test"].get(d, 0) for d in chosen)
    overlap = sorted(set(per_split["train"]) & set(per_split["test"]))
    print(f"  official split: {len(per_split['train'])} train / {len(per_split['valid'])} valid / "
          f"{len(per_split['test'])} test domains; domains in both train and test: {len(overlap)}")
    print(f"  selected: {len(test)} test domains ({sum(gb(d) for d in test):.1f} GB, {n_test} test patches) + "
          f"{len(train_chosen)} train domains  ->  {total:.1f} GB, {n_train} train patches")
    print(f"  test domains : {', '.join(test)}")
    print(f"  train domains: {', '.join(sorted(train_chosen))}")
    if a.dry_run:
        return

    from huggingface_hub import hf_hub_download
    os.makedirs(a.dest, exist_ok=True)
    shutil.copytree(a.splits_dir, os.path.join(a.dest, "splits"), dirs_exist_ok=True)
    free_gb = shutil.disk_usage(a.dest).free / 1e9
    if free_gb < total * 1.3:
        sys.exit(f"ERROR: only {free_gb:.0f} GB free at {a.dest}; need ~{total * 1.3:.0f} GB (zip + unpacked)")
    done_path = os.path.join(a.dest, ".downloaded")
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
    tmp = os.path.join(a.dest, ".zips")
    for i, dom in enumerate(chosen, 1):
        for kind in KINDS:
            key = f"{dom}_{kind}"
            if key in done:
                continue
            path, size = sizes[dom][kind]
            print(f"  [{i}/{len(chosen)}] {key}  ({size / 1e9:.2f} GB)", flush=True)
            local = hf_hub_download(REPO, path, repo_type="dataset", local_dir=tmp)
            with zipfile.ZipFile(local) as z:
                bad = [n for n in z.namelist() if n.startswith("/") or ".." in n.split("/")]
                if bad:
                    sys.exit(f"ERROR: {path} contains unsafe paths {bad[:3]}; not extracting")
                z.extractall(a.dest)
            os.remove(local)
            with open(done_path, "a") as fh:
                fh.write(key + "\n")
            done.add(key)
    shutil.rmtree(tmp, ignore_errors=True)
    with open(os.path.join(a.dest, "download_manifest.json"), "w") as fh:
        json.dump({"repo": REPO, "budget_gb": a.budget_gb, "total_gb": total, "test_domains": test,
                   "train_domains": sorted(train_chosen)}, fh, indent=1)
    print(f"  done: {a.dest}  ({total:.1f} GB).  Next:  FLAIR_DATA_DIR={a.dest} bash run_train.sh check")


if __name__ == "__main__":
    main()
