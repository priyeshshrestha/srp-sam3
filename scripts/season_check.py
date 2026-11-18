"""When were the patches flown, and does the NDVI level of a class depend on it? (FLAIR-HUB metadata)

    # 1. the acquisition dates: one gpkg inside the 2.1 GB GLOBAL_ALL_MTD.zip on the Hugging Face hub (public, no token).
    #    --fetch_to downloads the zip, keeps ONLY GLOBAL_AERIAL_MTD_DATES.gpkg, deletes the zip.
    python scripts/season_check.py --split outputs/train/split.json --fetch_to ~/flair_mtd --out outputs/train/season
    # 2. same, plus pseudo-NDVI per class and domain (reads images; --per_domain patches each):
    python scripts/season_check.py --split ... --dates ~/flair_mtd/GLOBAL_AERIAL_MTD_DATES.gpkg --ndvi \
        --root $FLAIR_DATA_DIR --band_order rgb --out outputs/train/season

Writes <out>/season.md, season.json and patch_dates.json ({patch path: YYYYMMDD}); train_report.py reads the last
one (--dates) to add the acquisition date to its per-domain table.

The NDVI here is (NIR-R)/(NIR+R) on 8-bit digital numbers, NOT reflectance, so only compare it across domains with that
caveat in mind. Descriptive statistics, not a test: domains differ in year, camera and stretch as well as in date.
"""
import argparse
import datetime
import json
import os
import shutil
import sys
import zipfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402

from datasets.flair import FLAIR_CLASSES, domain_of  # noqa: E402
from datasets.flair_rgb import hub_patch_id, read_aerial_dates  # noqa: E402

REPO = "IGNF/FLAIR-HUB"
ZIP_PATH, MEMBER = "data/GLOBAL_ALL_MTD.zip", "GLOBAL_AERIAL_MTD_DATES.gpkg"
NDVI_CLASSES = (6, 7, 10, 11)                      # coniferous, deciduous, herbaceous, agricultural
MIN_PIXELS = 2000                                  # a class counts in a patch only above this many pixels


def fetch_dates(dest):
    """Download GLOBAL_ALL_MTD.zip, extract only the aerial-dates gpkg into `dest`, delete the zip."""
    from huggingface_hub import hf_hub_download
    os.makedirs(dest, exist_ok=True)
    target = os.path.join(dest, MEMBER)
    if os.path.exists(target):
        return target
    tmp = os.path.join(dest, ".zip")
    local = hf_hub_download(REPO, ZIP_PATH, repo_type="dataset", local_dir=tmp)
    try:
        with zipfile.ZipFile(local) as z:
            name = next((n for n in z.namelist() if n.split("/")[-1] == MEMBER), None)
            if name is None:
                sys.exit(f"ERROR: {MEMBER} not found inside {ZIP_PATH}")
            with z.open(name) as src, open(target + ".part", "wb") as dst:       # fixed target name: no path from the zip
                shutil.copyfileobj(src, dst)
        os.replace(target + ".part", target)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return target


def doy(date):
    return datetime.datetime.strptime(date, "%Y%m%d").timetuple().tm_yday


def pseudo_ndvi_by_class(root, rels, band_order):
    """{class id: mean NDVI over its pixels} per patch (only classes with >= MIN_PIXELS)."""
    from train.data import load_rgbi
    out = []
    for rel in rels:
        x, y = load_rgbi(root, rel, band_order)
        r, n = x[0].astype(np.float32), x[3].astype(np.float32)
        nd = (n - r) / (n + r + 1e-4)
        out.append({c: float(nd[y == c].mean()) for c in NDVI_CLASSES if (y == c).sum() >= MIN_PIXELS})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--dates", help="GLOBAL_AERIAL_MTD_DATES.gpkg (or a per-domain *_AERIAL_MTD_DATES.gpkg)")
    ap.add_argument("--fetch_to", help="download the dates gpkg into this folder (needs internet)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ndvi", action="store_true", help="also compute pseudo-NDVI per class (reads images)")
    ap.add_argument("--root")
    ap.add_argument("--band_order", choices=["rgb", "bgr"])
    ap.add_argument("--per_domain", type=int, default=20, help="patches sampled per domain for --ndvi")
    a = ap.parse_args()
    if bool(a.dates) == bool(a.fetch_to):
        sys.exit("ERROR: give exactly one of --dates or --fetch_to")
    if a.ndvi and not (a.root and a.band_order):
        sys.exit("ERROR: --ndvi needs --root and --band_order")
    os.makedirs(a.out, exist_ok=True)

    from train.data import load_split
    split = load_split(a.split)
    dates = read_aerial_dates(a.dates or fetch_dates(os.path.expanduser(a.fetch_to)))
    role = {r: "test" for r in split["test"]}
    role.update({r: "train" for r in split["train"]})
    patch_dates = {}
    for rel in role:
        pid = hub_patch_id(rel)
        if pid and pid in dates:
            patch_dates[rel] = dates[pid]
    if not patch_dates:
        sys.exit("ERROR: no split patch has an acquisition date (not FLAIR-HUB paths, or the wrong gpkg?)")
    with open(os.path.join(a.out, "patch_dates.json"), "w") as fh:
        json.dump(patch_dates, fh)

    by_dom = {}
    for rel, d in patch_dates.items():
        by_dom.setdefault(domain_of(rel), []).append(rel)
    rng = np.random.default_rng(0)
    rows = []
    for dom, rels in sorted(by_dom.items()):
        ds = sorted(patch_dates[r] for r in rels)
        row = {"domain": dom, "role": role[rels[0]], "n_patches": len(rels), "first": ds[0], "median": ds[len(ds) // 2],
               "last": ds[-1], "median_doy": int(np.median([doy(d) for d in ds]))}
        if a.ndvi:
            pick = sorted(rels) if len(rels) <= a.per_domain else [sorted(rels)[i] for i in
                                                                  sorted(rng.permutation(len(rels))[:a.per_domain])]
            per = pseudo_ndvi_by_class(a.root, pick, a.band_order)
            row["ndvi"] = {FLAIR_CLASSES[c]: {"mean": float(np.mean([p[c] for p in per if c in p])),
                                              "n_patches": sum(c in p for p in per)}
                           for c in NDVI_CLASSES if any(c in p for p in per)}
            print(f"  {dom}: {len(pick)} patches read", flush=True)
        rows.append(row)

    lines = ["# Acquisition dates" + (" and pseudo-NDVI" if a.ndvi else ""), "",
             f"{len(patch_dates)} of {len(role)} split patches have a date; {len(rows)} domains "
             f"(range {min(r['first'] for r in rows)} to {max(r['last'] for r in rows)}).", "",
             "| Domain | train/test | patches | first | median | last | median day of year |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['domain']} | {r['role']} | {r['n_patches']} | {r['first']} | {r['median']} | {r['last']} | "
              f"{r['median_doy']} |" for r in rows]
    summary = {"domains": rows}
    for role_name in ("train", "test"):
        sel = [r for r in rows if r["role"] == role_name]
        if sel:
            lines.append(f"\n{role_name}: {len(sel)} domains, median day of year {int(np.median([r['median_doy'] for r in sel]))}, "
                         f"earliest {min(r['first'] for r in sel)}, latest {max(r['last'] for r in sel)}.")
    if a.ndvi:
        names = [FLAIR_CLASSES[c] for c in NDVI_CLASSES]
        lines += ["", f"## Pseudo-NDVI per class and domain (mean over up to {a.per_domain} patches with >= {MIN_PIXELS} px "
                      "of the class; 8-bit digital numbers, not reflectance)", "",
                  "| Domain | day | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
        for r in rows:
            cells = [f"{r['ndvi'][n]['mean']:.3f} ({r['ndvi'][n]['n_patches']})" if n in r["ndvi"] else "-" for n in names]
            lines.append(f"| {r['domain']} | {r['median_doy']} | " + " | ".join(cells) + " |")
        lines += ["", "Spread of the domain means per class, and Pearson r with day of year (domains with the class, n >= 5):", "",
                  "| Class | domains | sd of domain means | r vs day |", "|---|---|---|---|"]
        for n in names:
            v = [(r["median_doy"], r["ndvi"][n]["mean"]) for r in rows if n in r["ndvi"]]
            if len(v) < 5:
                continue
            arr = np.array(v)
            rr = float(np.corrcoef(arr.T)[0, 1]) if arr[:, 0].std() > 0 else float("nan")
            lines.append(f"| {n} | {len(v)} | {arr[:, 1].std():.3f} | {rr:+.2f} |")
            summary.setdefault("ndvi_vs_doy", {})[n] = {"n_domains": len(v), "sd": float(arr[:, 1].std()), "r": rr}
        lines += ["", "_A correlation across domains mixes season with year, camera and region; with few domains it shows "
                      "nothing causal._"]
    with open(os.path.join(a.out, "season.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(a.out, "season.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
