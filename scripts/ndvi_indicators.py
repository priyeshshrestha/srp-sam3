"""SAM + NDVI: segment with RGB only, then read ecology off the segments using the NIR band.

For every episode of a finished run this measures, against the ground-truth mask of the same
class on the same query patch:

  * NDVI of the segment          mean NDVI inside the predicted mask vs inside the true mask
  * vegetation cover             predicted vs true fraction of the patch
  * local heterogeneity          mean local Shannon entropy H of NDVI (7 px = 1.4 m window)

and one post-hoc refinement that needs no training and no query labels:

  * NDVI gate  keep only predicted pixels whose NDVI lies inside the [2nd, 98th] percentile of
               NDVI under the SUPPORT mask. The band comes from the support alone, so nothing is
               tuned on the queries being scored.

NIR is channel 3 in both FLAIR #1 (B/R,G,R/B,NIR,nDSM) and FLAIR-HUB (R,G,B,NIR); red is channel
0 for band order 'rgb' and 2 for 'bgr' (the band order recorded in the episodes file).

    python scripts/ndvi_indicators.py --root ./data --episodes outputs/episodes.json \
        --run outputs/runs/sam3_facebook-sam3_1shot --out outputs/ndvi/sam3_1shot.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.flair import FLAIR_CLASSES, _read_tif, mask_path_for  # noqa: E402
from datasets.flair_rgb import read_label  # noqa: E402
from ecological.indices import compute_ndvi_numpy, compute_shannon_entropy_torch  # noqa: E402
from evaluation.episodes import load_episodes  # noqa: E402
from evaluation.metrics import class_miou, confusion  # noqa: E402
from evaluation.report_lib import load_pred, load_run  # noqa: E402

VEGETATION = (6, 7, 8, 9, 10, 11, 12)
NIR = 3


def ndvi_of(path: str, band_order: str) -> np.ndarray:
    raw = _read_tif(path).astype(np.float32)
    if raw.shape[0] < 4:
        raise ValueError(f"{path} has {raw.shape[0]} channels; NDVI needs NIR (channel 3)")
    optical = raw[:4] / 255.0 if raw[:4].max() > 1.0 else raw[:4]
    red = optical[0] if band_order == "rgb" else optical[2]
    return compute_ndvi_numpy(optical[NIR], red)


def local_entropy(ndvi: np.ndarray, window: int, bins: int) -> np.ndarray:
    t = torch.from_numpy((ndvi + 1.0) / 2.0)[None, None].float()
    return compute_shannon_entropy_torch(t, kernel_size=window, num_bins=bins)[0, 0].numpy()


def support_band(ndvi_s: np.ndarray, mask_s: np.ndarray, lo_pct=2.0, hi_pct=98.0):
    vals = ndvi_s[mask_s]
    return float(np.percentile(vals, lo_pct)), float(np.percentile(vals, hi_pct))


def gate(pred: np.ndarray, ndvi_q: np.ndarray, band) -> np.ndarray:
    lo, hi = band
    return pred & (ndvi_q >= lo) & (ndvi_q <= hi)


def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--episodes", default="outputs/episodes.json")
    ap.add_argument("--run", required=True, help="a finished 1-shot run directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=7)
    ap.add_argument("--bins", type=int, default=16)
    args = ap.parse_args()

    ep_file = load_episodes(args.episodes)
    band_order = ep_file["meta"]["band_order"]
    episodes = {e["id"]: e for e in ep_file["episodes"]}
    run = load_run(args.run)
    if run["meta"].get("prompt", "fewshot") != "fewshot":
        sys.exit("ndvi_indicators needs a few-shot run (the NDVI gate is derived from the support)")

    rows, gated_records, raw_records = [], [], []
    for i, rec in sorted(run["records"].items()):
        ep = episodes[i]
        c = ep["class_id"]
        q_path = os.path.join(args.root, ep["query"])
        s_path = os.path.join(args.root, ep["support"])
        ndvi_q = ndvi_of(q_path, band_order)
        gt = read_label(mask_path_for(q_path)) == c
        pred = load_pred(args.run, i)

        ndvi_s = ndvi_of(s_path, band_order)
        s_mask = read_label(mask_path_for(s_path)) == c
        band = support_band(ndvi_s, s_mask)
        g = gate(pred, ndvi_q, band)
        raw_records.append({"id": i, "class_id": c, **confusion(pred, gt)})
        gated_records.append({"id": i, "class_id": c, **confusion(g, gt)})

        if c in VEGETATION:
            H = local_entropy(ndvi_q, args.window, args.bins)
            rows.append({
                "id": i, "class_id": c, "empty": bool(not pred.any()),
                "ndvi_gt": float(ndvi_q[gt].mean()),
                "ndvi_pred": float(ndvi_q[pred].mean()) if pred.any() else None,
                "cover_gt": float(gt.mean()), "cover_pred": float(pred.mean()),
                "H_gt": float(H[gt].mean()),
                "H_pred": float(H[pred].mean()) if pred.any() else None,
            })

    per_class = {}
    for c in VEGETATION:
        rs = [r for r in rows if r["class_id"] == c]
        ok = [r for r in rs if r["ndvi_pred"] is not None]
        if not rs:
            continue
        per_class[FLAIR_CLASSES[c]] = {
            "n": len(rs), "n_empty": len(rs) - len(ok),
            "ndvi_gt_mean": float(np.mean([r["ndvi_gt"] for r in rs])),
            "ndvi_pred_mean": float(np.mean([r["ndvi_pred"] for r in ok])) if ok else None,
            "ndvi_abs_err": float(np.mean([abs(r["ndvi_pred"] - r["ndvi_gt"]) for r in ok])) if ok else None,
            "cover_abs_err_pp": float(100 * np.mean([abs(r["cover_pred"] - r["cover_gt"]) for r in rs])),
        }
    ok = [r for r in rows if r["ndvi_pred"] is not None]
    miou_raw, per_raw = class_miou(raw_records)
    miou_g, per_g = class_miou(gated_records)
    veg = [c for c in per_raw if c in VEGETATION]
    other = [c for c in per_raw if c not in VEGETATION]
    summary = {
        "run": run["name"], "band_order": band_order, "window_px": args.window,
        "n_vegetation_episodes": len(rows), "n_empty_predictions": len(rows) - len(ok),
        "ndvi_abs_err_mean": float(np.mean([abs(r["ndvi_pred"] - r["ndvi_gt"]) for r in ok])) if ok else None,
        "ndvi_r": pearson([r["ndvi_gt"] for r in ok], [r["ndvi_pred"] for r in ok]),
        "cover_abs_err_pp": float(100 * np.mean([abs(r["cover_pred"] - r["cover_gt"]) for r in rows])) if rows else None,
        "cover_r": pearson([r["cover_gt"] for r in rows], [r["cover_pred"] for r in rows]),
        "H_r": pearson([r["H_gt"] for r in ok], [r["H_pred"] for r in ok]),
        "H_gt_mean": float(np.mean([r["H_gt"] for r in ok])) if ok else None,
        "H_pred_mean": float(np.mean([r["H_pred"] for r in ok])) if ok else None,
        "gate": {
            "miou_raw": miou_raw, "miou_gated": miou_g,
            "veg_raw": float(np.mean([per_raw[c] for c in veg])) if veg else None,
            "veg_gated": float(np.mean([per_g[c] for c in veg])) if veg else None,
            "other_raw": float(np.mean([per_raw[c] for c in other])) if other else None,
            "other_gated": float(np.mean([per_g[c] for c in other])) if other else None,
            "per_class_raw": {FLAIR_CLASSES[c]: v for c, v in per_raw.items()},
            "per_class_gated": {FLAIR_CLASSES[c]: v for c, v in per_g.items()},
        },
        "per_class": per_class,
        "episodes": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"{run['name']}: NDVI |err| {summary['ndvi_abs_err_mean']}, cover |err| {summary['cover_abs_err_pp']} pp, "
          f"H r {summary['H_r']:.3f}; gate mIoU {miou_raw:.4f} -> {miou_g:.4f}")


if __name__ == "__main__":
    main()
