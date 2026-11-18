"""Does the trained RGB+NIR model actually USE the NIR band? Three checks on finished training runs.

    python scripts/nir_diagnostic.py --train_dir outputs/train --root $FLAIR_DATA_DIR --band_order rgb

1. Weights (no GPU): size of the learned NIR input filters relative to the R, G, B filters (they started
   at exactly zero), and how far the RGB filters moved from pretrained SAM 3 (both arms train them).
2. Loss curves (no GPU): mean training loss early / middle / late, per run -- still falling at the end?
3. NIR ablation (GPU): each RGB+NIR model on a fixed test subset with
       normal     the real NIR band
       nir_flat   NIR replaced by its own patch mean (no spatial NIR information left)
       nir_swap   NIR taken from a DIFFERENT test patch (realistic values, wrong content)
   next to the RGB-only model of the same seed on the same patches. A large drop under nir_flat /
   nir_swap means the model relies on NIR. Caveat: both are inputs the model never saw in training, so a
   drop below the RGB-only model is possible even with modest reliance.

Resumable (each run x condition is saved as it finishes). Exit 3 = out of GPU memory.
Writes <train_dir>/diagnostic/nir_diagnostic.{json,md}.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from datasets.flair import FLAIR_CLASSES  # noqa: E402
from train.data import CHANNELS, load_rgbi, load_split, test_subset  # noqa: E402
from train.metrics import K, confusion, iou_per_class, miou  # noqa: E402
from train.sam3_rgbi import CLASSES, PATCH_EMBED, SAM3_MODEL_ID, normalise, predict_classes  # noqa: E402

CONDITIONS = ("normal", "nir_flat", "nir_swap")
VEG = (6, 7, 8, 10, 11)


def f3(v):
    return "—" if v is None else f"{v:.3f}"


def pretrained_patch_embed():
    """(1024, 3, 14, 14) pretrained SAM 3 patch-embedding weight, read from the cached safetensors only."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    path = hf_hub_download(SAM3_MODEL_ID, "model.safetensors")
    with safe_open(path, "pt") as fh:
        key = next(k for k in fh.keys() if k.endswith("patch_embeddings.projection.weight"))
        return fh.get_tensor(key).float()


def weight_report(runs):
    w0, out = None, {}
    for name, d in runs.items():
        st = torch.load(os.path.join(d, "final.pt"), map_location="cpu")["trainable"]
        key = next((k for k in st if k.endswith(PATCH_EMBED + ".weight")), None)
        if key is None:                                  # not a SAM 3 run (e.g. the tiny test stand-in)
            continue
        if w0 is None:
            w0 = pretrained_patch_embed()
        w = st[key].float()
        norms = [float(w[:, c].norm()) for c in range(w.shape[1])]
        rec = {"norm_R": norms[0], "norm_G": norms[1], "norm_B": norms[2],
               "rgb_rel_change": float((w[:, :3] - w0).norm() / w0.norm())}
        if w.shape[1] == 4:
            rec["norm_NIR"] = norms[3]
            rec["nir_over_mean_rgb"] = norms[3] / float(np.mean(norms[:3]))
        out[name] = rec
    return out


def loss_report(runs):
    out = {}
    for name, d in runs.items():
        p = os.path.join(d, "train_log.jsonl")
        if not os.path.exists(p):
            continue
        recs = [json.loads(l) for l in open(p) if l.strip()]
        steps = max(r["step"] for r in recs)

        def window(lo, hi):
            v = [r["loss"] for r in recs if lo < r["step"] <= hi]
            return float(np.mean(v)) if v else None
        q = steps // 8
        out[name] = {"steps": steps, "first_eighth": window(0, q), "middle_eighth": window(4 * q - q, 4 * q),
                     "last_eighth": window(steps - q, steps), "second_last_eighth": window(steps - 2 * q, steps - q)}
    return out


def ablation(runs, a, rels, out_dir, dev):
    from scripts.eval_sam3 import load_net
    res = {}
    for name, d in sorted(runs.items()):
        cfg = json.load(open(os.path.join(d, "config.json")))
        conds = CONDITIONS if CHANNELS[cfg["channels"]] == 4 else ("normal",)       # arms that see NIR at all
        todo = [c for c in conds if not os.path.exists(os.path.join(out_dir, f"abl_{name}_{c}.npy"))]
        if todo:
            net = load_net(d, cfg["model"], cfg["channels"], cfg["lora_rank"], False, dev)
            n_ch = CHANNELS[cfg["channels"]]
            for c in todo:
                conf = np.zeros((K, K), np.int64)
                with torch.inference_mode():
                    for i, rel in enumerate(rels):
                        x, y = load_rgbi(a.root, rel, a.band_order)
                        x = x.copy()
                        if c == "nir_flat":
                            x[3] = int(round(float(x[3].mean())))
                        elif c == "nir_swap":
                            x[3] = load_rgbi(a.root, rels[(i + len(rels) // 2) % len(rels)], a.band_order)[0][3]
                        xt = normalise(torch.from_numpy(x[:n_ch]).unsqueeze(0).to(dev))
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                            pred = predict_classes(net, xt, out_size=y.shape)[0].cpu().numpy()
                        conf += confusion(pred, y)
                np.save(os.path.join(out_dir, f"abl_{name}_{c}.npy"), conf)
                print(f"  {name} {c}: mIoU {miou(conf):.4f}", flush=True)
            del net
            if dev == "cuda":
                torch.cuda.empty_cache()
        for c in conds:
            res.setdefault(name, {})[c] = np.load(os.path.join(out_dir, f"abl_{name}_{c}.npy"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", default="outputs/train")
    ap.add_argument("--root", required=True)
    ap.add_argument("--band_order", required=True, choices=["rgb", "bgr"])
    ap.add_argument("--n_images", type=int, default=1000, help="fixed test subset for the ablation")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip_ablation", action="store_true")
    a = ap.parse_args()
    from scripts.train_sam3 import pick_device

    run_dirs = sorted(glob.glob(os.path.join(a.train_dir, "runs", "rgb*_s*")))
    runs = {os.path.basename(d): d for d in run_dirs if os.path.exists(os.path.join(d, "final.pt"))}
    if not any(n.startswith("rgbi_") for n in runs):
        sys.exit(f"ERROR: no finished RGB+NIR run (final.pt) under {a.train_dir}/runs")
    out_dir = os.path.join(a.train_dir, "diagnostic")
    os.makedirs(out_dir, exist_ok=True)

    print("1. NIR input weights")
    W = weight_report(runs)
    print("2. Loss curves")
    L = loss_report(runs)
    A = {}
    if not a.skip_ablation:
        split = load_split(os.path.join(a.train_dir, "split.json"))
        rels = test_subset(split, a.n_images)
        print(f"3. NIR ablation on {len(rels)} fixed test patches")
        dev = pick_device(a.device)
        try:
            A = ablation(runs, a, rels, out_dir, dev)
        except torch.cuda.OutOfMemoryError:
            print("OUT OF GPU MEMORY (finished parts are saved; re-run to resume)")
            sys.exit(3)

    lines = ["# NIR diagnostic", "", "## 1. Learned input filters (patch embedding, L2 norm per input band)", "",
             "NIR filters started at exactly 0. `NIR / mean RGB` = how large the NIR filters grew relative to the "
             "colour filters. `RGB moved` = relative change of the R, G, B filters from pretrained SAM 3.", "",
             "| Run | R | G | B | NIR | NIR / mean RGB | RGB moved |", "|---|---|---|---|---|---|---|"]
    for n, r in sorted(W.items()):
        lines.append(f"| {n} | {r['norm_R']:.3f} | {r['norm_G']:.3f} | {r['norm_B']:.3f} | "
                     f"{r.get('norm_NIR', float('nan')):.3f} | {r.get('nir_over_mean_rgb', float('nan')):.3f} | "
                     f"{100 * r['rgb_rel_change']:.2f} % |")
    lines += ["", "## 2. Training loss (mean over each eighth of training)", "",
              "| Run | steps | first 1/8 | middle 1/8 | second-last 1/8 | last 1/8 | still falling? |", "|---|---|---|---|---|---|---|"]
    for n, r in sorted(L.items()):
        fall = (r["last_eighth"] is not None and r["second_last_eighth"] is not None
                and r["last_eighth"] < r["second_last_eighth"] - 0.005)
        lines.append(f"| {n} | {r['steps']} | {f3(r['first_eighth'])} | {f3(r['middle_eighth'])} | "
                     f"{f3(r['second_last_eighth'])} | {f3(r['last_eighth'])} | {'yes' if fall else 'flat'} |")
    lines.append("")
    lines.append("_Each run's batches differ along the schedule, so compare runs at the same eighth, not across eighths "
                 "of one run too finely; the learning rate also decays to 0 (cosine), which flattens the end._")

    summary = {"weights": W, "loss": L}
    if A:
        lines += ["", f"## 3. NIR ablation (mIoU on {a.n_images} fixed test patches)", "",
                  "| Run | RGB-only model (same seed) | real NIR | NIR flattened | NIR from another patch |",
                  "|---|---|---|---|---|"]
        abl = {}
        for n in sorted(k for k in A if k.startswith(("rgbi_", "rgb_ndvi_"))):
            seed = n.rsplit("_s", 1)[1]
            rgb = A.get(f"rgb_s{seed}", {}).get("normal")
            row = {c: miou(A[n][c]) for c in CONDITIONS}
            row["rgb_only"] = miou(rgb) if rgb is not None else None
            abl[n] = row
            lines.append(f"| {n} | {row['rgb_only']:.4f} | {row['normal']:.4f} | {row['nir_flat']:.4f} | "
                         f"{row['nir_swap']:.4f} |" if row["rgb_only"] is not None else
                         f"| {n} | — | {row['normal']:.4f} | {row['nir_flat']:.4f} | {row['nir_swap']:.4f} |")
        # per class, pooled over seeds
        raw = [n for n in abl if n.startswith("rgbi_s")]            # the plain RGB+NIR arm (not the NDVI-adapter arm)
        pooled = {c: sum(A[n][c] for n in raw) for c in CONDITIONS}
        iou = {c: iou_per_class(pooled[c]) for c in CONDITIONS}
        lines += ["", "Per class (seeds pooled): IoU with real NIR, and the change when NIR is flattened / swapped.", "",
                  "| Class | real NIR | flattened Δ | swapped Δ |", "|---|---|---|---|"]
        for j, c in enumerate(CLASSES):
            lines.append(f"| {FLAIR_CLASSES[c]} | {iou['normal'][j]:.3f} | {iou['nir_flat'][j] - iou['normal'][j]:+.3f} | "
                         f"{iou['nir_swap'][j] - iou['normal'][j]:+.3f} |")
        summary["ablation"] = abl
        summary["ablation_per_class"] = {int(c): {k: float(iou[k][j]) for k in CONDITIONS} for j, c in enumerate(CLASSES)}
        lines += ["", "_Reading: a large drop when NIR is flattened or swapped = the model relies on NIR. Both are "
                  "inputs never seen in training, so falling below the RGB-only model does not by itself prove heavy "
                  "reliance._"]

    with open(os.path.join(out_dir, "nir_diagnostic.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    with open(os.path.join(out_dir, "nir_diagnostic.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
