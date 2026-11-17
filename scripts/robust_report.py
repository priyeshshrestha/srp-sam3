"""Report for the NIR-robustness and few-shot-adaptation experiments (no GPU needed).

    python scripts/robust_report.py --train_dir outputs/train --out outputs/train/robust_report

Reads <train_dir>/shift/*.npz (scripts/nir_shift.py) and <train_dir>/adapt/*/cells/*.npz
(scripts/adapt_kshot.py). Three arms, recognised from each run's config:
    RGB              channels rgb
    RGB+NIR          channels rgbi
    RGB+NIR robust   channels rgbi + nir_aug (NIR-robust training)

1. NIR shift: mIoU per simulated camera (mean +- sd over seeds) and paired differences with 95% CIs
   (bootstrap over test images, seeds pooled; only seeds present in both arms). RGB is unaffected by every
   NIR condition, so its "normal" number is the reference in every row.
2. Tree classes (coniferous, deciduous) under the key conditions: where the NIR reliance was found.
3. Few-shot adaptation to unseen domains: mIoU by K (pooled over the chosen domains), and the difference
   to RGB at the same K with a 95% CI (paired by evaluation image).
Figures: gain_curve.png (mIoU vs NIR gain), kshot_curve.png (mIoU vs K).
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402

from datasets.flair import FLAIR_CLASSES  # noqa: E402
from train.metrics import iou_per_class, miou, paired_bootstrap_delta  # noqa: E402
from train.nir import OUT_OF_TRAINING_RANGE, parse_condition  # noqa: E402

ARMS = [("rgb", "RGB"), ("rgbi", "RGB+NIR"), ("rgbiaug", "RGB+NIR robust")]
LABEL = dict(ARMS)
TREES = (6, 7)
COND_TEXT = {"normal": "real NIR", "gain_0.6": "NIR x0.6", "gain_0.8": "NIR x0.8", "gain_1.2": "NIR x1.2",
             "gain_1.4": "NIR x1.4", "offset_-20": "NIR -20", "offset_+20": "NIR +20", "gamma_1.25": "NIR gamma 1.25",
             "camera_b": "camera B (x0.8, +15, gamma 1.15)", "missing": "no NIR"}


def cond_text(c):
    base, cal = parse_condition(c)
    return COND_TEXT.get(base, base) + (" + calibration" if cal else "")


def seen_in_training(c):
    base, _ = parse_condition(c)
    if base == "normal":
        return "–"
    return "no" if base in OUT_OF_TRAINING_RANGE else "yes"


def arm_of(cfg):
    if cfg.get("channels") == "rgb":
        return "rgb"
    return "rgbiaug" if cfg.get("nir_aug") else "rgbi"


def f4(v):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"


def ci(d):
    return f"{d['delta']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]{' *' if d['significant'] else ''}"


# ------------------------------------------------------------------------------------- shift
def load_shift(train_dir):
    runs = {}
    for js in sorted(glob.glob(os.path.join(train_dir, "shift", "*.json"))):
        if js.endswith("calibration_stats.json"):
            continue
        meta = json.load(open(js))
        z = np.load(js[:-5] + ".npz", allow_pickle=False)
        if not z["done"].all():
            print(f"  skipping {meta['run']}: unfinished")
            continue
        runs[meta["run"]] = {"meta": meta, "arm": arm_of(meta), "seed": meta["seed"],
                             "conf": {str(c): z["conf"][j] for j, c in enumerate(z["conditions"])},
                             "rels": [str(r) for r in z["rels"]]}
    return runs


def shift_section(runs, n_boot, lines, res, out):
    by = {arm: {r["seed"]: r for r in runs.values() if r["arm"] == arm} for arm, _ in ARMS}
    refs = {tuple(r["rels"]) for r in runs.values()}
    if len(refs) > 1:
        sys.exit("ERROR: shift results were computed on different test subsets; re-run with the same --n_images")
    order = []
    for r in runs.values():
        order += [c for c in r["conf"] if c not in order]
    nir_arms = [a for a in ("rgbi", "rgbiaug") if by[a]]
    n_img = len(next(iter(runs.values()))["rels"])

    def conf_for(arm, seed, c):
        r = by[arm][seed]["conf"]
        return r["normal"] if arm == "rgb" else r.get(c)

    def stat(arm, c):
        v = [miou(conf_for(arm, s, c).sum(0)) for s in by[arm] if conf_for(arm, s, c) is not None]
        return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (None, None, 0)

    def delta(a1, a2, c):
        seeds = sorted(s for s in set(by[a1]) & set(by[a2])
                       if conf_for(a1, s, c) is not None and conf_for(a2, s, c) is not None)
        if not seeds:
            return None
        ca = sum(conf_for(a1, s, c) for s in seeds)
        cb = sum(conf_for(a2, s, c) for s in seeds)
        return {**paired_bootstrap_delta(ca, cb, n_boot=n_boot), "seeds": seeds}

    pairs = [(a1, a2) for a1, a2 in (("rgb", "rgbi"), ("rgb", "rgbiaug"), ("rgbi", "rgbiaug")) if by[a1] and by[a2]]
    lines += ["## 1. Does the NIR gain survive a change of NIR camera?", "",
              f"{n_img} fixed test patches from unseen domains; mean ± sd over seeds "
              f"({', '.join(f'{LABEL[a]}: {sorted(by[a])}' for a, _ in ARMS if by[a])}). "
              "Δ = paired difference, seeds pooled, 95% CI from resampling test images; * = CI excludes 0. "
              "RGB never sees NIR, so its number is the same in every row.", "",
              "| NIR condition | seen by robust model in training? | " + " | ".join(LABEL[a] for a, _ in ARMS if by[a])
              + " | " + " | ".join(f"Δ {LABEL[b]} − {LABEL[a]}" for a, b in pairs) + " |",
              "|---|---|" + "---|" * (sum(1 for a, _ in ARMS if by[a]) + len(pairs))]
    res["shift"] = {}
    for c in order:
        row = {a: stat(a, c) for a, _ in ARMS if by[a]}
        ds = {f"{b}-{a}": delta(a, b, c) for a, b in pairs}
        res["shift"][c] = {"miou": {a: {"mean": m, "sd": s, "n_seeds": n} for a, (m, s, n) in row.items()},
                           "delta": {k: v for k, v in ds.items() if v}}
        cells = [("—" if row[a][0] is None else f"{row[a][0]:.4f} ± {row[a][1]:.4f}") for a, _ in ARMS if by[a]]
        lines.append(f"| {cond_text(c)} | {seen_in_training(c)} | " + " | ".join(cells) + " | "
                     + " | ".join(ci(ds[f'{b}-{a}']) if ds[f'{b}-{a}'] else "—" for a, b in pairs) + " |")

    # trees
    key = [c for c in ("normal", "camera_b", "calib:camera_b", "gain_0.6", "gain_1.4", "missing") if c in order]
    lines += ["", "## 2. Tree classes (where the NIR reliance was found)", "",
              "IoU, seeds pooled.", "", "| Model | NIR condition | " + " | ".join(FLAIR_CLASSES[t] for t in TREES) + " |",
              "|---|---|" + "---|" * len(TREES)]
    res["trees"] = {}
    for arm, lab in ARMS:
        if not by[arm]:
            continue
        for c in (key if arm != "rgb" else ["normal"]):
            confs = [conf_for(arm, s, c) for s in by[arm] if conf_for(arm, s, c) is not None]
            if not confs:
                continue
            iou = iou_per_class(sum(confs).sum(0))
            res["trees"].setdefault(arm, {})[c] = {int(t): float(iou[t - 1]) for t in TREES}
            lines.append(f"| {lab} | {cond_text(c) if arm != 'rgb' else '(no NIR input)'} | "
                         + " | ".join(f"{iou[t - 1]:.3f}" for t in TREES) + " |")

    # gain curve
    gains = [(0.6, "gain_0.6"), (0.8, "gain_0.8"), (1.0, "normal"), (1.2, "gain_1.2"), (1.4, "gain_1.4")]
    if any(c in order for _, c in gains[:2] + gains[3:]):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        styles = {"rgb": ("#7f7f7f", "--"), "rgbi": ("#d62728", "-"), "rgbiaug": ("#1f77b4", "-")}
        for arm, lab in ARMS:
            if not by[arm]:
                continue
            xs, ys = zip(*[(g, stat(arm, c)[0]) for g, c in gains if stat(arm, c)[0] is not None])
            col, ls = styles[arm]
            ax.plot(xs, ys, ls, marker="o", color=col, label=lab)
            cx = [(g, stat(arm, "calib:" + c)[0]) for g, c in gains if arm != "rgb"
                  and "calib:" + c in order and stat(arm, "calib:" + c)[0] is not None]
            if cx:
                ax.plot(*zip(*cx), ":", marker="s", color=col, alpha=0.8, label=f"{lab} + calibration")
        ax.axvspan(0.7, 1.3, color="#cccccc", alpha=0.25, lw=0, label="robust training range")
        ax.set_xlabel("NIR gain of the test camera (1.0 = FLAIR's own camera)")
        ax.set_ylabel(f"mIoU ({n_img} test patches, mean over seeds)")
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "gain_curve.png"), dpi=160)
        plt.close(fig)
        lines += ["", "![mIoU vs NIR gain of the test camera](gain_curve.png)"]


# ------------------------------------------------------------------------------------- adapt
def load_adapt(train_dir):
    runs = {}
    for cfg_p in sorted(glob.glob(os.path.join(train_dir, "adapt", "*", "config.json"))):
        cfg = json.load(open(cfg_p))
        cells = {}
        for p in sorted(glob.glob(os.path.join(os.path.dirname(cfg_p), "cells", "*.npz"))):
            z = np.load(p, allow_pickle=False)
            cells[(str(z["domain"]), str(z["condition"]), int(z["k"]))] = {"conf": z["conf"],
                                                                          "rels": [str(r) for r in z["rels"]]}
        if cells:
            runs[cfg["run"]] = {"cfg": cfg, "arm": arm_of(cfg["source"]), "cells": cells}
    return runs


def adapt_section(runs, n_boot, lines, res, out):
    doms = sorted({d for r in runs.values() for (d, _, _) in r["cells"]})
    ks = sorted({k for r in runs.values() for (_, _, k) in r["cells"]})
    by_arm = {}
    for name, r in runs.items():
        by_arm.setdefault(r["arm"], []).append(name)
    for arm, names in by_arm.items():
        if len(names) > 1:
            print(f"  note: several adapted runs for {LABEL[arm]} ({names}); using {sorted(names)[0]}")
    use = {arm: runs[sorted(names)[0]] for arm, names in by_arm.items()}

    def stacked(arm, cond, k):
        """Per-image confusions over all domains (fixed domain order), or None if any cell is missing."""
        r = use[arm]
        c = "normal" if arm == "rgb" else cond
        parts = [r["cells"].get((d, c, k)) for d in doms]
        if any(p is None for p in parts):
            return None
        return np.concatenate([p["conf"] for p in parts])

    conds = []
    for arm in ("rgbi", "rgbiaug"):
        if arm in use:
            conds += [c for c in use[arm]["cfg"]["conditions"] if c not in conds]
    conds = conds or ["normal"]
    src = {arm: use[arm]["cfg"]["source"] for arm in use}
    lines += ["", "## 3. Few-shot adaptation to an unseen region", "",
              f"Domains: {', '.join(doms)}. K labelled patches per domain, then evaluation on other patches of the same "
              f"domain (pooled over domains). K = 0 is the trained model without adaptation. Source runs: "
              + "; ".join(f"{LABEL[a]} = {use[a]['cfg']['run']} ({src[a].get('steps')} steps)" for a, _ in ARMS if a in use)
              + f". Adaptation: {next(iter(use.values()))['cfg']['steps']} steps, "
              f"lr {next(iter(use.values()))['cfg']['lr']}. Δ vs RGB at the same K, 95% CI over evaluation images.", "",
              "| Model | condition | " + " | ".join(f"K={k}" for k in ks) + " |", "|---|---|" + "---|" * len(ks)]
    res["adapt"] = {}
    curves = {}
    for arm, lab in ARMS:
        if arm not in use:
            continue
        for cond in (conds if arm != "rgb" else ["normal"]):
            vals = []
            for k in ks:
                s = stacked(arm, cond, k)
                vals.append(None if s is None else miou(s.sum(0)))
            curves[(arm, cond)] = vals
            res["adapt"].setdefault(arm, {})[cond] = {int(k): v for k, v in zip(ks, vals)}
            lines.append(f"| {lab} | {cond_text(cond) if arm != 'rgb' else '(no NIR input)'} | "
                         + " | ".join(f4(v) for v in vals) + " |")
    if "rgb" in use:
        lines += ["", "| Δ vs RGB | condition | " + " | ".join(f"K={k}" for k in ks) + " |", "|---|---|" + "---|" * len(ks)]
        res["adapt_delta_vs_rgb"] = {}
        for arm, lab in ARMS[1:]:
            if arm not in use:
                continue
            for cond in conds:
                row = []
                for k in ks:
                    a_, b_ = stacked("rgb", cond, k), stacked(arm, cond, k)
                    if a_ is None or b_ is None or a_.shape != b_.shape:
                        row.append("—")
                        continue
                    d = paired_bootstrap_delta(a_, b_, n_boot=n_boot)
                    res["adapt_delta_vs_rgb"].setdefault(arm, {}).setdefault(cond, {})[int(k)] = d
                    row.append(ci(d))
                lines.append(f"| {lab} | {cond_text(cond)} | " + " | ".join(row) + " |")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    colours = {"rgb": "#7f7f7f", "rgbi": "#d62728", "rgbiaug": "#1f77b4"}
    dashes = {"normal": "-", "camera_b": "--", "calib:camera_b": ":"}
    xs = [max(k, 0.5) for k in ks]
    for (arm, cond), vals in curves.items():
        pts = [(x, v) for x, v in zip(xs, vals) if v is not None]
        if pts:
            ax.plot(*zip(*pts), dashes.get(cond, "-."), marker="o", color=colours[arm],
                    label=f"{LABEL[arm]}" + ("" if arm == "rgb" else f", {cond_text(cond)}"))
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("labelled patches from the new region (K)")
    ax.set_ylabel("mIoU on the new region")
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "kshot_curve.png"), dpi=160)
    plt.close(fig)
    lines += ["", "![mIoU vs number of labelled patches from the new region](kshot_curve.png)"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", default="outputs/train")
    ap.add_argument("--out", help="default <train_dir>/robust_report")
    ap.add_argument("--n_boot", type=int, default=1000)
    a = ap.parse_args()
    out = a.out or os.path.join(a.train_dir, "robust_report")
    os.makedirs(out, exist_ok=True)
    shift, adapt = load_shift(a.train_dir), load_adapt(a.train_dir)
    if not shift and not adapt:
        sys.exit(f"ERROR: no finished results under {a.train_dir}/shift or {a.train_dir}/adapt")
    lines = ["# NIR robustness and few-shot adaptation", ""]
    tiny = [n for n, r in shift.items() if not r["meta"].get("is_measurement", True)]
    tiny += [n for n, r in adapt.items() if r["cfg"]["source"].get("model") == "tiny"]
    if tiny:
        lines += [f"**WARNING: {', '.join(sorted(set(tiny)))} use the `tiny` pipeline stand-in. These numbers are NOT results.**", ""]
    res = {}
    if shift:
        shift_section(shift, a.n_boot, lines, res, out)
    if adapt:
        adapt_section(adapt, a.n_boot, lines, res, out)
    lines += ["", "## Reading guide", "",
              "- Simulated cameras change only the NIR band. gain_0.6 / gain_1.4 lie outside the range the robust "
              "model was trained on (gain 0.7–1.3); judge robustness on those rows and on calibration, not only on rows "
              "the robust model has seen.",
              "- Calibration rescales each domain's NIR to the mean and std of the FLAIR training set using unlabelled "
              "patches. It undoes a pure gain/offset change almost exactly by construction; the informative rows are "
              "gamma / camera B and the cost (or gain) on real NIR (`real NIR + calibration`).",
              "- A simulated camera is not a real drone camera. The EU UAV data remains the real test."]
    with open(os.path.join(out, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
