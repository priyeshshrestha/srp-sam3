"""Report for the RGB vs RGB+NIR training experiment.

    python scripts/train_report.py --runs outputs/train/runs/* --root ... --split ... --band_order rgb \
        --out outputs/train/report

The headline is the PAIRED difference RGBI - RGB (same seeds, same test images), with a 95% CI from
resampling test images, plus the spread over seeds. Untrained SAM 3 (zero-shot, text prompts) is
the reference that shows what training itself buys.

Figures: 10 test images chosen by a fixed rule on the first paired seed (manifest.json says why):
4 where NIR helped most, 2 where it hurt most, 2 where both models do well, 2 where both fail.
They are re-predicted from the saved checkpoints, so this step needs the GPU and the SAM 3 weights.
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402

from datasets.flair import FLAIR_CLASSES  # noqa: E402
from train.metrics import iou_per_class, miou, paired_bootstrap_delta, summary  # noqa: E402
from train.sam3_rgbi import CLASSES  # noqa: E402

# Official FLAIR-HUB label colours (github.com/IGNF/FLAIR-HUB README), keyed by FLAIR #1 id.
CLASS_HEX = {1: "#db0e9a", 2: "#938e7b", 3: "#f80c00", 4: "#a97101", 5: "#1553ae", 6: "#194a26", 7: "#46e483",
             8: "#f3a60d", 9: "#660082", 10: "#55ff00", 11: "#fff30d", 12: "#e4df7c", 13: "#3de6eb",
             14: "#ffffff", 15: "#8ab3a0", 16: "#6b714f", 17: "#c5dc42", 18: "#9999ff", 19: "#000000"}


def load(run_dir):
    with open(os.path.join(run_dir, "eval.json")) as fh:
        meta = json.load(fh)
    z = np.load(os.path.join(run_dir, "eval.npz"), allow_pickle=False)
    return {"dir": run_dir, "meta": meta, "conf": z["conf"], "rels": [str(r) for r in z["rels"]],
            "split_sha": str(z["split_sha"]), "name": os.path.basename(os.path.normpath(run_dir))}


# A figure patch must show a SCENE: >= MIN_CLASSES classes each covering >= MIN_CLASS_FRAC of the
# evaluated pixels. Without this the rule picks single-class patches, whose per-image mIoU is
# all-or-nothing (0.000 or 1.000) and says nothing about the models.
MIN_CLASSES, MIN_CLASS_FRAC = 3, 0.05


def eligible_for_figures(conf):
    """(N,) bool from per-image confusions (N, K, K): multi-class scenes only."""
    gt = conf.sum(2).astype(np.float64)                          # (N, K) ground-truth pixels per class
    frac = gt / np.maximum(gt.sum(1, keepdims=True), 1)
    return (frac >= MIN_CLASS_FRAC).sum(1) >= MIN_CLASSES


def training_setup(split_path, runs):
    """Facts about the experiment (data sizes, schedule) for downstream use, e.g. the deck."""
    from train.data import load_split
    sp = load_split(split_path)
    out = {"split_mode": sp["mode"], "n_train": len(sp["train"]), "n_train_domains": len(sp["train_domains"]),
           "n_test": len(sp["test"]), "n_test_domains": len(sp["test_domains"])}
    for r in runs:
        cfg_p = os.path.join(r["dir"], "config.json")
        if not r["meta"]["zero_shot"] and os.path.exists(cfg_p):
            cfg = json.load(open(cfg_p))
            out.update({k: cfg.get(k) for k in ("steps", "effective_batch", "lr", "prompts", "lora_rank", "model")})
            break
    return out


def select_figures(m_rgb, m_rgbi, n=10, eligible=None):
    """Fixed rule on per-image mIoU of the RGB and RGBI models (same seed), among eligible images."""
    d = m_rgbi - m_rgb
    ok = np.ones(len(d), bool) if eligible is None else np.asarray(eligible, bool)
    picks, used = [], set()

    def take(order, k, why):
        for i in order:
            if len(picks) >= n or k <= 0:
                return
            if i not in used and ok[i] and np.isfinite(d[i]):
                used.add(int(i))
                picks.append((int(i), why))
                k -= 1

    take(np.argsort(-d), 4, "NIR helped most (largest RGBI - RGB)")
    take(np.argsort(d), 2, "NIR hurt most (largest RGB - RGBI)")
    take(np.argsort(-np.minimum(m_rgb, m_rgbi)), 2, "both models good (highest min mIoU)")
    take(np.argsort(np.maximum(m_rgb, m_rgbi)), 2, "both models fail (lowest max mIoU)")
    take(np.argsort(-np.abs(d)), n - len(picks), "filler: next largest |difference|")
    return picks


def colourise(lbl):
    out = np.zeros(lbl.shape + (3,), dtype=np.float32)
    for c, h in CLASS_HEX.items():
        out[lbl == c] = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--band_order", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_figures", type=int, default=10)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    runs = []
    for d in a.runs:
        if os.path.exists(os.path.join(d, "eval.json")) and os.path.exists(os.path.join(d, "eval.npz")):
            runs.append(load(d))
        else:
            print(f"  skipping {d}: no finished evaluation")
    if not runs:
        sys.exit("ERROR: no finished evaluations to report")
    shas = {r["split_sha"] for r in runs}
    if len(shas) > 1:
        sys.exit(f"ERROR: runs come from different splits {sorted(shas)}; refusing to mix them")
    ref = runs[0]["rels"]
    for r in runs:
        if r["rels"] != ref:
            sys.exit(f"ERROR: {r['dir']} was evaluated on different test images (TEST_LIMIT changed?)")
    mock = [r["name"] for r in runs if not r["meta"].get("is_measurement", True)]

    zs = [r for r in runs if r["meta"]["zero_shot"]]
    # NIR-robust runs (nir_aug) belong to the robustness report (scripts/robust_report.py), not here
    skipped = [r["name"] for r in runs if r["meta"].get("nir_aug")]
    if skipped:
        print(f"  not in this report (NIR-robust runs, see robust_report.py): {' '.join(skipped)}")
    by = {ch: {r["meta"]["seed"]: r for r in runs if not r["meta"]["zero_shot"] and r["meta"]["channels"] == ch
               and not r["meta"].get("nir_aug")} for ch in ("rgb", "rgbi")}
    seeds = sorted(set(by["rgb"]) & set(by["rgbi"]))

    def cond_stats(rs):
        ms = [miou(r["conf"].sum(0)) for r in rs]
        return (float(np.mean(ms)), float(np.std(ms)), ms) if ms else (float("nan"), float("nan"), [])

    lines = ["# SAM 3 fine-tuned on FLAIR: RGB vs RGB+NIR", ""]
    if mock:
        lines += [f"**WARNING: {', '.join(mock)} used the `tiny` stand-in model. These numbers test the pipeline "
                  f"and are NOT results.**", ""]
    lines += [f"Test set: {len(ref)} patches from held-out domains (split sha `{runs[0]['split_sha']}`). "
              f"mIoU over FLAIR classes 1-12 (pixels of classes 13-19 ignored), confusion accumulated over "
              f"all test images. Prediction = argmax over the 12 class prompts.", "",
              "| Model | Training | Seeds | mIoU (mean ± sd over seeds) | per seed |", "|---|---|---|---|---|"]
    for r in zs:
        lines.append(f"| SAM 3 zero-shot ({r['meta']['channels']}) | none | - | {miou(r['conf'].sum(0)):.4f} | - |")
    for ch, label in (("rgb", "RGB"), ("rgbi", "RGB+NIR")):
        rs = [by[ch][s] for s in sorted(by[ch])]
        mean, sd, ms = cond_stats(rs)
        if rs:
            lines.append(f"| SAM 3 + LoRA, {label} | fine-tuned | {len(rs)} | **{mean:.4f}** ± {sd:.4f} | "
                         f"{', '.join(f'{m:.4f}' for m in ms)} |")
    res = {"n_test": len(ref), "split_sha": runs[0]["split_sha"], "seeds_paired": seeds,
           "runs": {r["name"]: summary(r["conf"].sum(0)) for r in runs}, "setup": training_setup(a.split, runs)}

    if seeds:
        a_conf = sum(by["rgb"][s]["conf"] for s in seeds)
        b_conf = sum(by["rgbi"][s]["conf"] for s in seeds)
        bs = paired_bootstrap_delta(a_conf, b_conf, n_boot=a.n_boot)
        per_seed = [miou(by["rgbi"][s]["conf"].sum(0)) - miou(by["rgb"][s]["conf"].sum(0)) for s in seeds]
        res["delta_rgbi_minus_rgb"] = {**bs, "per_seed": per_seed}
        lines += ["", "## Does NIR help? (paired: same seeds, same test images)", "",
                  f"**mIoU(RGB+NIR) − mIoU(RGB) = {bs['delta']:+.4f}**, 95% CI [{bs['ci95'][0]:+.4f}, {bs['ci95'][1]:+.4f}] "
                  f"({'significant' if bs['significant'] else 'NOT significant'}; bootstrap over {bs['n_images']} test "
                  f"images, {len(seeds)} seed(s) pooled). Per seed: {', '.join(f'{d:+.4f}' for d in per_seed)}.", ""]
        if len(seeds) < 3:
            lines.append("_Fewer than 3 paired seeds: the seed-to-seed spread is not yet known._")

        # Per-class
        iou = {k: np.nanmean([iou_per_class(by[k][s]["conf"].sum(0)) for s in seeds], axis=0) for k in ("rgb", "rgbi")}
        iou_zs = iou_per_class(zs[0]["conf"].sum(0)) if zs else np.full(len(CLASSES), np.nan)
        gt_px = a_conf.sum((0, 2))
        lines += ["", "## Per class (IoU, mean over paired seeds)", "",
                  "Δ and its 95% CI are computed on the confusions pooled over seeds, with the same image resampling "
                  "as the headline; * = CI excludes 0.", "",
                  ("| Class | zero-shot | RGB | RGB+NIR | Δ | 95% CI of Δ | test pixels |" if zs else
                   "| Class | RGB | RGB+NIR | Δ | 95% CI of Δ | test pixels |"),
                  "|---|---|---|---|---|---|" + ("---|" if zs else "")]
        rows = []
        for j, c in enumerate(CLASSES):
            pc = bs["per_class"][int(c)]
            d = pc["delta"] if pc["delta"] is not None else float("nan")
            star = " *" if pc["significant"] else ""
            rows.append([c, FLAIR_CLASSES[c], iou_zs[j], iou["rgb"][j], iou["rgbi"][j], d, pc["ci95"][0], pc["ci95"][1],
                         int(pc["significant"]), int(gt_px[j])])
            lines.append(f"| {FLAIR_CLASSES[c]} | " + (f"{iou_zs[j]:.3f} | " if zs else "") +
                         f"{iou['rgb'][j]:.3f} | {iou['rgbi'][j]:.3f} | "
                         f"{d:+.3f}{star} | [{pc['ci95'][0]:+.3f}, {pc['ci95'][1]:+.3f}] | {int(gt_px[j]):,} |")
        with open(os.path.join(a.out, "per_class.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["class_id", "class", "zero_shot", "rgb", "rgbi", "delta_pooled", "ci95_lo", "ci95_hi",
                        "significant", "test_pixels"])
            w.writerows(rows)
        _per_class_chart(rows, os.path.join(a.out, "per_class_iou.png"))
        if a.n_figures:
            _figures(a, by["rgb"][seeds[0]], by["rgbi"][seeds[0]], zs[0] if zs else None, ref, lines)
    else:
        lines += ["", "_No seed has both an RGB and an RGB+NIR run finished: no paired comparison yet._"]

    with open(os.path.join(a.out, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("\n".join(lines))


def _per_class_chart(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(11, 4))
    series = [(2, "SAM 3 zero-shot", "#c4c4c4"), (3, "fine-tuned RGB", "#7d7d7d"), (4, "fine-tuned RGB+NIR", "#2a4d38")]
    series = [s_ for s_ in series if not all(np.isnan(r[s_[0]]) for r in rows)]       # no zero-shot run -> no bars
    w = 0.8 / len(series)
    for k, (col, label, colour) in enumerate(series):
        ax.bar(x + (k - (len(series) - 1) / 2) * w, [np.nan_to_num(r[col]) for r in rows], w, label=label, color=colour)
    ax.set_xticks(x)
    ax.set_xticklabels([r[1].replace("_", " ") for r in rows], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("IoU")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _figures(a, run_rgb, run_rgbi, run_zs, rels, lines):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from scripts.eval_sam3 import load_net
    from scripts.train_sam3 import pick_device
    from train.data import load_rgbi
    from train.sam3_rgbi import normalise, predict_classes

    m_rgb = np.array([miou(c) for c in run_rgb["conf"]])
    m_rgbi = np.array([miou(c) for c in run_rgbi["conf"]])
    eligible = eligible_for_figures(run_rgb["conf"])
    picks = select_figures(m_rgb, m_rgbi, a.n_figures, eligible)
    dev = pick_device(a.device)
    panels = ([("SAM 3 zero-shot", run_zs)] if run_zs else []) + [("fine-tuned RGB", run_rgb),
                                                                 ("fine-tuned RGB+NIR", run_rgbi)]
    data = [load_rgbi(a.root, rels[i], a.band_order) for i, _ in picks]
    preds = {}
    for name, run in panels:
        m = run["meta"]
        net = load_net(None if m["zero_shot"] else run["dir"], m["model"], m["channels"], 16, m["zero_shot"], dev)
        n_ch = 4 if m["channels"] == "rgbi" else 3
        with torch.inference_mode():
            for (i, _), (x, y) in zip(picks, data):
                xt = normalise(torch.from_numpy(x[:n_ch]).unsqueeze(0).to(dev))
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                    preds[(name, i)] = predict_classes(net, xt, out_size=y.shape)[0].cpu().numpy()
        del net
        if dev == "cuda":
            torch.cuda.empty_cache()

    fig_dir = os.path.join(a.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    for old in os.listdir(fig_dir):                       # never mix figures from an earlier selection
        if old.endswith(".png") or old == "manifest.json":
            os.remove(os.path.join(fig_dir, old))
    manifest = []
    for n, ((i, why), (x, y)) in enumerate(zip(picks, data), 1):
        cols = 3 + len(panels)
        fig, ax = plt.subplots(1, cols, figsize=(3.0 * cols, 3.6))
        rgb = np.transpose(x[:3], (1, 2, 0))
        cir = np.transpose(x[[3, 0, 1]], (1, 2, 0))
        ax[0].imshow(rgb); ax[0].set_title("RGB", fontsize=9)
        ax[1].imshow(cir); ax[1].set_title("NIR-R-G (false colour)", fontsize=9)
        shown = np.where((y >= 1) & (y <= 12), y, 0)
        ax[2].imshow(colourise(shown)); ax[2].set_title("Ground truth", fontsize=9)
        for k, (name, run) in enumerate(panels):
            p = np.where(shown > 0, preds[(name, i)], 0)
            ax[3 + k].imshow(colourise(p))
            ax[3 + k].set_title(f"{name}\nmIoU {miou(run['conf'][i]):.3f}", fontsize=9, fontweight="bold")
        for axx in ax:
            axx.axis("off")
        present = [c for c in CLASSES if (y == c).any()]
        handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_HEX[c]) for c in present]
        fig.legend(handles, [FLAIR_CLASSES[c].replace("_", " ") for c in present], loc="lower center",
                   ncol=min(6, len(present)), fontsize=8, frameon=False)
        fig.suptitle(f"{os.path.basename(rels[i])}  —  {why}", fontsize=9)
        fig.tight_layout(rect=(0, 0.08, 1, 0.95))
        fname = f"{n:02d}_{os.path.splitext(os.path.basename(rels[i]))[0]}.png"
        fig.savefig(os.path.join(fig_dir, fname), dpi=130)
        plt.close(fig)
        manifest.append({"file": fname, "patch": rels[i], "why": why, "miou_rgb": float(m_rgb[i]),
                         "miou_rgbi": float(m_rgbi[i]), "seed": run_rgb["meta"]["seed"]})
    with open(os.path.join(fig_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    lines += ["", f"## Figures ({len(picks)}, chosen by rule, seed {run_rgb['meta']['seed']}; see figures/manifest.json)", "",
              f"Eligible: test patches with >= {MIN_CLASSES} classes each covering >= {MIN_CLASS_FRAC:.0%} of the patch "
              f"({int(eligible.sum())} of {len(eligible)}). Per-image mIoU = mean IoU over the classes in that patch.", ""]
    lines += [f"- `{m['file']}`: {m['why']} (RGB {m['miou_rgb']:.3f}, RGB+NIR {m['miou_rgbi']:.3f})" for m in manifest]


if __name__ == "__main__":
    main()
