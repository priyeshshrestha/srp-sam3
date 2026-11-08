"""Tables, statistics and figures from saved runs. Needs no GPU and no model.

    python scripts/report.py --root ./data --episodes outputs/episodes.json \
        --runs outputs/runs/sam1_vit_b outputs/runs/sam2_small outputs/runs/sam3_facebook-sam3 \
        --out outputs/report

Writes: summary.md, per_class.csv, per_class_iou.png, and figures/NN_<class>_ep<id>.png plus
figures/manifest.json (why each episode was chosen). Baselines 'all_fg' (predict the whole patch
as the class) and 'empty' are computed from the ground truth and listed beside the models: a
model that does not beat all_fg on a class has learned nothing about that class.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.flair import FLAIR_CLASSES  # noqa: E402
from datasets.flair_rgb import load_patch  # noqa: E402
from evaluation.episodes import episodes_hash, load_episodes  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    baseline_record, class_miou, episode_iou, fb_iou, paired_bootstrap,
)
from evaluation.report_lib import load_pred, load_run, select_figure_episodes  # noqa: E402


def overlay(rgb, mask, colour, alpha=0.55):
    out = rgb.astype(np.float32) / 255.0
    out[mask] = (1 - alpha) * out[mask] + alpha * np.array(colour, dtype=np.float32)
    return out


def draw_episode(path, ep, s_rgb, s_bin, q_rgb, gt, runs, title_note):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = 3 + len(runs)
    fig, ax = plt.subplots(1, n, figsize=(3.3 * n, 3.9))
    cname = FLAIR_CLASSES.get(ep["class_id"], str(ep["class_id"]))
    ax[0].imshow(overlay(s_rgb, s_bin, (1, 0, 1)))
    ax[0].set_title(f"Support ({ep['support_domain']})\n{cname} in magenta", fontsize=9)
    ax[1].imshow(q_rgb)
    ax[1].set_title(f"Query ({ep['query_domain']})", fontsize=9)
    ax[2].imshow(overlay(q_rgb, gt, (1, 0, 1)))
    ax[2].set_title(f"Ground truth (magenta): {gt.mean()*100:.0f}% of patch\n(all-fg baseline IoU {gt.mean():.2f})", fontsize=9)
    for k, run in enumerate(runs):
        pred = load_pred(run["dir"], ep["id"])
        iou = episode_iou(run["records"][ep["id"]])
        img = q_rgb.astype(np.float32) / 255.0 * 0.45
        img[pred & gt] = [0.0, 0.85, 0.0]      # true positive
        img[pred & ~gt] = [0.9, 0.0, 0.0]      # false positive
        img[~pred & gt] = [0.0, 0.6, 1.0]      # false negative
        ax[3 + k].imshow(img)
        ax[3 + k].set_title(f"{run['name']}\nIoU {iou:.3f}", fontsize=9, fontweight="bold")
    for a in ax:
        a.axis("off")
    fig.suptitle(f"episode {ep['id']} - {title_note}   [green=TP  red=FP  blue=FN]", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="FLAIR root (only needed for figures)")
    ap.add_argument("--episodes", default="outputs/episodes.json")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="outputs/report")
    ap.add_argument("--n_figures", type=int, default=10)
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()

    ep_file = load_episodes(args.episodes)
    episodes = {e["id"]: e for e in ep_file["episodes"]}
    band_order = ep_file["meta"]["band_order"]
    runs = [load_run(r) for r in args.runs]
    sha = episodes_hash(ep_file["episodes"])
    for r in runs:
        got = r["meta"].get("episodes_sha")
        if got != sha:
            sys.exit(f"ERROR: run {r['name']} was computed on a different episode file "
                     f"({got} vs {sha}); refusing to compare. Delete it and re-run.")
    not_measurements = [r["name"] for r in runs if r["meta"].get("is_measurement") is False]
    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)

    # Compare on the episodes EVERY model finished, so the comparison is genuinely paired.
    common = sorted(set.intersection(*(set(r["records"]) for r in runs)))
    if not common:
        sys.exit("No episode was completed by all runs.")
    print(f"{len(common)} episodes completed by all {len(runs)} runs")
    for r in runs:
        r["common"] = {i: r["records"][i] for i in common}

    # Baselines from ground truth.
    gt_cache = {}
    def gt_of(i):
        if i not in gt_cache:
            _, m = load_patch(args.root, episodes[i]["query"], band_order)
            gt_cache[i] = m == episodes[i]["class_id"]
        return gt_cache[i]
    baselines = []
    for kind in ("all_fg", "empty"):
        recs = {}
        for i in common:
            recs[i] = {"id": i, "class_id": episodes[i]["class_id"], **baseline_record(gt_of(i), kind)}
        baselines.append({"name": f"baseline:{kind}", "common": recs})

    everyone = runs + baselines
    classes = sorted({episodes[i]["class_id"] for i in common})

    # ---- summary table
    rows = []
    for r in everyone:
        recs = list(r["common"].values())
        acc, per_acc = class_miou(recs, True)
        epi, per_epi = class_miou(recs, False)
        secs = [x.get("seconds") for x in r["common"].values() if x.get("seconds") is not None]
        pa = [x["pred_area"] for x in r["common"].values() if "pred_area" in x]
        empty = float(np.mean([a == 0 for a in pa])) if pa else None
        rows.append({"name": r["name"], "miou_acc": acc, "miou_episode": epi, "fb_iou": fb_iou(recs),
                     "empty_frac": empty,
                     "sec_per_ep": float(np.mean(secs)) if secs else None,
                     "per_acc": per_acc, "per_epi": per_epi})

    lines = ["# SAM 1 / SAM 2 / SAM 3 on FLAIR (RGB only, one-shot, training-free)", ""]
    if not_measurements:
        lines += [f"> **WARNING: {', '.join(not_measurements)} is a pipeline-test stand-in, NOT a model "
                  f"measurement. Do not cite these numbers.**", ""]
    if ep_file["meta"].get("domain_mode") == "same":
        lines += ["> **Same-domain episodes** (support and query from the same FLAIR domain): numbers are "
                  "optimistic and say nothing about geographic/seasonal transfer.", ""]
    lines += [
             f"- episodes compared: **{len(common)}** (completed by every run), "
             f"{len(classes)} classes, band order `{band_order}`",
             f"- episodes file: `{args.episodes}`; domain mode `{ep_file['meta'].get('domain_mode')}`, "
             f"coverage filter {ep_file['meta'].get('min_frac')}-{ep_file['meta'].get('max_frac')}, "
             f"seed {ep_file['meta'].get('seed')}", "",
             "| run | mIoU (class-accumulated) | mIoU (mean of episodes) | FB-IoU | empty predictions | s/episode |",
             "|---|---|---|---|---|---|"]
    for row in sorted(rows, key=lambda x: -x["miou_acc"]):
        sec = "-" if row["sec_per_ep"] is None else f"{row['sec_per_ep']:.2f}"
        emp = "-" if row["empty_frac"] is None else f"{row['empty_frac']*100:.0f}%"
        lines.append(f"| {row['name']} | {row['miou_acc']:.4f} | {row['miou_episode']:.4f} | "
                     f"{row['fb_iou']:.4f} | {emp} | {sec} |")

    # ---- paired comparisons
    lines += ["", "## Paired bootstrap on mIoU difference (episodes resampled within class)", "",
              "| A | B | A - B | 95% CI | significant |", "|---|---|---|---|---|"]
    for a in range(len(runs)):
        for b in range(a + 1, len(runs)):
            res = paired_bootstrap(runs[a]["common"], runs[b]["common"], n_boot=args.n_boot)
            lines.append(f"| {runs[a]['name']} | {runs[b]['name']} | {res['diff']:+.4f} | "
                         f"[{res['ci_lo']:+.4f}, {res['ci_hi']:+.4f}] | {'yes' if res['significant'] else 'no'} |")

    # ---- per class
    lines += ["", "## Per-class IoU (class-accumulated)", "",
              "| class | n | " + " | ".join(r["name"] for r in everyone) + " |",
              "|---|---|" + "---|" * len(everyone)]
    n_by_class = {c: sum(1 for i in common if episodes[i]["class_id"] == c) for c in classes}
    for c in classes:
        cells = [f"{row['per_acc'].get(c, float('nan')):.3f}" for row in rows]
        lines.append(f"| {FLAIR_CLASSES.get(c, c)} | {n_by_class[c]} | " + " | ".join(cells) + " |")

    # ---- SAM 3 box fallback diagnostic
    for r in runs:
        fb = [x["diag"].get("box_fallback") for x in r["common"].values() if "box_fallback" in x.get("diag", {})]
        if fb:
            lines += ["", f"_{r['name']}: {sum(fb)}/{len(fb)} episodes used the whole-class box fallback "
                          f"(no usable connected component)._"]

    with open(os.path.join(args.out, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out, "per_class.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class_id", "class", "n_episodes"] + [r["name"] for r in everyone])
        for c in classes:
            w.writerow([c, FLAIR_CLASSES.get(c, c), n_by_class[c]] +
                       [round(row["per_acc"].get(c, float("nan")), 4) for row in rows])
    print("\n".join(lines[:14]))

    # ---- per-class bar chart
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(1.4 * len(classes) + 4, 5))
    width = 0.8 / len(everyone)
    for k, row in enumerate(rows):
        ax.bar(np.arange(len(classes)) + k * width,
               [row["per_acc"].get(c, 0) for c in classes], width, label=row["name"])
    ax.set_xticks(np.arange(len(classes)) + 0.4 - width / 2)
    ax.set_xticklabels([FLAIR_CLASSES.get(c, c) for c in classes], rotation=35, ha="right")
    ax.set_ylabel("IoU (class-accumulated)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "per_class_iou.png"), dpi=150)
    plt.close(fig)

    # ---- qualitative figures, episodes chosen by rule
    ious = {i: {r["name"]: episode_iou(r["common"][i]) for r in runs} for i in common}
    class_of = {i: episodes[i]["class_id"] for i in common}
    gt_frac = {i: float(gt_of(i).mean()) for i in common}
    picked = select_figure_episodes(ious, class_of, gt_frac, n_total=args.n_figures)
    manifest = []
    for n, p in enumerate(picked, 1):
        ep = episodes[p["id"]]
        s_rgb, s_msk = load_patch(args.root, ep["support"], band_order)
        q_rgb, _ = load_patch(args.root, ep["query"], band_order)
        cname = FLAIR_CLASSES.get(ep["class_id"], str(ep["class_id"]))
        fname = f"{n:02d}_{cname}_ep{ep['id']}.png"
        draw_episode(os.path.join(args.out, "figures", fname), ep, s_rgb, s_msk == ep["class_id"],
                     q_rgb, gt_of(p["id"]), runs, p["reason"])
        manifest.append({"file": fname, **p, "class": cname, "ious": ious[p["id"]]})
    with open(os.path.join(args.out, "figures", "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"\nWrote {len(manifest)} figures and summary to {args.out}")


if __name__ == "__main__":
    main()
