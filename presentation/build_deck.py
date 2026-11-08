"""Build the final-phase deck from REAL results, in the style of the previous presentation.

Every result number, chart and prediction image is computed here from finished runs
(outputs/runs/<model>_<variant>_<arm>/) and the NDVI analysis (outputs/ndvi/). Nothing is typed in
by hand except the recap numbers from the previous presentation and cited literature facts. A
model or arm that has not been run simply does not appear, and the deck says what is pending.

    python presentation/build_deck.py --root ./data --out presentation/out/SRP_Group7_Final.pptx
"""
import argparse
import glob
import json
import os
import re
import sys
import textwrap

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from lxml import etree  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE  # noqa: E402
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN  # noqa: E402
from pptx.oxml.ns import qn  # noqa: E402
from pptx.util import Inches, Pt  # noqa: E402
from scipy import ndimage  # noqa: E402

from datasets.flair import FLAIR_CLASSES, mask_path_for  # noqa: E402
from datasets.flair_rgb import load_patch, read_label  # noqa: E402
from evaluation.episodes import episodes_hash, load_episodes  # noqa: E402
from evaluation.metrics import class_miou, episode_iou, fb_iou, paired_bootstrap  # noqa: E402
from evaluation.report_lib import load_pred, load_run, select_figure_episodes  # noqa: E402

# ----------------------------------------------------------------------------- style (measured
# from SRP_Group7_Presentation.pdf: Helvetica Neue, 24 pt titles, #1a1a1a / #666666 text, grey
# #b0b0b0 rule at 0.95 in, #f2f2f2 boxes with #555555 outline, #2a4d38 accent, #e5e5e5 tables)
INK, GREY, RULE, BOX, BOXLINE, ACCENT, TABLE = "1A1A1A", "666666", "B0B0B0", "F2F2F2", "555555", "2A4D38", "E5E5E5"
FONT = "Helvetica Neue"
FOOTER = "Group 7 · Few-Shot Learning for Biodiversity Assessment"
LOGO = os.path.join(HERE, "assets", "uni_hildesheim_logo.png")
W_IN, H_IN = 10.0, 5.625

MODEL_ORDER = ["sam1", "sam2", "sam3"]
MODEL_NAME = {"sam1": "SAM 1", "sam2": "SAM 2.1", "sam3": "SAM 3"}
MODEL_VARIANT = {"vit_b": "ViT-B", "vit_l": "ViT-L", "vit_h": "ViT-H", "tiny": "Hiera-T", "small": "Hiera-S",
                 "base_plus": "Hiera-B+", "large": "Hiera-L", "facebook-sam3": ""}
MODEL_COLOUR = {"sam1": "#c4c4c4", "sam2": "#7d7d7d", "sam3": "#2a4d38"}
CLASS_SHORT = {1: "Building", 2: "Pervious surf.", 3: "Impervious surf.", 4: "Bare soil", 5: "Water",
               6: "Coniferous", 7: "Deciduous", 8: "Brushwood", 9: "Vineyard", 10: "Herbaceous",
               11: "Agricultural", 12: "Plowed land"}
VEG = (6, 7, 8, 9, 10, 11, 12)
BUILT = (1, 2, 3, 4)
# Official FLAIR-HUB label colours (github.com/IGNF/FLAIR-HUB README), keyed by FLAIR #1 id.
CLASS_HEX = {1: "#db0e9a", 2: "#938e7b", 3: "#f80c00", 4: "#a97101", 5: "#1553ae", 6: "#194a26", 7: "#46e483",
             8: "#f3a60d", 9: "#660082", 10: "#55ff00", 11: "#fff30d", 12: "#e4df7c", 13: "#3de6eb",
             14: "#ffffff", 15: "#8ab3a0", 16: "#6b714f", 17: "#c5dc42", 18: "#9999ff", 19: "#000000"}

for fam in ("Helvetica Neue", "Helvetica", "Arial"):
    try:
        font_manager.findfont(fam, fallback_to_default=False)
        plt.rcParams["font.family"] = [fam, "DejaVu Sans"]      # DejaVu supplies arrows and maths glyphs
        break
    except Exception:
        continue
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#b0b0b0", "axes.labelcolor": "#1a1a1a",
                     "xtick.color": "#1a1a1a", "ytick.color": "#1a1a1a", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e5e5e5",
                     "grid.linewidth": 0.7, "axes.axisbelow": True, "savefig.dpi": 300,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.03})


# ============================================================================ results

class Results:
    """Everything the deck says about experiments, computed from the saved runs."""

    def __init__(self, root, episodes_path, runs_dir, ndvi_dir):
        self.root = root
        ep = load_episodes(episodes_path)
        self.meta = ep["meta"]
        self.episodes = {e["id"]: e for e in ep["episodes"]}
        self.sha = episodes_hash(ep["episodes"])
        self.band_order = self.meta["band_order"]
        self.runs, self.incomplete = {}, []
        for d in sorted(glob.glob(os.path.join(runs_dir, "*"))):
            if not os.path.exists(os.path.join(d, "results.jsonl")):
                continue
            r = load_run(d)
            if r["meta"].get("episodes_sha") != self.sha or r["meta"].get("is_measurement") is False:
                continue                                 # other episode file, or the mock model
            m = r["meta"]["model"]
            arm = r["meta"].get("arm") or ("1shot" if d.endswith("1shot") else None)
            if m not in MODEL_ORDER or not arm or not r["records"]:
                continue
            planned = r["meta"].get("n_episodes", len(r["records"]))
            if len(r["records"]) < planned:              # still running or interrupted: never show partial runs
                self.incomplete.append(f"{MODEL_NAME[m]} {arm} ({len(r['records'])}/{planned})")
                continue
            self.runs[(m, arm)] = r
        self.models = [m for m in MODEL_ORDER if (m, "1shot") in self.runs]
        self.ndvi = {}
        for f in glob.glob(os.path.join(ndvi_dir, "*.json")):
            d = json.load(open(f))
            m = d["run"].split("_")[0]
            self.ndvi[m] = d

    def has(self, model, arm):
        return (model, arm) in self.runs

    def label(self, model):
        r = next(v for (m, a), v in self.runs.items() if m == model)
        var = MODEL_VARIANT.get(r["meta"]["variant"].replace("/", "-"), r["meta"]["variant"])
        return f"{MODEL_NAME[model]} {var}".strip()

    def recs(self, model, arm, ids=None):
        rr = self.runs[(model, arm)]["records"]
        return [rr[i] for i in (ids if ids is not None else sorted(rr)) if i in rr]

    def common(self, keys):
        sets = [set(self.runs[k]["records"]) for k in keys if k in self.runs]
        return sorted(set.intersection(*sets)) if sets else []

    def miou(self, model, arm, ids=None):
        return class_miou(self.recs(model, arm, ids))

    def baseline(self, ids):
        """predict-all: IoU = GT area / patch area; exact from the stored confusion counts."""
        ref = next(iter(self.runs.values()))["records"]
        out = []
        for i in ids:
            r = ref[i]
            total = r["fg_union"] + r["bg_inter"]
            out.append({"class_id": r["class_id"], "fg_inter": r["gt_area"], "fg_union": total,
                        "bg_inter": 0, "bg_union": total - r["gt_area"]})
        return out

    def empty_rate(self, model, arm):
        rs = self.recs(model, arm)
        return float(np.mean([r["pred_area"] == 0 for r in rs]))

    def seconds(self, model, arm):
        return float(np.median([r["seconds"] for r in self.recs(model, arm)]))

    def n(self, model, arm):
        return len(self.runs[(model, arm)]["records"])

    def unique_queries(self):
        return len({e["query"] for e in self.episodes.values()})


class TrainResults:
    """The training experiment (SAM 3 fine-tuned, RGB vs RGB+NIR), from run_train.sh's report
    (outputs/train/report: summary.json, per_class.csv, figures/). Missing report -> .ok is False."""

    def __init__(self, report_dir, train_dir):
        self.dir = report_dir
        p = os.path.join(report_dir, "summary.json")
        self.ok = os.path.exists(p) and "delta_rgbi_minus_rgb" in json.load(open(p))
        if not self.ok:
            return
        import csv
        self.s = json.load(open(p))
        self.d = self.s["delta_rgbi_minus_rgb"]
        with open(os.path.join(report_dir, "per_class.csv"), newline="") as fh:
            self.per_class = [{k: (v if k == "class" else float(v)) for k, v in row.items()} for row in csv.DictReader(fh)]
        man = os.path.join(report_dir, "figures", "manifest.json")
        self.figs = json.load(open(man)) if os.path.exists(man) else []
        setup = dict(self.s.get("setup") or {})
        sp = os.path.join(train_dir, "split.json")            # older reports: read the split / a run config
        if "n_train" not in setup and os.path.exists(sp):
            d = json.load(open(sp))
            setup.update(n_train=len(d["train"]), n_train_domains=len(d["train_domains"]), n_test=len(d["test"]),
                         n_test_domains=len(d["test_domains"]), split_mode=d["mode"])
        cfgs = sorted(glob.glob(os.path.join(train_dir, "runs", "*_s*", "config.json")))
        if "steps" not in setup and cfgs:
            cfg = json.load(open(cfgs[0]))
            setup.update({k: cfg.get(k) for k in ("steps", "effective_batch", "lr", "prompts", "lora_rank")})
        setup.setdefault("n_test", self.s["n_test"])
        self.setup = setup

    def mious(self, ch):
        return [v["miou"] for k, v in sorted(self.s["runs"].items()) if k.startswith(f"{ch}_s")]

    def stats(self, ch):
        v = self.mious(ch)
        return float(np.mean(v)), float(np.std(v)), len(v)

    def zero_shot(self):
        """Untrained SAM 3 (the same model whatever the step count): from this report, else the other one."""
        r = self.s["runs"].get("zeroshot_rgb")
        if r:
            return r["miou"]
        other = getattr(self, "other", None)
        return other.zero_shot() if other is not None and other is not self else None

    def fig(self, why_prefix):
        """First rule-chosen figure whose reason starts with `why_prefix`."""
        for m in self.figs:
            if m["why"].startswith(why_prefix):
                return os.path.join(self.dir, "figures", m["file"]), m
        return None, None


TEAM = ["Ali Eralp Sen", "Awais Ali", "Naman Sethi", "Priyesh Shrestha", "Samir Paudyal"]
# Student IDs are personal data and stay out of the (public) repository. To show them on the title
# slide, put one "Name: ID" per line in presentation/team_ids.local.txt (git-ignored).
TEAM_IDS_FILE = os.path.join(HERE, "team_ids.local.txt")


def team_lines():
    ids = {}
    if os.path.exists(TEAM_IDS_FILE):
        for line in open(TEAM_IDS_FILE, encoding="utf-8"):
            if ":" in line:
                name, sid = line.split(":", 1)
                ids[name.strip()] = sid.strip()
    names = [f"{n} ({ids[n]})" if ids.get(n) else n for n in TEAM]
    return ["  ".join(names[:3]), "  ".join(names[3:])]


def setup_text(st, key, fmt_="{:,}"):
    return fmt_.format(st[key]) if st.get(key) is not None else "—"


# ============================================================================ figures

def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def fig_fewshot(res, out):
    models = [m for m in res.models]
    arms = [a for a in ("1shot", "5shot") if any(res.has(m, a) for m in models)]
    ids5 = res.common([(m, "5shot") for m in models if res.has(m, "5shot")])
    fig, ax = plt.subplots(figsize=(4.3, 2.8))
    x = np.arange(len(models))
    width = 0.36 if len(arms) == 2 else 0.5
    for j, arm in enumerate(arms):
        vals = []
        for m in models:
            if not res.has(m, arm):
                vals.append(np.nan)
                continue
            ids = ids5 if arm == "5shot" else None
            vals.append(res.miou(m, arm, ids)[0])
        off = (j - (len(arms) - 1) / 2) * width
        bars = ax.bar(x + off, vals, width, color="#9e9e9e" if arm == "1shot" else "#2a4d38",
                      label="1-shot" if arm == "1shot" else f"5-shot ({len(ids5)} episodes)")
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}", ha="center", fontsize=7.5)
    base = class_miou(res.baseline(sorted(res.episodes)))[0]
    ax.axhline(base, color="#1a1a1a", lw=0.9, ls=(0, (4, 3)))
    ax.text(len(models) - 0.5, base + 0.006, f"predict-all baseline {base:.3f}", ha="right", fontsize=7, color="#666666")
    ax.set_xticks(x, [res.label(m) for m in models])
    ax.set_ylabel("mIoU (class-accumulated)")
    ax.set_ylim(0, max(0.55, ax.get_ylim()[1]))
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return save(fig, out)


def fig_perclass(res, out):
    ids = res.common([(m, "1shot") for m in res.models])
    classes = sorted({res.episodes[i]["class_id"] for i in ids})
    fig, ax = plt.subplots(figsize=(6.5, 2.7))
    x = np.arange(len(classes))
    w = 0.8 / max(1, len(res.models))
    for k, m in enumerate(res.models):
        per = res.miou(m, "1shot", ids)[1]
        ax.bar(x + (k - (len(res.models) - 1) / 2) * w, [per.get(c, 0) for c in classes], w,
               color=MODEL_COLOUR[m], label=res.label(m))
    base = class_miou(res.baseline(ids))[1]
    for xi, c in zip(x, classes):
        ax.plot([xi - 0.42, xi + 0.42], [base[c]] * 2, color="#1a1a1a", lw=1.1)
    ax.plot([], [], color="#1a1a1a", lw=1.1, label="predict-all baseline")
    ax.set_xticks(x, [CLASS_SHORT[c] for c in classes], rotation=32, ha="right", fontsize=7.5)
    ax.set_ylabel("IoU, 1-shot")
    ax.legend(frameon=False, ncol=4, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, 1.14))
    return save(fig, out)


def fig_train_delta(tr, out):
    """Per-class IoU change from adding NIR, with the 95 % CI from the paired bootstrap."""
    rows = sorted(tr.per_class, key=lambda r: r["delta_pooled"])
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    y = np.arange(len(rows))
    cols = ["#2a4d38" if r["significant"] and r["delta_pooled"] > 0 else
            ("#7d7d7d" if r["significant"] else "#c4c4c4") for r in rows]
    d = np.array([r["delta_pooled"] for r in rows])
    err = np.array([[r["delta_pooled"] - r["ci95_lo"], r["ci95_hi"] - r["delta_pooled"]] for r in rows]).T
    ax.barh(y, d, color=cols, height=0.65)
    ax.errorbar(d, y, xerr=err, fmt="none", ecolor="#1a1a1a", elinewidth=0.8, capsize=2)
    ax.axvline(0, color="#1a1a1a", lw=0.8)
    ax.set_yticks(y, [CLASS_SHORT[int(r["class_id"])] for r in rows], fontsize=8)
    ax.set_xlabel("Δ IoU  (RGB+NIR − RGB), 95 % CI", fontsize=8)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return save(fig, out)


def fig_upper(res, out):
    rows = []
    for m in res.models:
        if res.has(m, "oracle"):
            rows.append((res.label(m) + "\noracle box", res.miou(m, "oracle")[0], MODEL_COLOUR[m]))
    if res.has("sam3", "text"):
        rows.append(("SAM 3\ntext only", res.miou("sam3", "text")[0], "#8fb39a"))
    for m in res.models:
        rows.append((res.label(m) + "\n1-shot", res.miou(m, "1shot")[0], MODEL_COLOUR[m]))
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    bars = ax.bar(np.arange(len(rows)), [r[1] for r in rows], 0.6, color=[r[2] for r in rows])
    for i, (b, r) in enumerate(zip(bars, rows)):
        ax.text(b.get_x() + b.get_width() / 2, r[1] + 0.008, f"{r[1]:.3f}", ha="center", fontsize=7.5)
        if "1-shot" in r[0]:
            b.set_hatch("////")
            b.set_edgecolor("white")
    base = class_miou(res.baseline(sorted(res.episodes)))[0]
    ax.axhline(base, color="#1a1a1a", lw=0.9, ls=(0, (4, 3)))
    ax.text(-0.45, base + 0.01, f"predict-all baseline {base:.3f}", ha="left", fontsize=7, color="#666666")
    ax.set_xticks(np.arange(len(rows)), [r[0] for r in rows], fontsize=7)
    ax.set_ylabel("mIoU (class-accumulated)")
    return save(fig, out)


def overlay(rgb, mask, colour=(1.0, 0.0, 1.0), alpha=0.45):
    out = rgb.astype(np.float32) / 255.0
    out[mask] = (1 - alpha) * out[mask] + alpha * np.array(colour)
    edge = mask & ~ndimage.binary_erosion(mask, iterations=2)
    out[edge] = colour
    return out


def error_map(rgb, pred, gt):
    img = rgb.astype(np.float32) / 255.0 * 0.45
    img[pred & gt] = (0.0, 0.85, 0.0)
    img[pred & ~gt] = (0.9, 0.0, 0.0)
    img[~pred & gt] = (0.0, 0.6, 1.0)
    return img


def fig_qualitative(res, picked, out):
    models = res.models
    cols = 3 + len(models)
    rows = len(picked)
    cell, gap, lab_w, top = 0.74, 0.17, 1.2, 0.24
    fig = plt.figure(figsize=(cols * (cell + 0.04) + lab_w, rows * (cell + gap) + top + 0.05))
    W, H = fig.get_figwidth(), fig.get_figheight()
    heads = ["Support + mask", "Query", "Ground truth"] + [res.label(m) for m in models]
    for r, p in enumerate(picked):
        ep = res.episodes[p["id"]]
        c = ep["class_id"]
        s_rgb, s_msk = load_patch(res.root, ep["support"], res.band_order)
        q_rgb, q_msk = load_patch(res.root, ep["query"], res.band_order)
        gt = q_msk == c
        preds = [load_pred(res.runs[(m, "1shot")]["dir"], p["id"]) for m in models]
        ious = [episode_iou(res.runs[(m, "1shot")]["records"][p["id"]]) for m in models]
        panels = [overlay(s_rgb, s_msk == c), q_rgb / 255.0, overlay(q_rgb, gt)] + \
                 [error_map(q_rgb, pr, gt) for pr in preds]
        best = int(np.argmax(ious))
        for j, img in enumerate(panels):
            ax = fig.add_axes([(lab_w + j * (cell + 0.04)) / W, 1 - (top + r * (cell + gap) + cell) / H, cell / W, cell / H])
            ax.imshow(np.clip(img, 0, 1))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(heads[j], fontsize=7.5, fontweight="bold", pad=3)
            if j >= 3:
                k = j - 3
                txt = "no mask" if not preds[k].any() else f"IoU {ious[k]:.2f}"
                ax.text(0.5, -0.06, txt, transform=ax.transAxes, ha="center", va="top", fontsize=7,
                        fontweight="bold" if k == best and len(models) > 1 else "normal",
                        color="#2a4d38" if k == best and len(models) > 1 else "#1a1a1a")
        label = "\n".join(textwrap.wrap(FLAIR_CLASSES[c].replace("_", " ").capitalize(), 12))
        why = {"largest disagreement between models": "", "all models succeed (highest worst-model IoU)": "all succeed",
               "all models fail (lowest best-model IoU)": "all fail"}.get(p.get("reason", ""), "")
        fig.text(0.06 / W, 1 - (top + r * (cell + gap) + cell / 2) / H,
                 f"{label}\n{ep['support_domain'][:4]} → {ep['query_domain'][:4]}" + (f"\n({why})" if why else ""),
                 fontsize=6.8, va="center")
    return save(fig, out)


def fig_dataset(res, out, n=3):
    """Real FLAIR patches: RGB and labels (official FLAIR-HUB colours)."""
    # Pick varied, class-rich queries deterministically: most distinct classes, different domains.
    cands = []
    for rel in sorted({e["query"] for e in res.episodes.values()}):
        lab = read_label(mask_path_for(os.path.join(res.root, rel)))
        k = len([c for c in np.unique(lab) if (lab == c).mean() > 0.03])
        cands.append((k, rel))
    chosen, doms = [], set()
    for k, rel in sorted(cands, reverse=True):
        d = re.search(r"D\d{3}[-_]\d{4}", rel).group(0)
        if d not in doms:
            chosen.append(rel)
            doms.add(d)
        if len(chosen) == n:
            break
    fig, axes = plt.subplots(2, n, figsize=(1.55 * n, 3.3))
    present = set()
    for j, rel in enumerate(chosen):
        rgb, lab = load_patch(res.root, rel, res.band_order)
        present |= set(np.unique(lab).tolist())
        col = np.zeros(lab.shape + (3,))
        for c, hx in CLASS_HEX.items():
            col[lab == c] = [int(hx[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        axes[0, j].imshow(rgb)
        axes[1, j].imshow(col)
        axes[0, j].set_title(re.search(r"D\d{3}[-_]\d{4}", rel).group(0).replace("-", "_"), fontsize=7)
        for a in axes[:, j]:
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
    from matplotlib.patches import Patch
    handles = [Patch(color=CLASS_HEX[c], label=CLASS_SHORT[c]) for c in range(1, 13) if c in present]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.2, frameon=False,
               bbox_to_anchor=(0.5, -0.13), handlelength=1.0, columnspacing=0.8)
    fig.subplots_adjust(wspace=0.05, hspace=0.12)
    return save(fig, out)


def fig_canvas(res, out):
    from models.canvas import box_to_canvas_xyxy, build_canvas
    from models.sam3_fss import support_box
    ep = next(e for e in res.episodes.values() if e["class_id"] == 9) if any(
        e["class_id"] == 9 for e in res.episodes.values()) else next(iter(res.episodes.values()))
    s_rgb, s_msk = load_patch(res.root, ep["support"], res.band_order)
    q_rgb, _ = load_patch(res.root, ep["query"], res.band_order)
    box, _ = support_box(s_msk == ep["class_id"])
    canvas, layout, _ = build_canvas(s_rgb, q_rgb, support_box_xywh=box)
    x0, y0, x1, y1 = box_to_canvas_xyxy(box, layout.support, s_msk.shape[::-1])
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    ax.imshow(canvas)
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#00ff00", lw=1.6))
    ax.axhline(layout.support[3] - 0.5, color="white", lw=1.4)
    for yy, t in ((40, "support 60 %"), (layout.support[3] + 45, "query 40 %")):
        ax.text(14, yy, t, color="white", fontsize=7, fontweight="bold",
                bbox=dict(fc="black", alpha=0.45, lw=0, pad=1.5))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return save(fig, out)


def fig_ndvi(res, model, out):
    """Real episode: RGB | model segment | NDVI | NDVI inside the segment | local Shannon H."""
    from scripts.ndvi_indicators import local_entropy, ndvi_of
    d = res.ndvi[model]
    good = sorted([e for e in d["episodes"] if e["ndvi_pred"] is not None],
                  key=lambda e: -episode_iou(res.runs[(model, "1shot")]["records"][e["id"]]))
    picks, seen = [], set()
    for e in good:
        if e["class_id"] not in seen:
            picks.append(e)
            seen.add(e["class_id"])
        if len(picks) == 2:
            break
    fig, axes = plt.subplots(len(picks), 5, figsize=(6.4, 1.38 * len(picks) + 0.3))
    axes = np.atleast_2d(axes)
    heads = ["RGB", f"{MODEL_NAME[model]} segment", "NDVI (NIR, red)", "NDVI in segment", "Shannon H, 7 px"]
    for r, e in enumerate(picks):
        ep = res.episodes[e["id"]]
        q = os.path.join(res.root, ep["query"])
        rgb, lab = load_patch(res.root, ep["query"], res.band_order)
        ndvi = ndvi_of(q, res.band_order)
        H = local_entropy(ndvi, 7, 16)
        pred = load_pred(res.runs[(model, "1shot")]["dir"], e["id"])
        seg = np.where(pred, ndvi, np.nan)
        ims = [rgb / 255.0, overlay(rgb, pred, (0.16, 0.30, 0.22), 0.55)]
        for j in range(5):
            ax = axes[r, j]
            if j < 2:
                ax.imshow(np.clip(ims[j], 0, 1))
            elif j == 2:
                im = ax.imshow(ndvi, cmap="RdYlGn", vmin=-0.4, vmax=0.6)
            elif j == 3:
                ax.imshow(rgb.mean(-1), cmap="gray", vmin=0, vmax=255, alpha=0.35)
                ax.imshow(seg, cmap="RdYlGn", vmin=-0.4, vmax=0.6)
            else:
                im2 = ax.imshow(H, cmap="magma", vmin=0, vmax=0.8)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(heads[j], fontsize=7.2, fontweight="bold", pad=3)
        axes[r, 0].set_ylabel(CLASS_SHORT[e["class_id"]], fontsize=7)
    fig.subplots_adjust(wspace=0.04, hspace=0.06, right=0.9)
    cax = fig.add_axes([0.915, 0.55, 0.012, 0.3])
    fig.colorbar(im, cax=cax).ax.tick_params(labelsize=6)
    cax.set_title("NDVI", fontsize=6)
    cax2 = fig.add_axes([0.915, 0.13, 0.012, 0.3])
    fig.colorbar(im2, cax=cax2).ax.tick_params(labelsize=6)
    cax2.set_title("H", fontsize=6)
    return save(fig, out)


def fig_ndvi_classes(res, out):
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    names = [FLAIR_CLASSES[c] for c in VEG]
    y = np.arange(len(names))[::-1]
    for k, m in enumerate([m for m in res.models if m in res.ndvi]):
        pc = res.ndvi[m]["per_class"]
        for yi, n in zip(y, names):
            if n not in pc or pc[n]["ndvi_pred_mean"] is None:
                continue
            gt, pr = pc[n]["ndvi_gt_mean"], pc[n]["ndvi_pred_mean"]
            if k == 0:
                ax.scatter([gt], [yi], s=26, facecolor="white", edgecolor="#1a1a1a", lw=1, zorder=3)
            ax.plot([gt, pr], [yi + 0.12 * (k - 0.5), yi + 0.12 * (k - 0.5)], color="#b0b0b0", lw=1.2, zorder=1)
            ax.scatter([pr], [yi + 0.12 * (k - 0.5)], s=22, color=MODEL_COLOUR[m], zorder=2)
    ax.scatter([], [], s=26, facecolor="white", edgecolor="#1a1a1a", label="ground-truth segments")
    for m in [m for m in res.models if m in res.ndvi]:
        ax.scatter([], [], s=22, color=MODEL_COLOUR[m], label=f"{MODEL_NAME[m]} segments")
    ax.set_yticks(y, [CLASS_SHORT[c] for c in VEG], fontsize=7.5)
    ax.set_xlabel("mean NDVI of the segment")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2)
    return save(fig, out)


# ============================================================================ pptx helpers

def rgb(hexstr):
    return RGBColor.from_string(hexstr)


def _runs(paragraph, text, size, colour, bold=False, italic=False):
    """Add runs to a paragraph; **bold** spans become bold runs, "\n" becomes a real line break."""
    for k, line in enumerate(text.split("\n")):
        if k:
            paragraph.add_line_break()
        for part in re.split(r"(\*\*.+?\*\*)", line):
            if not part:
                continue
            r = paragraph.add_run()
            is_b = part.startswith("**") and part.endswith("**")
            r.text = part[2:-2] if is_b else part
            r.font.name = FONT
            r.font.size = Pt(size)
            r.font.bold = bold or is_b
            r.font.italic = italic
            r.font.color.rgb = rgb(colour)


def text(slide, x, y, w, h, paras, size=12, colour=INK, bold=False, italic=False, align=None,
         anchor=MSO_ANCHOR.TOP, spacing=4, bullets=False, line=None):
    """paras: str or list of str. Each str is a paragraph."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paras = [paras] if isinstance(paras, str) else paras
    for i, ptxt in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if spacing:
            p.space_after = Pt(spacing)
        if line:
            p.line_spacing = line
        if align:
            p.alignment = align
        if bullets:
            pPr = p._p.get_or_add_pPr()
            pPr.set("marL", str(int(Inches(0.27))))
            pPr.set("indent", str(-int(Inches(0.2))))
            etree.SubElement(pPr, qn("a:buFont")).set("typeface", "Arial")
            etree.SubElement(pPr, qn("a:buChar")).set("char", "•")
        _runs(p, ptxt, size, colour, bold, italic)
    return tb


def box(slide, x, y, w, h, label="", fill=BOX, line=BOXLINE, lw=1.0, size=10, colour=INK, bold=True,
        shape=MSO_SHAPE.RECTANGLE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, sub=None):
    sh = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.shadow.inherit = False
    if fill:
        sh.fill.solid()
        sh.fill.fore_color.rgb = rgb(fill)
    else:
        sh.fill.background()
    if line:
        sh.line.color.rgb = rgb(line)
        sh.line.width = Pt(lw)
    else:
        sh.line.fill.background()
    tf = sh.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    tf.vertical_anchor = anchor
    if label:
        p = tf.paragraphs[0]
        p.alignment = align
        _runs(p, label, size, colour, bold)
        if sub:
            p2 = tf.add_paragraph()
            p2.alignment = align
            _runs(p2, sub, size - 1.5, colour if colour != INK else GREY, False)
    return sh


def arrow(slide, x1, y1, x2, y2, colour=BOXLINE, lw=1.0):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(colour)
    c.line.width = Pt(lw)
    ln = c.line._get_or_add_ln()
    tail = etree.SubElement(ln, qn("a:tailEnd"))
    tail.set("type", "triangle")
    tail.set("w", "med")
    tail.set("len", "med")
    return c


def hline(slide, x1, x2, y, colour=RULE, lw=0.75):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y), Inches(x2), Inches(y))
    c.line.color.rgb = rgb(colour)
    c.line.width = Pt(lw)
    return c


def picture(slide, path, x, y, w=None, h=None):
    return slide.shapes.add_picture(path, Inches(x), Inches(y),
                                    width=Inches(w) if w else None, height=Inches(h) if h else None)


def table(slide, x, y, col_w, rows, size=10, row_h=0.28, bold_cells=(), header=True, align_num=True):
    n_r, n_c = len(rows), len(rows[0])
    gf = slide.shapes.add_table(n_r, n_c, Inches(x), Inches(y), Inches(sum(col_w)), Inches(row_h * n_r))
    tbl = gf.table
    tblPr = tbl._tbl.tblPr
    tblPr.set("firstRow", "0")
    tblPr.set("bandRow", "0")
    sid = tblPr.find(qn("a:tableStyleId"))
    if sid is None:
        sid = etree.SubElement(tblPr, qn("a:tableStyleId"))
    sid.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"          # "No Style, No Grid"
    for j, w in enumerate(col_w):
        tbl.columns[j].width = Inches(w)
    for i in range(n_r):
        tbl.rows[i].height = Inches(row_h)
        for j in range(n_c):
            cell = tbl.cell(i, j)
            tcPr = cell._tc.get_or_add_tcPr()
            for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
                ln = etree.SubElement(tcPr, qn(tag), w=str(int(Pt(0.75))), cap="flat", cmpd="sng", algn="ctr")
                etree.SubElement(etree.SubElement(ln, qn("a:solidFill")), qn("a:srgbClr"), val=TABLE)
                etree.SubElement(ln, qn("a:prstDash"), val="solid")
            if header and i == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = rgb(TABLE)
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = rgb("FFFFFF")
            cell.margin_left = cell.margin_right = Inches(0.07)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            val = str(rows[i][j])
            is_num = bool(re.fullmatch(r"[-+−]?[\d.]+( ?%| s| pp)?|—|–|n/a", val.strip()))
            if align_num and is_num and j > 0:
                p.alignment = PP_ALIGN.CENTER
            _runs(p, val, size, INK, bold=(header and i == 0) or (i, j) in bold_cells)
    return gf


def speaker_script_sections(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "speaker_script.md")):
    """{slide number: script text} from the '## <n> · title · time' sections; markdown bold removed."""
    if not os.path.exists(path):
        return {}
    out, cur, buf = {}, None, []
    for line in open(path, encoding="utf-8"):
        m = re.match(r"^## (\d+) ", line)
        if line.startswith("## "):
            if cur is not None:
                out[cur] = "".join(buf).strip()
            cur, buf = (int(m.group(1)) if m else None), []
            continue
        if cur is not None:
            buf.append(line.replace("**", ""))
    if cur is not None:
        out[cur] = "".join(buf).strip()
    return out


class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width = Inches(W_IN)
        self.prs.slide_height = Inches(H_IN)
        self.slides = []

    def slide(self, title=None, notes=""):  # notes are replaced by the speaker script in finish()
        s = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        if title:
            text(s, 0.5, 0.40, 8.1, 0.5, title, size=24 if len(title) < 52 else 21, bold=True,
                 anchor=MSO_ANCHOR.MIDDLE, spacing=0)
            hline(s, 0.5, 9.5, 0.95)
            picture(s, LOGO, 8.74, 0.07, w=0.82)
        if notes:
            s.notes_slide.notes_text_frame.text = notes
        self.slides.append(s)
        return s

    def finish(self, path):
        total = len(self.slides)
        for i, s in enumerate(self.slides[1:], start=2):
            text(s, 0.5, 5.28, 6.5, 0.22, FOOTER, size=9, colour=GREY, spacing=0)
            text(s, 8.5, 5.28, 1.0, 0.22, f"{i} / {total}", size=9, colour=GREY, align=PP_ALIGN.RIGHT, spacing=0)
        # Presenter notes = the word-for-word script (presentation/speaker_script.md, "## <n> · ..." sections).
        for n, body in speaker_script_sections().items():
            if 1 <= n <= total:
                self.slides[n - 1].notes_slide.notes_text_frame.text = body
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.prs.save(path)


# ============================================================================ the deck

def fmt(v, d=3):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


def build(res: Results, figdir: str, out: str, tr: "TrainResults" = None):
    D = Deck()
    tr = tr if tr is not None and tr.ok else None
    has3, has2, has1 = "sam3" in res.models, "sam2" in res.models, "sam1" in res.models
    n_eps = len(res.episodes)
    n_cls = len({e["class_id"] for e in res.episodes.values()})
    doms = res.meta.get("n_domains")
    pending = [MODEL_NAME[m] for m in MODEL_ORDER if m not in res.models]
    if res.incomplete:
        print("  not shown (incomplete runs):", ", ".join(res.incomplete))

    # ---------------------------------------------------------------- 1 title
    s = D.slide(notes="Final-phase presentation. Previous phase: LoveDA baselines and SAM2. This phase: FLAIR, "
                      "a controlled SAM 1/2/3 few-shot comparison, and NIR-based ecological read-out.")
    text(s, 0.5, 1.7, 7.4, 1.15, ["Few-Shot Learning for Biodiversity", "Assessment"], size=32, bold=True,
         spacing=0, line=1.05)
    text(s, 0.5, 2.88, 8.4, 0.5, "Segment Anything 1, 2 and 3 for few-shot habitat segmentation on FLAIR, "
         "and what near-infrared adds on top", size=14, colour=GREY, spacing=0)
    picture(s, LOGO, 7.87, 0.09, w=1.81)
    text(s, 0.66, 3.5, 6, 0.25, "Group 7  SRP 2025/26 · Final presentation", size=12, spacing=0)
    text(s, 0.66, 3.88, 7.5, 0.72, ["**Team Members**"] + team_lines(), size=12, spacing=1)
    text(s, 0.66, 4.62, 5, 0.72, ["**Supervisors:**", "Prof. Dr. Niels Landwehr", "Ahmad Bdeir"], size=12, spacing=1)

    # ---------------------------------------------------------------- 2 recap: problem
    s = D.slide("Recap — problem and research question",
                notes="Same problem as last time: habitat maps need pixel labels, experts can only label a few "
                      "images per class, so we need few-shot segmentation.")
    text(s, 0.5, 1.2, 4.3, 0.3, "**Why this matters**", size=13)
    text(s, 0.5, 1.55, 4.3, 1.6, [
        "Biodiversity monitoring needs habitat maps across many regions and landscapes.",
        "UAV imagery gives 0.1–0.3 m per pixel, cheaply and at scale.",
        "Pixel-level labels need domain experts: only a handful of labelled images per class."],
        size=11, bullets=True)
    text(s, 5.1, 1.2, 4.4, 0.3, "**Research question**", size=13)
    text(s, 5.1, 1.55, 4.4, 0.9, "Can a model segment **novel habitat classes in unseen regions** from "
         "K ∈ {1, 5} labelled support examples per class?", size=12)
    table(s, 5.1, 2.55, [1.2, 1.55, 1.65], [
        ["Symbol", "Meaning", ""],
        ["S = {(xᵢ, yᵢ)}", "K labelled pairs", "image + mask"],
        ["x_q", "query image", "same class"],
        ["ŷ_q = f(x_q ; S)", "predicted mask", "maximise mIoU"]], size=9.5, row_h=0.27)
    text(s, 0.5, 3.45, 4.3, 0.3, "**Where we were heading**", size=13)
    text(s, 0.5, 3.8, 4.3, 1.2, [
        "SAM2 encoder + prototype few-shot + NDVI / Shannon ecological channels.",
        "Then transfer to the EU UAV data (Germany first)."], size=11, bullets=True)

    # ---------------------------------------------------------------- 3 recap: results + what changed
    rec = [["HRNet-W32", "Supervised, full labels", "0.580"],
           ["DeepLabv2 + CBST", "UDA Rural → Urban", "0.351"],
           ["SAM2 Hiera-S + FPN", "Frozen encoder, 1.5 M params", "0.4746"],
           ["SAM2 zero-shot *", "Oracle: GT majority vote", "0.846"]]
    s = D.slide("Recap — results, and what had to change",
                notes="Numbers are from the previous presentation (LoveDA val). The right column is why this "
                      "phase changed dataset and approach.")
    text(s, 0.5, 1.2, 4.6, 0.3, "**LoveDA results (previous presentation)**", size=13)
    table(s, 0.5, 1.6, [1.6, 2.0, 0.8], [["Approach", "Setting", "mIoU"]] + rec, size=9.5, row_h=0.3,
          bold_cells={(3, 2)})
    text(s, 0.5, 3.2, 4.5, 0.3, "* upper bound, uses ground truth to label SAM's masks — not a usable model.",
         size=9, colour=GREY, italic=True)
    box(s, 5.35, 1.2, 4.15, 3.75, fill=BOX, line=RULE, lw=0.75)
    text(s, 5.55, 1.35, 3.8, 0.3, "**What we learned → what changed**", size=12.5)
    text(s, 5.55, 1.75, 3.8, 3.1, [
        "LoveDA is **RGB only** — no NIR, so NDVI and Shannon cannot be computed there.",
        "ISPRS Potsdam was flown **leaf-off**: tree NDVI ≈ 0.21 overlaps bare soil.",
        "Training a prompt head on a frozen SAM stalled at **0.06** novel-class mIoU.",
        "→ New dataset with NIR (**FLAIR**) and **training-free** prompting of SAM 1, 2 and 3."],
        size=10.5, bullets=True, spacing=6)

    # ---------------------------------------------------------------- 4 why FLAIR
    s = D.slide("Why we moved to FLAIR",
                notes="FLAIR is the closest public proxy for the EU UAV target: NIR, sub-metre, European "
                      "landscapes, many regions and dates.")
    table(s, 0.5, 1.2, [1.35, 1.75, 1.95, 2.1, 1.85], [
        ["", "LoveDA", "ISPRS Potsdam", "FLAIR", "EU UAV (target)"],
        ["Bands", "RGB", "RGB + IR + DSM", "RGB + NIR (+ height)", "RGB (+ NIR subset)"],
        ["Resolution", "0.3 m", "0.05 m", "0.2 m", "UAV, 0.1–0.3 m"],
        ["Region", "3 Chinese cities", "one German city", "France, many départements", "PT · DE · UK · AT"],
        ["Acquisitions", "—", "one date, leaf-off", "different dates / seasons", "—"],
        ["Classes", "7 cartographic", "6 cartographic", "19, incl. ecological", "habitat classes"],
        ["NDVI usable?", "no NIR", "leaf-off", "yes", "yes (NIR subset)"]],
        size=9.5, row_h=0.29, bold_cells={(i, 3) for i in range(1, 7)})
    text(s, 0.5, 3.45, 9.0, 1.7, [
        "**NIR on every patch** → NDVI and Shannon heterogeneity are computable; season becomes a variable, not a handicap.",
        "**0.2 m, European landscapes** → the closest public stand-in for our EU UAV data.",
        "**Many spatio-temporal domains** → we can put support and query in different regions/dates "
        "(tests geographic + seasonal transfer, not memorisation).",
        "**Ecological classes** (coniferous vs deciduous, brushwood, vineyard) instead of tree vs low vegetation."],
        size=11, bullets=True, spacing=5)

    # ---------------------------------------------------------------- 5 FLAIR dataset
    fp = fig_dataset(res, os.path.join(figdir, "dataset.png"))
    s = D.slide("The FLAIR dataset (IGN, France)",
                notes="Real FLAIR-HUB patches from our evaluation set, with their labels in the official colours.")
    stats = [("93,462", "patches of 512 × 512 px at 0.2 m\n(FLAIR #1 + #2; FLAIR-HUB extends it)"),
             ("55", "spatio-temporal domains\n974 areas · 980 km²"),
             ("19", "land-cover classes\nwe evaluate the 12 main ones"),
             ("4 + 1", "bands: R, G, B, NIR\n+ height (not used: our target has none)")]
    for k, (big, small) in enumerate(stats):
        y = 1.2 + k * 0.92
        text(s, 0.5, y, 1.35, 0.6, f"**{big}**", size=26, colour=ACCENT, spacing=0)
        text(s, 1.95, y + 0.06, 2.9, 0.75, small.split("\n"), size=10.5, spacing=0)
    picture(s, fp, 5.15, 1.15, h=3.75)
    text(s, 0.5, 5.0, 9, 0.2, "Sources: IGN FLAIR card (huggingface.co/datasets/IGNF/FLAIR), github.com/IGNF/FLAIR-HUB; "
         "Etalab Open Licence 2.0. Images: real patches from our evaluation set.", size=8, colour=GREY, italic=True)

    # ---------------------------------------------------------------- 6 SAM family
    s = D.slide("Segment Anything: SAM 1 → SAM 2 → SAM 3",
                notes="All three are frozen in this work. Parameter counts for SAM 2.1 and SAM 3 were counted from "
                      "the checkpoints we ran.")
    table(s, 0.5, 1.2, [1.55, 2.4, 2.4, 2.65], [
        ["", "SAM 1  (Kirillov et al., 2023)", "SAM 2  (Ravi et al., 2024)", "SAM 3  (Carion et al., 2025)"],
        ["Image encoder", "ViT (B / L / H)", "Hiera, hierarchical multi-scale", "Perception Encoder ViT, shared by detector + tracker"],
        ["Model we run", "ViT-B · 91 M params", "2.1 Hiera-S · 46 M params", "841 M params (454 M vision + 353 M text)"],
        ["Training data", "SA-1B: 11 M images, 1.1 B masks", "SA-1B + SA-V video", "SA-Co: 4 M unique concept labels"],
        ["Prompts", "points · boxes · masks", "points · boxes · masks, + video memory", "text phrases · image exemplars · points / boxes"],
        ["Output", "class-agnostic masks", "class-agnostic masks + tracking", "every instance of a concept"],
        ["Few-shot here", "prototype → point + box prompts", "prototype → point + box prompts", "exemplar box on a support + query canvas"]],
        size=9, row_h=0.43, bold_cells={(2, 3), (6, 3)})

    # ---------------------------------------------------------------- 7 SAM 3
    s = D.slide("SAM 3 — Promptable Concept Segmentation",
                notes="SAM 3 segments all instances of a concept from a text phrase and/or image exemplars. Its "
                      "exemplars must lie in the image being segmented — hence the canvas on the next slide.")
    box(s, 0.5, 1.35, 1.35, 0.55, "Text prompt", sub="“vineyard”")
    box(s, 0.5, 2.1, 1.35, 0.55, "Image exemplar", sub="box on the image")
    box(s, 0.5, 3.15, 1.35, 0.6, "Image", sub="1008 × 1008")
    box(s, 2.25, 1.55, 1.55, 0.9, "Text + geometry\nencoders")
    box(s, 2.25, 3.05, 1.55, 0.8, "Perception\nEncoder ❄", sub="shared backbone")
    box(s, 4.2, 2.2, 1.45, 0.9, "Fusion\nencoder")
    box(s, 6.0, 2.2, 1.5, 0.9, "DETR decoder\n+ presence head")
    box(s, 7.85, 2.2, 1.6, 0.9, "Masks for every\ninstance", fill=ACCENT, line=ACCENT, colour="FFFFFF")
    for (x1, y1, x2, y2) in ((1.85, 1.62, 2.25, 1.9), (1.85, 2.37, 2.25, 2.1), (1.85, 3.45, 2.25, 3.45),
                             (3.8, 2.0, 4.2, 2.5), (3.8, 3.45, 4.2, 2.85), (5.65, 2.65, 6.0, 2.65),
                             (7.5, 2.65, 7.85, 2.65)):
        arrow(s, x1, y1, x2, y2)
    text(s, 0.5, 4.1, 9.0, 1.1, [
        "**Presence head** decouples recognition (“is the concept here?”) from localisation — fewer false positives.",
        "SA-Co data engine: 4 M unique concept labels with hard negatives. Paper: doubles prior accuracy on image + video PCS.",
        "Limitation for few-shot: exemplars must be **in the same image** — a reference from another image cannot be used directly."],
        size=10.5, bullets=True, spacing=3)

    # ---------------------------------------------------------------- 8 protocol
    cp = fig_canvas(res, os.path.join(figdir, "canvas.png"))
    s = D.slide("How each model does few-shot",
                notes="The only thing that changes between rows of our results is the model and its prompting recipe; "
                      "episodes, metrics and data are identical.")
    text(s, 0.5, 1.12, 5.7, 0.3, "**SAM 1 · SAM 2 — prototype prompting (PerSAM-style)**", size=11.5)
    xs = [0.5, 1.72, 2.94, 4.16, 5.38]
    labels = [("Support\n+ mask", None), ("Frozen\nencoder", None), ("Prototype", "masked avg"),
              ("Cosine map\non query", None), ("Points + box\n→ decoder", None)]
    for i, (x, (l, sub)) in enumerate(zip(xs, labels)):
        box(s, x, 1.5, 1.02, 0.62, l, sub=sub, size=9, fill=ACCENT if i == 4 else BOX,
            line=ACCENT if i == 4 else BOXLINE, colour="FFFFFF" if i == 4 else INK)
        if i:
            arrow(s, x - 0.2, 1.81, x, 1.81)
    text(s, 0.5, 2.35, 5.7, 0.3, "**SAM 3 — spatial-concatenation canvas (Tsai et al., 2026)**", size=11.5)
    labels = [("Support on top,\nquery below", None), ("Exemplar box\non support", None), ("SAM 3 ❄", None),
              ("Crop query\nhalf → mask", None)]
    for i, (x, (l, sub)) in enumerate(zip([0.5, 1.95, 3.4, 4.85], labels)):
        box(s, x, 2.73, 1.25, 0.62, l, size=9, fill=ACCENT if i == 3 else BOX, line=ACCENT if i == 3 else BOXLINE,
            colour="FFFFFF" if i == 3 else INK)
        if i:
            arrow(s, x - 0.2, 3.04, x, 3.04)
    picture(s, cp, 7.25, 1.12, h=2.25)
    text(s, 7.1, 3.4, 2.4, 0.3, "a real canvas; green = exemplar box", size=8, colour=GREY, italic=True,
         align=PP_ALIGN.CENTER)
    text(s, 0.5, 3.62, 9.0, 1.55, [
        f"**{n_eps} episodes** ({n_cls} classes × {n_eps // max(1, n_cls)}), support and query from **different FLAIR "
        f"domains** ({doms} domains); every model runs the same list.",
        "Class must cover 2–90 % of both patches (a class filling the frame lets “predict everything” score ≈ 1).",
        "Metrics: class-accumulated mIoU (few-shot standard), FB-IoU, empty-prediction rate; baselines "
        "predict-all / predict-nothing; paired bootstrap 95 % CI.",
        "All models frozen — **no training anywhere** in these results."], size=10, bullets=True, spacing=3)

    # ---------------------------------------------------------------- 9 upper bounds
    if any(res.has(m, "oracle") for m in res.models) or res.has("sam3", "text"):
        up = fig_upper(res, os.path.join(figdir, "upper.png"))
        s = D.slide("Experiment 1 — upper bounds",
                    notes="Oracle: the prompt box comes from the query's own ground truth — an upper bound on mask "
                          "quality, not a usable method. Text only: SAM 3 given the class phrase, no labelled example.")
        picture(s, up, 0.45, 1.15, w=5.3)
        rows = [["Setting", "Model", "mIoU"]]
        for m in res.models:
            if res.has(m, "oracle"):
                rows.append(["Oracle box (upper bound)", res.label(m), fmt(res.miou(m, "oracle")[0])])
        if res.has("sam3", "text"):
            rows.append(["Class phrase only", "SAM 3", fmt(res.miou("sam3", "text")[0])])
        for m in res.models:
            rows.append(["1-shot (the task)", res.label(m), fmt(res.miou(m, "1shot")[0])])
        table(s, 5.95, 1.2, [1.75, 1.2, 0.6], rows, size=9, row_h=0.27)
        obs = []
        if res.has("sam2", "oracle") and res.has("sam3", "oracle"):
            a, b = res.miou("sam3", "oracle")[0], res.miou("sam2", "oracle")[0]
            obs.append(f"With ideal prompts SAM 3 reaches **{a:.3f}** vs SAM 2 **{b:.3f}**.")
        if res.has("sam3", "text"):
            t, one = res.miou("sam3", "text")[0], res.miou("sam3", "1shot")[0]
            obs.append(f"Class name alone: **{t:.3f}** — " + ("one labelled example is worth more than the name." if one > t
                                                               else "the name alone already matches one example."))
        obs.append("Gap between oracle and 1-shot = what the few-shot prompting loses: the decoder is not the bottleneck.")
        text(s, 5.95, 1.35 + 0.27 * len(rows), 3.55, 2.2, obs, size=9.5, bullets=True, spacing=3)
        text(s, 0.5, 4.95, 9, 0.25, "Oracle prompt = box of the largest ground-truth region of the query (+ interior "
             "positive / far negative point for SAM 1/2). Upper bound only.", size=8, colour=GREY, italic=True)

    # ---------------------------------------------------------------- 10 few-shot main
    fs = fig_fewshot(res, os.path.join(figdir, "fewshot.png"))
    s = D.slide("Experiment 2 — few-shot results (K = 1, 5)",
                notes="Main result. Same episodes for every model; 5-shot on the common subset of episodes.")
    picture(s, fs, 0.45, 1.12, w=4.75)
    ids_all = sorted(res.episodes)
    base_all = class_miou(res.baseline(ids_all))[0]
    ids5 = res.common([(m, "5shot") for m in res.models if res.has(m, "5shot")])
    rows = [["Model", "1-shot", "5-shot", "FB-IoU", "empty", "s / ep."],
            ["Predict-all baseline", fmt(base_all), fmt(class_miou(res.baseline(ids5))[0]) if ids5 else "—",
             fmt(fb_iou(res.baseline(ids_all))), "–", "–"]]
    best1 = max(res.models, key=lambda m: res.miou(m, "1shot")[0])
    bold = set()
    for m in res.models:
        r1 = res.miou(m, "1shot")[0]
        r5 = res.miou(m, "5shot", ids5)[0] if res.has(m, "5shot") and ids5 else None
        rows.append([res.label(m), fmt(r1), fmt(r5), fmt(fb_iou(res.recs(m, "1shot"))),
                     f"{100 * res.empty_rate(m, '1shot'):.0f} %", f"{res.seconds(m, '1shot'):.2f}"])
        if m == best1:
            bold |= {(len(rows) - 1, 0), (len(rows) - 1, 1)}
    for p in pending:
        rows.append([f"{p} (cluster run)", "pending", "", "", "", ""])
    table(s, 5.3, 1.2, [1.33, 0.52, 0.52, 0.55, 0.55, 0.53], rows, size=8.5, row_h=0.27, bold_cells=bold)
    lines = []
    for m in res.models:
        v = res.miou(m, "1shot")[0]
        lines.append(f"{res.label(m)} 1-shot is **{'above' if v > base_all else 'below'}** predict-all "
                     f"({v:.3f} vs {base_all:.3f}).")
    if has2 and has3:
        bt = paired_bootstrap(res.runs[("sam3", "1shot")]["records"], res.runs[("sam2", "1shot")]["records"])
        sig = "significant" if bt["significant"] else "not significant"
        lines.append(f"SAM 3 − SAM 2 (1-shot): **{bt['diff']:+.3f}** [{bt['ci_lo']:+.3f}, {bt['ci_hi']:+.3f}], {sig}.")
    for m in res.models:
        if res.has(m, "5shot") and ids5:
            g = res.miou(m, "5shot", ids5)[0] - res.miou(m, "1shot", ids5)[0]
            lines.append(f"{res.label(m)}: 1 → 5 shots **{g:+.3f}** on the same {len(ids5)} episodes.")
    text(s, 5.3, 1.35 + 0.27 * len(rows), 4.2, 1.9, lines, size=9, bullets=True, spacing=2)
    text(s, 0.5, 4.95, 9, 0.25, f"{n_eps} cross-domain episodes; 5-shot on {len(ids5)} of them (compute). "
         "s / ep. = median seconds per episode on an Apple M1 Pro GPU (MPS). Toy subset of FLAIR-HUB — preliminary.",
         size=8, colour=GREY, italic=True)

    # ---------------------------------------------------------------- 11 per class
    pc = fig_perclass(res, os.path.join(figdir, "perclass.png"))
    s = D.slide("Per-class IoU (1-shot)",
                notes="Black ticks are the predict-all baseline: a bar below its tick means the model did worse than "
                      "labelling the whole patch as the class.")
    picture(s, pc, 0.45, 1.1, w=6.3)
    ids = res.common([(m, "1shot") for m in res.models])
    per = {m: res.miou(m, "1shot", ids)[1] for m in res.models}
    base_pc = class_miou(res.baseline(ids))[1]
    classes = sorted(base_pc)
    obs = []
    if len(res.models) >= 2:
        best = {c: max(res.models, key=lambda m: per[m].get(c, 0)) for c in classes}
        for m in res.models:
            wins = [c for c in classes if best[c] == m]
            if wins:
                obs.append(f"**{res.label(m)}** best on {len(wins)}/{len(classes)}: " +
                           ", ".join(CLASS_SHORT[c] for c in wins))
    for m in res.models:
        below = [c for c in classes if per[m].get(c, 0) < base_pc[c]]
        obs.append(f"{res.label(m)} below the predict-all tick on **{len(below)}/{len(classes)}** classes")
    text(s, 6.95, 1.2, 2.55, 3.8, obs, size=9.5, bullets=True, spacing=4)

    # ---------------------------------------------------------------- 12/13 qualitative
    ious = {i: {m: episode_iou(res.runs[(m, "1shot")]["records"][i]) for m in res.models} for i in ids}
    cls_of = {i: res.episodes[i]["class_id"] for i in ids}
    ref = res.runs[(res.models[0], "1shot")]["records"]
    gfrac = {i: ref[i]["gt_area"] / (ref[i]["fg_union"] + ref[i]["bg_inter"]) for i in ids}
    if len(res.models) >= 2:
        picked = select_figure_episodes(ious, cls_of, gfrac, n_total=10, n_disagree=5, n_succeed=3)
        groups = [("Examples — where the models disagree most", picked[:5]),
                  ("Examples — where all models succeed or fail", picked[5:])]
    else:
        order = sorted(ids, key=lambda i: -list(ious[i].values())[0])
        groups = [("Qualitative results — best episodes", [{"id": i, "reason": "best"} for i in order[:5]]),
                  ("Qualitative results — worst episodes", [{"id": i, "reason": "worst"} for i in order[-5:]])]
    for k, (title, sel) in enumerate(groups):
        if not sel:
            continue
        qp = fig_qualitative(res, sel, os.path.join(figdir, f"qualitative_{k + 1}.png"))
        s = D.slide(title, notes="Episodes chosen by a fixed rule after all runs (largest IoU spread; highest "
                                 "worst-model IoU; lowest best-model IoU), not by hand.")
        picture(s, qp, 0.5, 1.07, h=4.13)
        text(s, 7.2, 1.2, 2.3, 3.9, [
            "**How to read**",
            "Magenta: support mask / ground truth.",
            "Green: correct · red: false positive · blue: missed.",
            "Row label: class and support → query domain.",
            "**Selection rule**",
            "Chosen automatically after all runs — not hand-picked."], size=9, spacing=3)

    # ---------------------------------------------------------------- 14 why SAM 3 (data-driven)
    if has3 and has2:
        m3, m2 = res.miou("sam3", "1shot")[0], res.miou("sam2", "1shot")[0]
        bt = paired_bootstrap(res.runs[("sam3", "1shot")]["records"], res.runs[("sam2", "1shot")]["records"])
        wins3 = sum(1 for c in classes if per["sam3"].get(c, 0) > per["sam2"].get(c, 0))
        t3, t2 = res.seconds("sam3", "1shot"), res.seconds("sam2", "1shot")
        sam3_better = bt["significant"] and bt["diff"] > 0
        title = "Why we continue with SAM 3" if sam3_better else "SAM 3 vs SAM 2 — what the numbers say"
        s = D.slide(title, notes="Every number on this slide is computed from the runs; the title follows the result.")
        cards = [("Accuracy", f"{bt['diff']:+.3f}", f"1-shot mIoU vs SAM 2\n95 % CI [{bt['ci_lo']:+.3f}, {bt['ci_hi']:+.3f}]"),
                 ("Classes", f"{wins3}/{len(classes)}", "classes where SAM 3\nbeats SAM 2")]
        if res.has("sam3", "5shot") and res.has("sam2", "5shot") and ids5:
            g3 = res.miou("sam3", "5shot", ids5)[0] - res.miou("sam3", "1shot", ids5)[0]
            cards.append(("More shots", f"{g3:+.3f}", "SAM 3 gain from\n1 → 5 labelled examples"))
        cards.append(("Cost", f"{t3 / max(t2, 1e-6):.1f}×", f"slower than SAM 2\n({t3:.2f} s vs {t2:.2f} s per episode)"))
        wcard = (9.0 - 0.15 * (len(cards) - 1)) / len(cards)
        for k, (head, big, small) in enumerate(cards):
            x = 0.5 + k * (wcard + 0.15)
            box(s, x, 1.25, wcard, 2.55, fill=BOX, line=RULE, lw=0.75)
            text(s, x + 0.2, 1.4, wcard - 0.4, 0.3, f"{k + 1:02d}", size=11, colour="7A9E7E", bold=True)
            text(s, x + 0.2, 1.72, wcard - 0.4, 0.35, head, size=15, bold=True)
            hline(s, x + 0.2, x + wcard - 0.2, 2.2, colour=RULE)
            text(s, x + 0.2, 2.3, wcard - 0.4, 0.6, big, size=30, bold=True, colour=ACCENT if k == 0 else INK)
            text(s, x + 0.2, 3.0, wcard - 0.4, 0.75, small.split("\n"), size=9.5, colour=GREY, spacing=0)
        rec3, rec2 = res.runs[("sam3", "1shot")]["records"], res.runs[("sam2", "1shot")]["records"]
        both = sorted(set(rec3) & set(rec2))
        fallback = class_miou([rec3[i] if rec3[i]["pred_area"] > 0 else rec2[i] for i in both])[0]
        verdict = ("SAM 3 is significantly more accurate on identical episodes and natively accepts image exemplars "
                   "and text — no hand-built prompt pipeline." if sam3_better else
                   "SAM 3 is **not** significantly better than SAM 2 on these episodes; the choice must weigh cost and "
                   "the concept interface, not accuracy alone.")
        text(s, 0.5, 3.98, 9, 1.2, [verdict,
             f"Trade-off: SAM 3 returns **no mask on {100 * res.empty_rate('sam3', '1shot'):.0f} %** of episodes "
             "(instance model, diffuse ‘stuff’ classes).",
             f"Simple fix, no labels needed: use SAM 2 when SAM 3 is empty → **{fallback:.3f}** mIoU "
             f"(SAM 3 alone {m3:.3f}, SAM 2 alone {m2:.3f})."], size=10, bullets=True, spacing=3)

    # ---------------------------------------------------------------- 15 adaptive SAM 3 (proposal; shown
    # only while the training experiment has no report -- otherwise its slides come after the NDVI ones)
    if tr is None:
        s = D.slide("Next: adaptive SAM 3 with an NDVI adapter",
                    notes="Proposed, not yet trained. Numbers on this slide are VegSAM's published results, not ours.")
        text(s, 0.5, 1.15, 3.6, 0.3, "**Inspiration: VegSAM** (Wu, Li & Hong, CVPRW 2026)", size=11)
        text(s, 0.5, 1.5, 3.6, 2.6, [
            "Adapts SAM to multispectral urban-tree segmentation.",
            "NDVI prior injected through a lightweight adapter inside the image encoder; LoRA for parameter-efficient tuning.",
            "Prompt-free decoder fusing four multi-scale feature maps.",
            "Reported: +1.13 / +1.04 IoU over its baseline on Potsdam / Vaihingen."], size=9.5, bullets=True, spacing=3)
        box(s, 4.3, 1.2, 1.2, 0.55, "RGB", size=9.5)
        box(s, 4.3, 2.75, 1.2, 0.62, "NIR + red", sub="→ NDVI", size=9.5)
        box(s, 5.85, 1.1, 1.75, 1.15, "SAM 3 encoder ❄", sub="+ LoRA (trainable)", size=9.5)
        box(s, 5.85, 2.65, 1.75, 0.82, "NDVI adapter", sub="trainable", size=9.5)
        box(s, 7.95, 1.1, 1.55, 1.15, "SAM 3 detector\n+ mask head ❄", size=9.5)
        box(s, 7.95, 2.75, 1.55, 0.62, "Habitat masks", fill=ACCENT, line=ACCENT, colour="FFFFFF", size=9.5)
        arrow(s, 5.5, 1.47, 5.85, 1.6)
        arrow(s, 5.5, 3.06, 5.85, 3.06)
        arrow(s, 6.72, 2.65, 6.72, 2.25)
        arrow(s, 7.6, 1.67, 7.95, 1.67)
        arrow(s, 8.72, 2.25, 8.72, 2.75)
        text(s, 4.3, 3.65, 5.2, 1.3, [
            "Train adapters on **base classes**, evaluate on **novel** classes (rotating folds) — few-shot stays few-shot.",
            "Exemplar prompting (canvas) unchanged, so the 1-shot numbers above are its baseline.",
            "RGB + NIR only → runs on the EU UAV sensor. **Status: next step, not yet trained.**"],
            size=9.5, bullets=True, spacing=3)

    # ---------------------------------------------------------------- 16 SAM + NDVI architecture
    s = D.slide("SAM 3 + NDVI — segment, then measure",
                notes="SAM stays RGB-only and untouched; NIR is only used after segmentation, so every indicator is "
                      "traceable to a segment.")
    box(s, 0.5, 1.3, 1.25, 0.65, "RGB", size=10)
    box(s, 2.1, 1.22, 1.75, 0.8, "SAM 3 few-shot ❄", sub="K support examples", size=10)
    box(s, 4.2, 1.22, 1.55, 0.8, "Habitat segments", size=10)
    box(s, 0.5, 2.55, 1.25, 0.65, "NIR + red", size=10)
    box(s, 2.1, 2.47, 1.75, 0.8, "NDVI map", sub="(NIR − red) / (NIR + red)", size=10)
    box(s, 2.1, 3.6, 1.75, 0.75, "Local Shannon H", sub="7 px = 1.4 m window", size=10)
    box(s, 4.2, 2.35, 1.95, 1.9, "Per-segment\nstatistics", sub="mean NDVI · cover ·\nheterogeneity", size=10)
    box(s, 6.55, 2.75, 1.4, 1.1, "NDVI gate\n(optional)", sub="support range", size=9.5)
    box(s, 8.3, 2.3, 1.2, 2.0, "Habitat map\n+ vigour\n+ diversity", fill=ACCENT, line=ACCENT, colour="FFFFFF", size=10)
    for (x1, y1, x2, y2) in ((1.75, 1.62, 2.1, 1.62), (3.85, 1.62, 4.2, 1.62), (1.75, 2.87, 2.1, 2.87),
                             (4.97, 2.02, 4.97, 2.35), (3.85, 2.87, 4.2, 2.87), (2.97, 3.27, 2.97, 3.6),
                             (3.85, 3.97, 4.2, 3.9), (6.15, 3.3, 6.55, 3.3), (7.95, 3.3, 8.3, 3.3)):
        arrow(s, x1, y1, x2, y2)
    text(s, 0.5, 4.5, 9, 0.7, [
        "Segmentation stays RGB-only: no retraining, SAM 3 untouched. NIR enters only after segmentation, so every "
        "indicator is traceable to a segment.",
        "Optional **NDVI gate**: keep predicted pixels inside the support's NDVI range (2nd–98th pct.) — no query labels used."],
        size=9.5, bullets=True, spacing=2)

    # ---------------------------------------------------------------- 17 SAM + NDVI results
    nd_models = [m for m in res.models if m in res.ndvi]
    if nd_models:
        main = "sam3" if "sam3" in nd_models else nd_models[0]
        nf = fig_ndvi(res, main, os.path.join(figdir, "ndvi_example.png"))
        nc = fig_ndvi_classes(res, os.path.join(figdir, "ndvi_classes.png"))
        s = D.slide("Results — ecology from the segments",
                    notes="All indicators compare the predicted segment with the ground-truth segment of the same "
                          "class on the same query patch (vegetation classes only).")
        picture(s, nf, 0.45, 1.08, w=5.55)
        rows = [["Indicator (vegetation)"] + [MODEL_NAME[m] for m in nd_models]]
        rows.append(["NDVI of segment, |error|"] + [fmt(res.ndvi[m]["ndvi_abs_err_mean"]) for m in nd_models])
        rows.append(["NDVI of segment, r"] + [fmt(res.ndvi[m]["ndvi_r"], 2) for m in nd_models])
        rows.append(["Cover, |error|"] + [f"{res.ndvi[m]['cover_abs_err_pp']:.1f} pp" for m in nd_models])
        rows.append(["Local heterogeneity H, r"] + [fmt(res.ndvi[m]["H_r"], 2) for m in nd_models])
        rows.append(["mIoU → with NDVI gate"] + [f"{res.ndvi[m]['gate']['miou_raw']:.3f} → {res.ndvi[m]['gate']['miou_gated']:.3f}"
                                                  for m in nd_models])
        table(s, 6.15, 1.12, [1.6] + [0.9] * len(nd_models), rows, size=8.5, row_h=0.27)
        picture(s, nc, 6.2, 2.85, w=3.3)
        g = res.ndvi[main]["gate"]
        verdict = ("The NDVI gate **hurts** ({:.3f} → {:.3f}): an NDVI range from a support in another region/date "
                   "does not transfer.".format(g["miou_raw"], g["miou_gated"]) if g["miou_gated"] < g["miou_raw"] else
                   "The NDVI gate helps ({:.3f} → {:.3f}).".format(g["miou_raw"], g["miou_gated"]))
        text(s, 0.5, 3.85, 5.5, 1.3, [verdict,
             "Segment NDVI tracks the ground truth well enough to rank habitats; cover is limited by mask quality."],
             size=9.5, bullets=True, spacing=3)

    # ---------------------------------------------------------------- Experiment 3: SAM 3 trained with NIR
    if tr is not None:
        st = tr.setup
        s = D.slide("Experiment 3 — training SAM 3 with NIR",
                    notes="SAM 3 fine-tuned twice, identically, once with RGB and once with RGB + NIR as input. The only "
                          "difference between the two arms is the NIR channel.")
        box(s, 0.5, 2.05, 1.25, 0.75, "R G B + NIR", sub="4 bands, 0.2 m", size=9.5)
        box(s, 2.05, 1.95, 1.55, 0.95, "Patch embedding\n3 → 4 channels", sub="NIR filters start at 0", size=9)
        box(s, 3.9, 1.85, 1.6, 1.15, "SAM 3 encoder ❄", sub="+ LoRA r = 16 (trained)", size=9.5)
        box(s, 3.9, 1.12, 1.6, 0.5, "Class phrase", sub="“conifer tree”", size=8.5)
        box(s, 5.8, 1.95, 1.45, 0.95, "Fusion encoder", sub="trained", size=9.5)
        box(s, 7.55, 1.95, 1.95, 0.95, "Semantic head → 12\nclass maps → argmax", fill=ACCENT, line=ACCENT,
            colour="FFFFFF", size=9)
        for (x1, y1, x2, y2) in ((1.75, 2.42, 2.05, 2.42), (3.6, 2.42, 3.9, 2.42), (5.5, 2.42, 5.8, 2.42),
                                 (7.25, 2.42, 7.55, 2.42), (5.5, 1.37, 6.2, 1.95)):
            arrow(s, x1, y1, x2, y2)
        text(s, 0.5, 3.25, 9.0, 1.85, [
            "**Control arm:** the same run with RGB only — same images, order, augmentations, prompts and seeds. "
            "The NIR channel is the **only** difference.",
            "NIR filters start at zero, so at step 0 the RGB + NIR model **is** pretrained SAM 3; training decides "
            "how much NIR to use.",
            f"Official FLAIR-HUB split: **{setup_text(st, 'n_train')}** training patches "
            f"({setup_text(st, 'n_train_domains')} domains) · **{setup_text(st, 'n_test')}** test patches from "
            f"**{setup_text(st, 'n_test_domains')} unseen domains**.",
            f"{setup_text(st, 'steps')} steps × batch {setup_text(st, 'effective_batch')} · 3 seeds per arm · "
            "16.5 M of 841 M parameters trained · shared 16 GB GPUs.",
            "Related: VegSAM (Wu, Li & Hong, 2026) injects NDVI through an adapter; we give SAM the raw NIR band."],
            size=9.5, bullets=True, spacing=3)

        m_rgb, sd_rgb, n_rgb = tr.stats("rgb")
        m_nir, sd_nir, n_nir = tr.stats("rgbi")
        zs = tr.zero_shot()
        d, (lo, hi) = tr.d["delta"], tr.d["ci95"]
        dp = os.path.join(figdir, "train_delta.png")
        fig_train_delta(tr, dp)
        s = D.slide("Experiment 3 — does NIR help?",
                    notes="Paired comparison: same seeds, same test images. CI from resampling the test images.")
        rows = [["Model", "mIoU", "per seed"]]
        if zs is not None:
            rows.append(["SAM 3 zero-shot (text)", fmt(zs), "—"])
        rows.append([f"Fine-tuned, RGB", f"{m_rgb:.4f} ± {sd_rgb:.4f}", ", ".join(f"{v:.3f}" for v in tr.mious("rgb"))])
        rows.append([f"Fine-tuned, RGB + NIR", f"{m_nir:.4f} ± {sd_nir:.4f}", ", ".join(f"{v:.3f}" for v in tr.mious("rgbi"))])
        table(s, 0.5, 1.15, [1.6, 1.3, 1.5], rows, size=9, row_h=0.28, bold_cells={(len(rows) - 1, 0), (len(rows) - 1, 1)})
        box(s, 0.5, 2.45, 4.4, 1.05, fill=BOX, line=RULE, lw=0.75)
        text(s, 0.65, 2.52, 4.1, 0.3, "Effect of NIR (RGB + NIR − RGB)", size=10, colour=GREY)
        text(s, 0.65, 2.78, 1.9, 0.6, f"**{d:+.4f}**", size=24, colour=ACCENT if tr.d["significant"] else INK)
        text(s, 2.55, 2.85, 2.3, 0.6, [f"95 % CI [{lo:+.4f}, {hi:+.4f}]",
             "every seed: " + ", ".join(f"{v:+.3f}" for v in tr.d["per_seed"])], size=8.5, spacing=0)
        sig = [r for r in tr.per_class if r["significant"]]
        up = sorted([r for r in sig if r["delta_pooled"] > 0], key=lambda r: -r["delta_pooled"])
        down = sorted([r for r in sig if r["delta_pooled"] < 0], key=lambda r: r["delta_pooled"])
        obs = []
        if zs is not None:
            obs.append(f"Training is the big step: **{zs:.3f} → {m_rgb:.3f}** mIoU on unseen domains.")
        obs.append(f"NIR improves **{len(up)} of {len(tr.per_class)}** classes significantly; largest: "
                   + ", ".join(f"{CLASS_SHORT[int(r['class_id'])].lower()} {r['delta_pooled']:+.3f}" for r in up[:3]) + ".")
        if down:
            r = down[0]
            obs.append(f"Largest significant loss: {CLASS_SHORT[int(r['class_id'])].lower()} {r['delta_pooled']:+.3f}.")
        other = getattr(tr, "other", None)
        if other is not None:
            o_rgb, o_nir = other.stats("rgb")[0], other.stats("rgbi")[0]
            span = (f"{other.setup['steps']:,} → {st['steps']:,} steps" if other.setup.get("steps") and st.get("steps")
                    else "Shorter runs → these runs")
            obs.append((f"{span}: RGB {o_rgb:.3f} → {m_rgb:.3f}, RGB + NIR {o_nir:.3f} → {m_nir:.3f}; "
                       f"NIR gain **{other.d['delta']:+.4f} → {d:+.4f}** "
                       + ("(unchanged)." if abs(d - other.d["delta"]) < 0.002 else
                          ("(grew)." if d > other.d["delta"] else "(shrank)."))))
        text(s, 0.5, 3.65, 4.4, 1.5, obs, size=9.5, bullets=True, spacing=3)
        picture(s, dp, 5.1, 1.1, w=4.4)
        text(s, 5.1, 4.35, 4.4, 0.6, "Green: significant gain · dark grey: significant loss · light grey: not "
             "significant. Error bars: 95 % CI (bootstrap over test images, seeds pooled).", size=8, colour=GREY,
             italic=True)

        helped, mh = tr.fig("NIR helped")
        hurt, mu = tr.fig("NIR hurt")
        if helped and hurt:
            s = D.slide("Experiment 3 — where NIR helps and hurts",
                        notes="Test patches chosen by a fixed rule after all runs (largest difference among patches with "
                              "at least three classes), seed 0. Illustrations; the evidence is the aggregate result.")
            picture(s, helped, 0.5, 1.05, w=9.0)
            picture(s, hurt, 0.5, 2.95, w=9.0)
            text(s, 0.5, 4.83, 9.0, 0.35,
                 f"Top: the test patch where NIR helped most (per-patch mIoU {mh['miou_rgb']:.2f} → {mh['miou_rgbi']:.2f}). "
                 f"Bottom: where it hurt most ({mu['miou_rgb']:.2f} → {mu['miou_rgbi']:.2f}). Chosen by rule among "
                 "multi-class patches; single patches exaggerate — the evidence is the aggregate.",
                 size=8, colour=GREY, italic=True)

    # ---------------------------------------------------------------- 18 conclusion
    s = D.slide("Conclusion", notes="Summary of what the evidence supports, and what it does not.")
    concl = [f"**FLAIR** made the ecological question testable: NIR, 0.2 m, European landscapes, {doms} domains in our set."]
    if res.models:
        parts = ", ".join(f"{res.label(m)} {res.miou(m, '1shot')[0]:.3f}" for m in res.models)
        concl.append(f"On identical cross-domain episodes, 1-shot mIoU: **{parts}** (predict-all {base_all:.3f}).")
    if has2 and has3:
        bt = paired_bootstrap(res.runs[("sam3", "1shot")]["records"], res.runs[("sam2", "1shot")]["records"])
        concl.append(f"SAM 3 − SAM 2 = {bt['diff']:+.3f} [{bt['ci_lo']:+.3f}, {bt['ci_hi']:+.3f}] — "
                     + ("significant." if bt["significant"] else "not significant."))
    for m in res.models:
        v = res.miou(m, "1shot")[0]
        if v <= base_all:
            concl.append(f"{res.label(m)}'s prototype prompting does **not** beat predicting the whole patch "
                         f"({v:.3f} vs {base_all:.3f}); its decoder is fine (oracle), the cross-domain matching is not.")
    if has2 and has3:
        r3, r2 = res.runs[("sam3", "1shot")]["records"], res.runs[("sam2", "1shot")]["records"]
        both = sorted(set(r3) & set(r2))
        fb = class_miou([r3[i] if r3[i]["pred_area"] > 0 else r2[i] for i in both])[0]
        concl.append(f"SAM 3 returns nothing on {100 * res.empty_rate('sam3', '1shot'):.0f} % of episodes; falling back "
                     f"to SAM 2 there gives **{fb:.3f}** — no labels, no tuning.")
    concl.append("Diffuse ‘stuff’ classes (pervious surface, brushwood) stay hard for every model.")
    if nd_models:
        g = res.ndvi[main]["gate"]
        concl.append(f"NIR **after** segmentation measures ecology well (segment NDVI r = {res.ndvi[main]['ndvi_r']:.2f}), "
                     f"but a support-derived NDVI gate {'hurts' if g['miou_gated'] < g['miou_raw'] else 'helps'} "
                     f"({g['miou_raw']:.3f} → {g['miou_gated']:.3f}).")
    if tr is not None:
        m_rgb, m_nir = tr.stats("rgb")[0], tr.stats("rgbi")[0]
        top = max(tr.per_class, key=lambda r: r["delta_pooled"])
        zs = tr.zero_shot()
        concl.append((f"Fine-tuning SAM 3 lifts it from {zs:.3f} to **{m_rgb:.3f}** on unseen domains; " if zs is not None
                      else f"Fine-tuned SAM 3 reaches **{m_rgb:.3f}** on unseen domains; ")
                     + f"NIR **inside** the trained model adds **{tr.d['delta']:+.4f}** "
                     f"[{tr.d['ci95'][0]:+.4f}, {tr.d['ci95'][1]:+.4f}] — small but consistent, mostly vegetation "
                     f"({CLASS_SHORT[int(top['class_id'])].lower()} {top['delta_pooled']:+.3f}).")
    concl.append("Everything uses RGB + NIR only — no elevation — so it transfers to the EU UAV sensor.")
    if pending:
        concl.append(f"Pending: {', '.join(pending)} (cluster run)" + ("." if tr is not None else
                     ", full FLAIR test domains, adaptive SAM 3."))
    text(s, 0.5, 1.2, 9, 3.9, concl, size=11.5, bullets=True, spacing=7)

    # ---------------------------------------------------------------- 19 future work
    s = D.slide("Future work — last phase: the EU UAV data",
                notes="The pipeline is dataset-agnostic: new data needs a loader, a band check and an episode file.")
    steps = [("1", "Ingest + verify", "PT · DE · UK · AT;\nband-order check"),
             ("2", "Few-shot SAM 3", "K ∈ {1, 5} labelled\ntiles per habitat"),
             (("3", "Fine-tune with NIR", "FLAIR-trained SAM 3\n+ few EU labels") if tr is not None else
              ("3", "Adaptive SAM 3", "NDVI adapter + LoRA,\nbase → novel classes")),
             ("4", "Ecological read-out", "NDVI, cover, Shannon\nper habitat + country")]
    for k, (num, head, sub) in enumerate(steps):
        x = 0.5 + k * 2.3
        box(s, x, 1.3, 2.0, 1.25, f"{num}  {head}", sub=sub, size=11, fill=ACCENT if k == 3 else BOX,
            line=ACCENT if k == 3 else BOXLINE, colour="FFFFFF" if k == 3 else INK)
        if k:
            arrow(s, x - 0.3, 1.92, x, 1.92)
    text(s, 0.5, 2.9, 4.4, 0.3, "**Open questions**", size=12)
    text(s, 0.5, 3.25, 4.4, 1.9, [
        "Do FLAIR supports transfer to EU habitats, or must supports come from the same country?",
        "How different is the UAV NIR response from FLAIR's (NDVI calibration)?",
        "Sites without NIR: RGB-only fallback."], size=10, bullets=True, spacing=3)
    text(s, 5.1, 2.9, 4.4, 0.3, "**Risks**", size=12)
    text(s, 5.1, 3.25, 4.4, 1.9, [
        "Data access for the EU UAV campaigns is still pending.",
        "Sparse labels limit the number of evaluation episodes.",
        "Compute: the cluster GPUs are shared; fine-tuning runs are queued onto whatever GPU memory is free."
        if tr is not None else "SAM 3 compute: ~1 s per episode on a 16 GB GPU is feasible, 5-shot is 5×."],
        size=10, bullets=True, spacing=3)

    # ---------------------------------------------------------------- 20 references
    s = D.slide("References", notes="")
    text(s, 0.5, 1.15, 9, 4.0, [r for r in [
        "Kirillov et al. Segment Anything. ICCV 2023.",
        "Ravi et al. SAM 2: Segment Anything in Images and Videos. arXiv:2408.00714, 2024.",
        "Carion et al. SAM 3: Segment Anything with Concepts. arXiv:2511.16719, 2025.",
        "Tsai, Lin & Wang. Few-Shot Semantic Segmentation Meets SAM3. arXiv:2604.05433, 2026.",
        "Zhang et al. Personalize Segment Anything Model with One Shot (PerSAM). arXiv:2305.03048, 2023.",
        "Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022 (arXiv:2106.09685)."  if tr is not None else "",
        "Wu, Li & Hong. VegSAM: Vegetation-aware Adapter for Segment Anything Model in Urban Tree Segmentation. CVPR Workshops (MORSE) 2026.",
        "Garioud et al. FLAIR #1: semantic segmentation and domain adaptation dataset. arXiv:2211.12979.",
        "IGN. FLAIR-HUB (github.com/IGNF/FLAIR-HUB); FLAIR #2: arXiv:2305.14467.",
        "Wang et al. LoveDA: A Remote Sensing Land-Cover Dataset for Domain Adaptive Semantic Segmentation. NeurIPS Datasets & Benchmarks 2021.",
        "Wang et al. PANet: Few-Shot Image Semantic Segmentation with Prototype Alignment. ICCV 2019.",
        "Rouse et al. Monitoring vegetation systems in the Great Plains with ERTS. 1974 (NDVI).",
        "Shannon. A Mathematical Theory of Communication. Bell System Technical Journal, 1948."] if r],
        size=9.5, spacing=4)

    D.finish(out)
    return len(D.slides)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data")
    ap.add_argument("--episodes", default="outputs/episodes.json")
    ap.add_argument("--runs_dir", default="outputs/runs")
    ap.add_argument("--ndvi_dir", default="outputs/ndvi")
    ap.add_argument("--train_dir", default="outputs/train", help="run_train.sh outputs (report/ inside)")
    ap.add_argument("--train_main_report", default="outputs/long_report",
                    help="report to present as the main training result if it exists (the 12,000-step runs); "
                         "the --train_dir report is then shown as the shorter-training comparison")
    ap.add_argument("--out", default="presentation/out/SRP_Group7_Final.pptx")
    args = ap.parse_args()
    res = Results(args.root, args.episodes, args.runs_dir, args.ndvi_dir)
    if not res.models:
        sys.exit("No finished 1-shot runs matching the episodes file; run `bash run.sh` first.")
    print("models:", res.models, "| arms:", sorted(res.runs))
    tr = TrainResults(os.path.join(args.train_dir, "report"), args.train_dir)
    main_tr = TrainResults(args.train_main_report, args.train_dir) if args.train_main_report else None
    if main_tr is not None and main_tr.ok:
        if tr.ok:
            main_tr.other = tr
        tr = main_tr
        print(f"training experiment: main result from {args.train_main_report}")
    print("training experiment:", "included" if tr.ok else f"no report in {args.train_dir}/report (proposal slide instead)")
    if tr.ok and "n_train" not in tr.setup:
        print(f"  note: training-set size unknown (copy {args.train_dir}/split.json from the cluster, or re-run the report)")
    n = build(res, os.path.join(os.path.dirname(os.path.abspath(args.out)), "fig"), args.out, tr)
    print(f"wrote {args.out} ({n} slides)")


if __name__ == "__main__":
    main()
