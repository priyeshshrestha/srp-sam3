"""Segmentation metrics and paired statistics for the few-shot comparison."""

from typing import Dict, List, Sequence, Tuple

import numpy as np


def confusion(pred: np.ndarray, gt: np.ndarray) -> Dict[str, int]:
    """Foreground and background intersection/union for one binary episode."""
    p, g = pred.astype(bool), gt.astype(bool)
    return {
        "fg_inter": int((p & g).sum()), "fg_union": int((p | g).sum()),
        "bg_inter": int((~p & ~g).sum()), "bg_union": int((~p | ~g).sum()),
        "pred_area": int(p.sum()), "gt_area": int(g.sum()),
    }


def episode_iou(rec: Dict) -> float:
    """IoU of one episode. Empty prediction AND empty GT cannot occur (GT >= min_frac)."""
    return rec["fg_inter"] / rec["fg_union"] if rec["fg_union"] > 0 else 0.0


def class_miou(records: Sequence[Dict], accumulate: bool = True) -> Tuple[float, Dict[int, float]]:
    """mIoU over classes.

    accumulate=True is the standard few-shot convention (HSNet/PFENet style): sum intersection
    and union over every episode of a class, divide once, then average classes. It weights large
    episodes more. accumulate=False averages per-episode IoUs within a class first.
    """
    by_class: Dict[int, List[Dict]] = {}
    for r in records:
        by_class.setdefault(r["class_id"], []).append(r)
    per_class = {}
    for c, rs in by_class.items():
        if accumulate:
            u = sum(r["fg_union"] for r in rs)
            per_class[c] = sum(r["fg_inter"] for r in rs) / u if u else 0.0
        else:
            per_class[c] = float(np.mean([episode_iou(r) for r in rs]))
    return (float(np.mean(list(per_class.values()))) if per_class else 0.0), per_class


def fb_iou(records: Sequence[Dict]) -> float:
    """Mean of pooled foreground IoU and pooled background IoU."""
    fg_u = sum(r["fg_union"] for r in records)
    bg_u = sum(r["bg_union"] for r in records)
    if not fg_u or not bg_u:
        return 0.0
    fg = sum(r["fg_inter"] for r in records) / fg_u
    bg = sum(r["bg_inter"] for r in records) / bg_u
    return (fg + bg) / 2.0


def baseline_record(gt: np.ndarray, kind: str) -> Dict:
    """'all_fg' predicts the whole patch as the class, 'empty' predicts nothing."""
    pred = np.ones_like(gt, dtype=bool) if kind == "all_fg" else np.zeros_like(gt, dtype=bool)
    return confusion(pred, gt)


def paired_bootstrap(
    records_a: Dict[int, Dict], records_b: Dict[int, Dict],
    n_boot: int = 2000, seed: int = 0, accumulate: bool = True,
) -> Dict[str, float]:
    """CI on mIoU(a) - mIoU(b) over episodes both models ran.

    Resamples episodes WITHIN each class (keeping class balance), recomputes mIoU for both
    models on the same resample, and reports the 2.5/97.5 percentiles of the difference.
    """
    ids = sorted(set(records_a) & set(records_b))
    by_class: Dict[int, List[int]] = {}
    for i in ids:
        by_class.setdefault(records_a[i]["class_id"], []).append(i)
    rng = np.random.default_rng(seed)

    def miou(recs, sample):
        return class_miou([recs[i] for i in sample], accumulate)[0]

    base = miou(records_a, ids) - miou(records_b, ids)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        sample = []
        for c, members in by_class.items():
            sample.extend(rng.choice(members, size=len(members), replace=True).tolist())
        diffs[b] = miou(records_a, sample) - miou(records_b, sample)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(base), "ci_lo": float(lo), "ci_hi": float(hi),
            "n_episodes": len(ids), "significant": bool(lo > 0 or hi < 0)}
