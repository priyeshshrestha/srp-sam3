"""Multi-class segmentation metrics for the trained models.

Evaluated pixels: ground truth in classes 1..12 (FLAIR's 12 evaluated classes). Pixels of classes
13..19 are ignored, as in the FLAIR #1 12-class setting. The prediction is always one of 1..12.
"""

from typing import Dict, Optional

import numpy as np

from train.sam3_rgbi import CLASSES

K = len(CLASSES)


def confusion(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """(K, K) int64, rows = ground truth, cols = prediction, over pixels with gt in 1..12."""
    keep = (gt >= 1) & (gt <= K)
    g = gt[keep].astype(np.int64) - 1
    p = np.clip(pred[keep].astype(np.int64), 1, K) - 1
    return np.bincount(g * K + p, minlength=K * K).reshape(K, K)


def iou_per_class(conf: np.ndarray) -> np.ndarray:
    """IoU per class; NaN for a class absent from both ground truth and prediction."""
    tp = np.diag(conf).astype(np.float64)
    denom = conf.sum(0) + conf.sum(1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, tp / denom, np.nan)


def miou(conf: np.ndarray) -> float:
    """Mean IoU over the classes that occur in the ground truth of the evaluated set."""
    iou = iou_per_class(conf)
    present = conf.sum(1) > 0
    return float(np.nanmean(iou[present])) if present.any() else float("nan")


def summary(conf: np.ndarray) -> Dict:
    iou = iou_per_class(conf)
    return {"miou": miou(conf), "pixel_acc": float(np.trace(conf) / max(1, conf.sum())),
            "iou_per_class": {int(c): (None if np.isnan(v) else float(v)) for c, v in zip(CLASSES, iou)},
            "gt_pixels_per_class": {int(c): int(n) for c, n in zip(CLASSES, conf.sum(1))}}


def image_miou(conf_i: np.ndarray) -> float:
    """mIoU of one image over the classes in its ground truth (for figure selection only)."""
    return miou(conf_i)


def paired_bootstrap_delta(conf_a: np.ndarray, conf_b: np.ndarray, n_boot: int = 1000,
                           seed: int = 0) -> Dict:
    """mIoU(b) - mIoU(a) with a 95% CI, resampling TEST IMAGES (same resample for both).

    conf_a / conf_b: (N, K, K) per-image confusions of two models on the same N images. To pool
    several seeds, pass the per-image confusions SUMMED over seeds (still paired by image)."""
    if conf_a.shape != conf_b.shape:
        raise ValueError("paired bootstrap needs the same images for both models")
    n = conf_a.shape[0]
    rng = np.random.default_rng(seed)
    point = miou(conf_b.sum(0)) - miou(conf_a.sum(0))
    point_cls = iou_per_class(conf_b.sum(0)) - iou_per_class(conf_a.sum(0))
    deltas = np.empty(n_boot)
    deltas_cls = np.empty((n_boot, conf_a.shape[1]))
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        cnt = np.bincount(idx, minlength=n)
        a = np.tensordot(cnt, conf_a, axes=1)
        b = np.tensordot(cnt, conf_b, axes=1)
        deltas[i] = miou(b) - miou(a)
        deltas_cls[i] = iou_per_class(b) - iou_per_class(a)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    with np.errstate(invalid="ignore"):
        lo_c, hi_c = np.nanpercentile(deltas_cls, [2.5, 97.5], axis=0)
    per_class = {int(c): {"delta": None if np.isnan(d) else float(d), "ci95": [float(l), float(h)],
                          "significant": bool(l > 0 or h < 0)}
                 for c, d, l, h in zip(CLASSES, point_cls, lo_c, hi_c)}
    return {"delta": float(point), "ci95": [float(lo), float(hi)], "n_images": int(n), "n_boot": n_boot,
            "significant": bool(lo > 0 or hi < 0), "per_class": per_class}
