"""Pure helpers for scripts/report.py: loading runs and choosing which episodes to draw."""

import json
import os
from typing import Dict, List, Sequence

import numpy as np


def load_run(run_dir: str) -> Dict:
    recs = {}
    with open(os.path.join(run_dir, "results.jsonl")) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                recs[r["id"]] = r
    meta_path = os.path.join(run_dir, "run_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    return {"name": os.path.basename(os.path.normpath(run_dir)), "records": recs, "meta": meta,
            "dir": run_dir}


def load_pred(run_dir: str, episode_id: int) -> np.ndarray:
    z = np.load(os.path.join(run_dir, "preds", f"{episode_id:06d}.npz"))
    shape = tuple(int(v) for v in z["shape"])
    return np.unpackbits(z["packed"])[: int(np.prod(shape))].reshape(shape).astype(bool)


def select_figure_episodes(
    ious: Dict[int, Dict[str, float]],   # episode id -> {model name: iou}
    class_of: Dict[int, int],
    gt_frac: Dict[int, float],           # episode id -> GT coverage of the query patch
    n_total: int = 10,
    n_disagree: int = 5,
    n_succeed: int = 3,
    min_gt_frac_fail: float = 0.05,
) -> List[Dict]:
    """Pick episodes by a fixed rule, applied AFTER all models ran (never by eyeballing).

      disagree : largest IoU spread between best and worst model, at most one per class
      succeed  : highest worst-model IoU (all models got it), unused classes preferred
      fail     : lowest best-model IoU among episodes whose class covers >= min_gt_frac_fail
                 of the query, so a tiny target cannot masquerade as a model failure
    Returns [{'id', 'reason'}] in that order, no duplicates, up to n_total.
    """
    ids = sorted(ious)
    chosen, used_ids, used_classes = [], set(), set()

    def take(ranked: Sequence[int], reason: str, limit: int, distinct_class: bool):
        n = 0
        for i in ranked:
            if n >= limit or len(chosen) >= n_total:
                break
            if i in used_ids or (distinct_class and class_of[i] in used_classes):
                continue
            chosen.append({"id": i, "reason": reason})
            used_ids.add(i)
            used_classes.add(class_of[i])
            n += 1

    spread = sorted(ids, key=lambda i: -(max(ious[i].values()) - min(ious[i].values())))
    take(spread, "largest disagreement between models", n_disagree, True)

    succeed = sorted(ids, key=lambda i: -min(ious[i].values()))
    take(succeed, "all models succeed (highest worst-model IoU)", n_succeed, True)

    n_fail = n_total - len(chosen)
    fail_pool = [i for i in ids if gt_frac[i] >= min_gt_frac_fail]
    fail = sorted(fail_pool, key=lambda i: max(ious[i].values()))
    take(fail, "all models fail (lowest best-model IoU)", n_fail, False)
    return chosen
