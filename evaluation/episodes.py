"""Model-independent episode generation.

An episode is (class, support patch, query patch). It is drawn ONCE, written to JSON, and every
model then runs the identical list, so model differences are paired and nothing about one
model's prompt format can decide which episodes exist.

Eligibility depends only on the data:
  - the class must cover at least `min_frac` of BOTH support and query (specks make IoU noise)
  - and at most `max_frac` of both (a class that fills the frame makes 'predict everything'
    score ~1.0, which is how the project once reported a meaningless 51.64% IoU)
  - support and query are different patches
  - domain_mode 'cross' requires different FLAIR domains, 'same' requires the same domain,
    'any' does not care. Cross-domain tests geographic/seasonal transfer; same-domain can be
    flattered by near-identical neighbouring patches.
"""

import hashlib
import json
from typing import Dict, List, Optional, Sequence

import numpy as np

DOMAIN_MODES = ("cross", "same", "any")
PATCH_PIXELS = 512 * 512


def eligible_patches(counts: Dict[str, Dict[int, int]], class_id: int,
                     min_frac: float, max_frac: float, patch_pixels: int = PATCH_PIXELS) -> List[str]:
    lo, hi = min_frac * patch_pixels, max_frac * patch_pixels
    return sorted(p for p, c in counts.items() if lo <= c.get(class_id, 0) <= hi)


def make_episodes(
    counts: Dict[str, Dict[int, int]],
    domains: Dict[str, str],
    classes: Sequence[int],
    n_per_class: int,
    seed: int,
    domain_mode: str = "cross",
    min_frac: float = 0.02,
    max_frac: float = 0.90,
    patch_pixels: int = PATCH_PIXELS,
    k_shot_max: int = 1,
) -> Dict:
    """Returns {'episodes': [...], 'skipped': {class_id: reason}}. Deterministic in `seed`.

    Every episode carries up to `k_shot_max` distinct supports. K-shot evaluation uses supports[:K]
    (skipping episodes with fewer than K), so the 1-shot support is also the first 5-shot support
    and the two settings stay paired.
    """
    if domain_mode not in DOMAIN_MODES:
        raise ValueError(f"domain_mode must be one of {DOMAIN_MODES}")
    rng = np.random.default_rng(seed)
    episodes, skipped = [], {}

    for c in classes:
        pool = eligible_patches(counts, c, min_frac, max_frac, patch_pixels)
        if len(pool) < 2:
            skipped[int(c)] = f"only {len(pool)} eligible patch(es)"
            continue

        made, tries = 0, 0
        while made < n_per_class and tries < n_per_class * 50:
            tries += 1
            q = pool[int(rng.integers(len(pool)))]
            cands = [s for s in pool if s != q and (
                domain_mode == "any"
                or (domain_mode == "cross" and domains[s] != domains[q])
                or (domain_mode == "same" and domains[s] == domains[q]))]
            if not cands:
                continue
            # Up to k_shot_max supports; K-shot runs skip episodes that hold fewer than K.
            picks = rng.choice(len(cands), size=min(k_shot_max, len(cands)), replace=False)
            supports = [cands[int(i)] for i in picks]
            s = supports[0]
            episodes.append({
                "id": len(episodes), "class_id": int(c),
                "support": s, "supports": supports, "query": q,
                "support_domain": domains[s], "query_domain": domains[q],
            })
            made += 1
        if made == 0:
            skipped[int(c)] = f"no valid support/query pair under domain_mode={domain_mode!r}"
        elif made < n_per_class:
            skipped[int(c)] = f"only {made}/{n_per_class} episodes could be drawn"
    return {"episodes": episodes, "skipped": skipped}


def save_episodes(path: str, payload: Dict, meta: Dict) -> None:
    with open(path, "w") as fh:
        json.dump({"meta": meta, **payload}, fh, indent=1)


def load_episodes(path: str) -> Dict:
    with open(path) as fh:
        return json.load(fh)


def episodes_hash(episodes: Sequence[Dict]) -> str:
    """Stable fingerprint of an episode list. Runs record it so results from different episode
    files can never be silently combined (episode ids would point at different images)."""
    blob = json.dumps([[e["id"], e["class_id"], e.get("supports", [e["support"]]), e["query"]]
                       for e in episodes])
    return hashlib.sha1(blob.encode()).hexdigest()[:12]
