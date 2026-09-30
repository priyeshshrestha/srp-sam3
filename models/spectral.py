"""Where spectral information enters a frozen RGB foundation model.

The measured result on FLAIR (1-shot, SAM ViT-B, prototype-cosine prompting):

    rgb          mIoU 0.2860    vegetation 0.348    built/bare 0.227
    cir          mIoU 0.2535    vegetation 0.347    built/bare 0.160
    ndvi_dsm_h   mIoU 0.2442    vegetation 0.337    built/bare 0.146

Read the columns, not the totals. On vegetation the three composites are tied. RGB's win
comes entirely from classes defined by albedo -- impervious surface 0.340 -> 0.124, plowed
land 0.374 -> 0.229 -- where substituting NIR for blue destroys the very signal that
separates them. So input substitution is not "using NIR", it is *trading away* the colour
priors the encoder was pretrained on, and on this evidence the trade is bad.

That is an argument about the INSERTION POINT, not about NIR. Everything in this module
inserts spectral evidence AFTER the frozen encoder, where no pretraining statistics are
violated and the RGB pathway stays intact:

    modulate   blend ecological plausibility into the similarity map, which is what
               generates the prompts in a PerSAM-style pipeline
    prompt     pick the positive/negative prompt points from the blended map
    dual       encode RGB and CIR separately and fuse their similarity maps

Both blends below are CONVEX and parameterised so that one end of the range reproduces the
RGB baseline exactly (`weight=0`, `alpha=1`). Tuned on held-out domains, neither can do
worse than the baseline -- which is the specific failure of input substitution they are
designed to avoid.
"""

from typing import Dict, Optional, Sequence

import numpy as np

INSERTIONS = ("none", "modulate", "prompt", "dual")


def normalise_map(sim: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Min-max a similarity map to [0, 1].

    Cosine similarity is in [-1, 1] but observed maps occupy a narrow, image-dependent band
    (roughly 0.45-0.90 in practice). Blending raw cosines against a [0, 1] plausibility would
    let the arbitrary offset dominate, so both operands are put on a common scale first.
    """
    s = np.asarray(sim, dtype=np.float64)
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < eps:
        return np.zeros_like(s)
    return (s - lo) / (hi - lo)


def reference_stats(
    mask: np.ndarray,
    ndvi: Optional[np.ndarray] = None,
    shannon: Optional[np.ndarray] = None,
    elevation: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Summarise what the target class looks like spectrally, from the SUPPORT only.

    Returns mean and standard deviation per available signal. The standard deviation matters
    as much as the mean: a class with broad natural variation (agricultural land across
    growth stages) should not be penalised for it, while a tight class (water) should be.

    Never call this on the query's ground truth -- that is the oracle mistake that made the
    predecessor's headline figure meaningless.
    """
    m = np.asarray(mask).astype(bool)
    stats: Dict[str, float] = {}
    if not m.any():
        return stats
    for name, signal in (("ndvi", ndvi), ("shannon", shannon), ("elevation", elevation)):
        if signal is None:
            continue
        values = np.asarray(signal, dtype=np.float64)[m]
        if values.size == 0:
            continue
        stats[f"{name}_mean"] = float(values.mean())
        # Floor the spread so a degenerate support patch cannot produce a delta function.
        stats[f"{name}_std"] = float(max(values.std(), 0.02))
    return stats


def ecological_plausibility(
    reference: Dict[str, float],
    ndvi: Optional[np.ndarray] = None,
    shannon: Optional[np.ndarray] = None,
    elevation: Optional[np.ndarray] = None,
    tolerance: float = 2.0,
) -> Optional[np.ndarray]:
    """Per-pixel plausibility in [0, 1] that a query pixel belongs to the reference class.

    A Gaussian kernel on the standardised deviation from the support's statistics, averaged
    over whichever signals are available. 1.0 means "spectrally indistinguishable from the
    support", 0.0 means "nothing like it".

    Args:
        reference: output of `reference_stats`, computed on the support
        tolerance: multiplies the support's spread. Larger is more permissive; 2.0 means a
            pixel two standard deviations from the support mean still scores ~0.6.

    Returns None when no signal is available, so callers can fall back cleanly rather than
    silently multiplying by an all-ones map.
    """
    if not reference:
        return None

    fields = []
    for name, signal in (("ndvi", ndvi), ("shannon", shannon), ("elevation", elevation)):
        mean_key, std_key = f"{name}_mean", f"{name}_std"
        if signal is None or mean_key not in reference:
            continue
        arr = np.asarray(signal, dtype=np.float64)
        z = (arr - reference[mean_key]) / (reference[std_key] * tolerance)
        fields.append(np.exp(-0.5 * z * z))

    if not fields:
        return None
    return np.clip(np.mean(fields, axis=0), 0.0, 1.0)


def modulate_similarity(
    sim_map: np.ndarray,
    plausibility: Optional[np.ndarray],
    weight: float = 0.3,
) -> np.ndarray:
    """Blend ecological plausibility into a similarity map.

        S' = (1 - w) * norm(S) + w * plausibility

    A convex combination, so `weight=0` reproduces the unmodified (normalised) map exactly
    and the result stays in [0, 1]. Sweeping w on held-out domains therefore has a guaranteed
    floor at the RGB baseline -- unlike input substitution, which can only trade.

    In a PerSAM-style pipeline the similarity map is what selects the prompt points, so
    modulating it changes what the decoder is asked about. That is the cheapest place for a
    spectral prior to act without the encoder ever seeing a non-RGB pixel.
    """
    base = normalise_map(sim_map)
    if plausibility is None or weight <= 0.0:
        return base
    w = float(np.clip(weight, 0.0, 1.0))
    plaus = np.asarray(plausibility, dtype=np.float64)
    if plaus.shape != base.shape:
        raise ValueError(f"plausibility shape {plaus.shape} != similarity shape {base.shape}")
    return np.clip((1.0 - w) * base + w * plaus, 0.0, 1.0)


def fuse_similarity(maps: Sequence[np.ndarray], alpha: float = 0.5) -> np.ndarray:
    """Fuse two similarity maps from different input composites.

        S = alpha * norm(S_0) + (1 - alpha) * norm(S_1)

    The dual-stream route: run the frozen encoder twice, once on RGB and once on CIR, and
    combine only the similarity maps. Costs 2x inference and zero training, and because
    `alpha=1` returns the RGB map untouched, a tuned alpha cannot underperform RGB.

    This is the direct answer to the objection that CIR *discards* RGB rather than adding to
    it: here nothing is discarded.
    """
    if len(maps) < 2:
        return normalise_map(maps[0])
    a = float(np.clip(alpha, 0.0, 1.0))
    first, second = normalise_map(maps[0]), normalise_map(maps[1])
    if first.shape != second.shape:
        raise ValueError(f"similarity maps differ in shape: {first.shape} vs {second.shape}")
    return np.clip(a * first + (1.0 - a) * second, 0.0, 1.0)


def select_prompt_points(
    score_map: np.ndarray,
    num_positive: int = 1,
    num_negative: int = 1,
    min_separation: int = 16,
):
    """Pick positive and negative point prompts from a score map.

    The baseline pipeline takes a single argmax and a single argmin. That is fragile on
    "stuff" classes, where the target is spatially extended and one peak describes it poorly.
    This takes several peaks with a minimum separation so the prompts sample the region
    rather than clustering on one hotspot.

    Returns (coords, labels) with coords as (x, y) and labels 1 = positive, 0 = negative,
    matching the SAM predictor convention.
    """
    s = np.asarray(score_map, dtype=np.float64)
    h, w = s.shape

    def _peaks(arr: np.ndarray, count: int):
        picked = []
        work = arr.copy()
        for _ in range(count):
            idx = int(np.argmax(work))
            y, x = divmod(idx, w)
            if not np.isfinite(work[y, x]):
                break
            picked.append((x, y))
            # Suppress a neighbourhood so the next pick is somewhere else entirely.
            y0, y1 = max(0, y - min_separation), min(h, y + min_separation + 1)
            x0, x1 = max(0, x - min_separation), min(w, x + min_separation + 1)
            work[y0:y1, x0:x1] = -np.inf
        return picked

    positives = _peaks(s, num_positive)
    negatives = _peaks(-s, num_negative)

    coords = np.array(positives + negatives, dtype=np.float32)
    labels = np.array([1] * len(positives) + [0] * len(negatives), dtype=np.int32)
    return coords, labels
