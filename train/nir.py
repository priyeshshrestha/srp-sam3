"""NIR-band transforms: simulated camera changes, NIR-robust training, and label-free NIR calibration.

Why this exists. The RGB+NIR model leans on NIR for trees (NIR flattened: coniferous -0.21, deciduous
-0.23 IoU). A drone camera records NIR with a different gain, offset and response curve than IGN's
aircraft camera, so the question is whether the NIR gain survives a camera change, and how to make it.

Everything here acts on the NIR band only, as float32 values on the 0..255 scale (before
`train.sam3_rgbi.normalise`). R, G, B are never touched, so the RGB-only model is unaffected by every
transform and is the fixed reference line.

  apply_nir(nir, gain, offset, gamma)   one camera response: 255 * (nir/255) ** gamma * gain + offset, clipped
  NIR_MISSING = 127.5                   normalises to exactly 0.0 -> the zero-initialised NIR filters add
                                        nothing, i.e. "no NIR" is a well-defined input, not a black band
  SHIFTS                                the named test conditions (simulated cameras)
  random_nir(nir, rng)                  the training augmentation (NIR-robust training)
  calibrate(nir, src, ref)              moment matching: rescale a new camera's NIR so its mean and std
                                        equal those of the FLAIR training set. Needs unlabelled images only.

Test conditions inside vs outside the training range. `random_nir` draws gain in [0.7, 1.3], offset in
[-25, 25] and gamma in [0.8, 1.25]. gain_0.8 / gain_1.2 / gamma_1.25 / camera_b fall INSIDE that range
(a robust model has seen such inputs); gain_0.6 and gain_1.4 fall OUTSIDE it. Report both groups
separately, so robustness is not just "tested on what it was trained on".
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

NIR_MISSING = 127.5

# name -> camera response (None = unchanged, "missing" = no NIR)
SHIFTS: Dict[str, object] = {
    "normal": None,
    "gain_0.6": {"gain": 0.6},
    "gain_0.8": {"gain": 0.8},
    "gain_1.2": {"gain": 1.2},
    "gain_1.4": {"gain": 1.4},
    "offset_-20": {"offset": -20.0},
    "offset_+20": {"offset": 20.0},
    "gamma_1.25": {"gamma": 1.25},
    "camera_b": {"gain": 0.8, "offset": 15.0, "gamma": 1.15},   # a plausible different sensor: all three at once
    "missing": "missing",
}
OUT_OF_TRAINING_RANGE = ("gain_0.6", "gain_1.4")
CALIB_PREFIX = "calib:"           # "calib:camera_b" = camera_b, then label-free calibration to FLAIR's statistics

# Training augmentation (fixed before any result; not tuned)
AUG = {"p_missing": 0.2, "p_clean": 0.2, "gain": (0.7, 1.3), "offset": (-25.0, 25.0), "gamma": (0.8, 1.25)}


def apply_nir(nir: np.ndarray, gain: float = 1.0, offset: float = 0.0, gamma: float = 1.0) -> np.ndarray:
    """float32 NIR on 0..255 after a camera response: gamma curve, then gain and offset, clipped to 0..255."""
    x = np.asarray(nir, dtype=np.float32)
    if gamma != 1.0:
        x = 255.0 * np.power(x / 255.0, gamma, dtype=np.float32)
    x = x * np.float32(gain) + np.float32(offset)
    return np.clip(x, 0.0, 255.0).astype(np.float32)


def shift_nir(nir: np.ndarray, condition: str) -> np.ndarray:
    """Apply a named condition from SHIFTS (without the calib: prefix)."""
    if condition not in SHIFTS:
        raise ValueError(f"unknown NIR condition {condition!r}; known: {sorted(SHIFTS)}")
    spec = SHIFTS[condition]
    if spec is None:
        return np.asarray(nir, dtype=np.float32)
    if spec == "missing":
        return np.full(np.shape(nir), NIR_MISSING, dtype=np.float32)
    return apply_nir(nir, **spec)


def parse_condition(cond: str) -> Tuple[str, bool]:
    """'calib:gain_0.6' -> ('gain_0.6', True);  'gain_0.6' -> ('gain_0.6', False)."""
    if cond.startswith(CALIB_PREFIX):
        base = cond[len(CALIB_PREFIX):]
        if SHIFTS.get(base) == "missing":
            raise ValueError("calibrating a missing NIR band is meaningless")
        shift_nir(np.zeros(1), base)              # validates the name
        return base, True
    shift_nir(np.zeros(1), cond)
    return cond, False


def random_nir(nir: np.ndarray, rng: np.random.Generator, aug: Optional[Dict] = None) -> np.ndarray:
    """NIR-robust training: with p_missing no NIR, with p_clean the real NIR, otherwise a random camera."""
    aug = aug or AUG
    u = rng.random()
    if u < aug["p_missing"]:
        return np.full(np.shape(nir), NIR_MISSING, dtype=np.float32)
    if u < aug["p_missing"] + aug["p_clean"]:
        return np.asarray(nir, dtype=np.float32)
    gain = rng.uniform(*aug["gain"])
    offset = rng.uniform(*aug["offset"])
    lg = np.log(aug["gamma"])
    gamma = float(np.exp(rng.uniform(lg[0], lg[1])))
    return apply_nir(nir, gain=gain, offset=offset, gamma=gamma)


# ------------------------------------------------------------------------------ calibration
MIN_STD = 1.0                     # a flat NIR band (std ~0) must not blow up the rescaling


def nir_moments(bands: Sequence[np.ndarray]) -> Tuple[float, float]:
    """Pooled mean and std of NIR over a set of patches (each (H, W))."""
    n = s = ss = 0.0
    for b in bands:
        b = np.asarray(b, dtype=np.float64)
        n += b.size
        s += b.sum()
        ss += np.square(b).sum()
    if n == 0:
        raise ValueError("no NIR pixels to compute statistics from")
    mean = s / n
    return float(mean), float(np.sqrt(max(ss / n - mean * mean, 0.0)))


def calibrate(nir: np.ndarray, src: Tuple[float, float], ref: Tuple[float, float]) -> np.ndarray:
    """Moment matching: (nir - mean_src) / std_src * std_ref + mean_ref, clipped to 0..255.
    src = statistics of the new camera / region (unlabelled images), ref = statistics of the FLAIR training set."""
    m_s, s_s = src
    m_r, s_r = ref
    x = (np.asarray(nir, dtype=np.float32) - np.float32(m_s)) / np.float32(max(s_s, MIN_STD))
    return np.clip(x * np.float32(s_r) + np.float32(m_r), 0.0, 255.0).astype(np.float32)
