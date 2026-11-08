"""A stand-in 'model' used ONLY to test the pipeline end to end without weights or a GPU.

It labels every query pixel whose colour is close to the mean colour of the support mask.
Its numbers are NOT a measurement of anything; runs are tagged `is_measurement: false`.
"""
from typing import Dict, Tuple

import numpy as np


def mock_predict(support_rgb, support_mask, query_rgb: np.ndarray) -> Tuple[np.ndarray, Dict]:
    if isinstance(support_rgb, np.ndarray) and support_rgb.ndim == 3:
        support_rgb, support_mask = [support_rgb], [support_mask]
    pix = np.concatenate([r[m] for r, m in zip(support_rgb, support_mask)]).astype(np.float32)
    mean = pix.mean(axis=0)
    dist = np.linalg.norm(query_rgb.astype(np.float32) - mean, axis=-1)
    return dist < 40.0, {"mock": True}
