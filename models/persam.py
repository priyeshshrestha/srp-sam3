"""Training-free one-shot segmentation for SAM 1 and SAM 2 (PerSAM-style front end).

SAM 1 and SAM 2 have no way to take a reference image, so the reference is turned into prompts
by feature matching, identically for both models:

  1. embed the support image, average its features under the support mask -> prototype
  2. embed the query image, cosine-match the prototype against every query feature
  3. positive point = most similar pixel, negative point = least similar pixel, box = region
     above a threshold relative to the map's own maximum
  4. decode 3 candidate masks, keep the one whose mean feature is closest to the prototype

Only the encoder/decoder differ between SAM 1 and SAM 2 (`SamAdapter`), so a difference in the
numbers is a difference in the model, not in the prompting. All models are frozen.

FLAIR patches are square, and SAM resizes the long side to 1024 and pads the rest, so a square
image maps 1:1 onto the feature grid. Non-square input would misalign the similarity map and is
rejected rather than silently mis-registered.
"""

from typing import Dict, Optional, Protocol, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class SamAdapter(Protocol):
    device: str

    def embed(self, rgb_uint8: np.ndarray) -> torch.Tensor:
        """Encode an (H, W, 3) uint8 image; return (1, C, h, w) features. Leaves the image set."""

    def decode(self, points: np.ndarray, labels: np.ndarray, box: Optional[np.ndarray]
               ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict 3 candidate masks for the LAST embedded image: (masks (3,H,W) bool, scores (3,))."""


def _masked_mean(feat: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
    """Mean of (1, C, h, w) features weighted by the area fraction of `mask` in each cell."""
    m = torch.from_numpy(np.asarray(mask, dtype=np.float32)).to(feat.device)[None, None]
    w = F.adaptive_avg_pool2d(m, feat.shape[-2:])
    if float(w.sum()) <= 0:
        return None
    return (feat * w).sum(dim=(2, 3)) / w.sum()


def persam_predict(
    adapter: SamAdapter,
    support_rgb,
    support_mask,
    query_rgb: np.ndarray,
    box_rel_thresh: float = 0.7,
    box_abs_floor: float = 0.2,
    min_box_pixels: int = 20,
) -> Tuple[np.ndarray, Dict]:
    """K-shot prediction (K = number of supports). Returns (bool mask, diagnostics).

    `support_rgb` / `support_mask` are one image and mask, or equal-length lists of them. With
    K > 1 the class prototype is the normalised mean of the K per-support prototypes.
    """
    if isinstance(support_rgb, np.ndarray) and support_rgb.ndim == 3:
        support_rgb, support_mask = [support_rgb], [support_mask]
    H, W = query_rgb.shape[:2]
    if H != W or any(s.shape[0] != s.shape[1] for s in support_rgb):
        raise ValueError("persam_predict needs square images (FLAIR patches are 512x512)")

    with torch.inference_mode():
        protos = []
        for s_rgb, s_mask in zip(support_rgb, support_mask):
            p = _masked_mean(adapter.embed(s_rgb), s_mask)
            if p is None:
                raise ValueError("support mask is empty")
            protos.append(F.normalize(p, dim=-1))
        proto = F.normalize(torch.stack(protos).mean(0), dim=-1)   # (1, C)

        feat_q = adapter.embed(query_rgb)                          # query is now the set image
        sim = torch.einsum("bc,bchw->bhw", proto, F.normalize(feat_q, dim=1))
        sim = F.interpolate(sim[:, None], size=(H, W), mode="bilinear",
                            align_corners=False)[0, 0]             # (H, W)

        max_val, max_idx = sim.view(-1).max(0)
        min_idx = sim.view(-1).argmin()
        yp, xp = divmod(int(max_idx), W)
        yn, xn = divmod(int(min_idx), W)
        points = np.array([[xp, yp], [xn, yn]], dtype=np.float32)
        labels = np.array([1, 0], dtype=np.int32)

        # Relative threshold: cosine maps sit in an image-dependent band, so anchor to the peak.
        thresh = max(box_abs_floor, box_rel_thresh * float(max_val))
        hot = (sim >= thresh).cpu().numpy()
        box = None
        if hot.sum() > min_box_pixels:
            ys, xs = np.where(hot)
            box = np.array([xs.min(), ys.min(), xs.max(), ys.max()])

        masks, scores = adapter.decode(points, labels, box)

        # Choose among candidates by prototype similarity, falling back to the model's own score.
        best, best_sim = int(np.argmax(scores)), -np.inf
        for i, cand in enumerate(masks):
            mean = _masked_mean(feat_q, cand)
            if mean is None:
                continue
            s = float((F.normalize(mean, dim=-1) * proto).sum())
            if s > best_sim:
                best, best_sim = i, s

    return masks[best].astype(bool), {
        "pos_point": [int(xp), int(yp)], "neg_point": [int(xn), int(yn)],
        "box": None if box is None else [int(v) for v in box],
        "peak_sim": float(max_val), "candidate": best, "k_shot": len(protos),
    }


def oracle_prompts(gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prompts taken from the QUERY's own ground truth -- an upper bound, never a usable method.

    box = tight box of the largest connected region; positive point = the pixel deepest inside
    that region; negative point = the background pixel farthest from any target pixel.
    """
    from scipy import ndimage
    gt = np.asarray(gt, bool)
    if not gt.any():
        raise ValueError("oracle prompts need a non-empty ground truth")
    lab, n = ndimage.label(gt, structure=np.ones((3, 3)))
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
    big = lab == (int(np.argmax(sizes)) + 1)
    ys, xs = np.nonzero(big)
    box = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
    yp, xp = np.unravel_index(int(np.argmax(ndimage.distance_transform_edt(big))), gt.shape)
    points, labels = [[xp, yp]], [1]
    if (~gt).any():
        yn, xn = np.unravel_index(int(np.argmax(ndimage.distance_transform_edt(~gt))), gt.shape)
        points.append([xn, yn])
        labels.append(0)
    return np.array(points, np.float32), np.array(labels, np.int32), box


def persam_oracle(adapter: SamAdapter, query_rgb: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """Decoder quality given ideal prompts: the model's own top-scoring candidate is kept."""
    points, labels, box = oracle_prompts(gt)
    with torch.inference_mode():
        adapter.embed(query_rgb)
        masks, scores = adapter.decode(points, labels, box)
    best = int(np.argmax(scores))
    return masks[best].astype(bool), {"box": [int(v) for v in box], "candidate": best, "oracle": True}


# ----------------------------------------------------------------------------- adapters


class Sam1Adapter:
    """segment_anything (SAM 1). `variant` in {vit_b, vit_l, vit_h}."""

    def __init__(self, checkpoint: str, variant: str, device: str):
        from segment_anything import SamPredictor, sam_model_registry
        sam = sam_model_registry[variant](checkpoint=checkpoint).to(device).eval()
        self.predictor, self.device = SamPredictor(sam), device

    def embed(self, rgb_uint8):
        self.predictor.set_image(rgb_uint8)            # expects HWC uint8 RGB
        return self.predictor.get_image_embedding()

    def decode(self, points, labels, box):
        masks, scores, _ = self.predictor.predict(
            point_coords=points, point_labels=labels, box=box, multimask_output=True)
        return masks, scores


SAM2_CONFIGS = {
    "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}


class Sam2Adapter:
    """sam2 (SAM 2.1). `variant` in {tiny, small, base_plus, large}."""

    def __init__(self, checkpoint: str, variant: str, device: str):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.predictor = SAM2ImagePredictor(build_sam2(SAM2_CONFIGS[variant], checkpoint, device=device))
        self.device = device

    def embed(self, rgb_uint8):
        self.predictor.set_image(rgb_uint8)
        # No public accessor: SAM 2 caches the encoder output on the predictor.
        return self.predictor._features["image_embed"]

    def decode(self, points, labels, box):
        masks, scores, _ = self.predictor.predict(
            point_coords=points, point_labels=labels, box=box, multimask_output=True)
        return masks, scores


def load_persam_adapter(model: str, variant: str, checkpoint: str, device: str) -> SamAdapter:
    if not checkpoint:
        raise ValueError(f"--checkpoint is required for {model}")
    if not __import__("os").path.exists(checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if model == "sam1":
        return Sam1Adapter(checkpoint, variant, device)
    if model == "sam2":
        return Sam2Adapter(checkpoint, variant, device)
    raise ValueError(f"not a PerSAM model: {model!r}")
