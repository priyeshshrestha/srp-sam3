"""SAM 3 few-shot segmentation via the spatial-concatenation canvas (Tsai, Lin & Wang,
arXiv 2604.05433), through the Hugging Face `transformers` implementation.

SAM 3 cannot take an exemplar from another image, so support and query are pasted onto one
canvas (models/canvas.py) and a box on the support half is the visual exemplar. The model then
returns instance masks for the concept; they are united and the query half is cropped out.

The support MASK never reaches the model, only one box. FLAIR classes are 'stuff' with no
instances, so the box is the tight bbox of the largest connected component (a documented choice,
not a tuned one). If no component qualifies, the whole-class bbox is used and the episode is
flagged `box_fallback` so its results can be examined separately.

K-shot: one canvas per support, then a pixel-wise majority vote over the K query masks.

Two reference arms that are NOT few-shot:
  text    the class phrase alone (CLASS_PHRASES, fixed before any result was seen)
  oracle  a box taken from the QUERY's own ground truth, placed on the query itself: an upper
          bound on how well SAM 3 can segment the class when told where it is

Auth: weights are gated. Log in once (`hf auth login` or HF_TOKEN in the environment); never put
a token in this repo.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np

from models.canvas import build_canvas, box_to_canvas_xyxy, extract_query_mask
from models.prompting import select_prompt_box

SAM3_MODEL_ID = "facebook/sam3"

# One short noun phrase per FLAIR class. Chosen a priori from the class definitions, never tuned
# on results. "pervious" = gardens/lawns/sports grounds; "impervious" = roads/car parks/pavements.
CLASS_PHRASES = {
    1: "building", 2: "lawn", 3: "paved road", 4: "bare soil", 5: "water", 6: "conifer tree",
    7: "deciduous tree", 8: "shrub", 9: "vineyard", 10: "grass", 11: "crop field", 12: "plowed field",
}


def support_box(support_mask: np.ndarray) -> Tuple[Tuple[int, int, int, int], bool]:
    """(x, y, w, h), box_fallback flag."""
    picked = select_prompt_box(support_mask, strategy="largest")
    if picked is not None:
        return picked.box_xywh, False
    ys, xs = np.nonzero(support_mask)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1), True


def majority_vote(masks: Sequence[np.ndarray]) -> np.ndarray:
    """Pixel-wise majority over K binary masks (ties at even K count as positive)."""
    stack = np.stack([np.asarray(m, bool) for m in masks]).astype(np.float32)
    return stack.mean(0) >= 0.5


class Sam3Fss:
    def __init__(self, device: str, model_id: str = SAM3_MODEL_ID, score_threshold: float = 0.5):
        import torch
        from transformers import Sam3Model, Sam3Processor
        self.torch, self.device, self.threshold = torch, device, score_threshold
        self.model = Sam3Model.from_pretrained(model_id).to(device).eval()
        self.processor = Sam3Processor.from_pretrained(model_id)

    def _union(self, image: np.ndarray, **prompt) -> Tuple[np.ndarray, int]:
        """Run SAM 3 on `image` with a text or box prompt; union of all returned instances."""
        torch = self.torch
        with torch.inference_mode():
            inputs = self.processor(images=image, return_tensors="pt", **prompt).to(self.device)
            outputs = self.model(**inputs)
            res = self.processor.post_process_instance_segmentation(
                outputs, threshold=self.threshold, mask_threshold=0.5,
                target_sizes=inputs.get("original_sizes").tolist())[0]
        masks = res["masks"]
        n = int(masks.shape[0]) if masks is not None else 0
        if not n:
            return np.zeros(image.shape[:2], dtype=bool), 0
        return masks.any(dim=0).cpu().numpy().astype(bool), n

    def _one_canvas(self, support_rgb, support_mask, query_rgb):
        box_xywh, fallback = support_box(support_mask)
        canvas, layout, _ = build_canvas(support_rgb, query_rgb, support_box_xywh=box_xywh)
        h, w = support_mask.shape
        box = box_to_canvas_xyxy(box_xywh, layout.support, (w, h))
        union, n = self._union(canvas, input_boxes=[[list(box)]], input_boxes_labels=[[1]])
        pred = extract_query_mask(union.astype(np.uint8), layout, out_size=query_rgb.shape[:2]).astype(bool)
        return pred, {"box_xyxy_canvas": list(box), "box_fallback": fallback, "n_masks": n}

    def predict(self, support_rgb, support_mask, query_rgb: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """K-shot when given lists of supports; one canvas per support, majority vote."""
        if isinstance(support_rgb, np.ndarray) and support_rgb.ndim == 3:
            return self._one_canvas(support_rgb, support_mask, query_rgb)
        preds, diags = [], []
        for s_rgb, s_mask in zip(support_rgb, support_mask):
            p, d = self._one_canvas(s_rgb, s_mask, query_rgb)
            preds.append(p)
            diags.append(d)
        return majority_vote(preds), {
            "k_shot": len(preds), "box_fallback": any(d["box_fallback"] for d in diags),
            "n_masks": [d["n_masks"] for d in diags],
            "empty_per_support": [int(not p.any()) for p in preds],
        }

    def predict_text(self, query_rgb: np.ndarray, class_id: int) -> Tuple[np.ndarray, Dict]:
        phrase = CLASS_PHRASES[class_id]
        pred, n = self._union(query_rgb, text=phrase)
        return pred, {"text": phrase, "n_masks": n}

    def predict_oracle(self, query_rgb: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, Dict]:
        from models.persam import oracle_prompts
        _, _, box = oracle_prompts(gt)
        pred, n = self._union(query_rgb, input_boxes=[[[int(v) for v in box]]], input_boxes_labels=[[1]])
        return pred, {"box": [int(v) for v in box], "n_masks": n, "oracle": True}
