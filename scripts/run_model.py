"""Run ONE model over the fixed episode list and save per-episode results and masks.

    python scripts/run_model.py --model sam2 --variant small \
        --checkpoint ../LoveDA/sam2_checkpoints/sam2.1_hiera_small.pt \
        --root ./data --episodes outputs/episodes.json

    python scripts/run_model.py --model sam1 --variant vit_b --checkpoint <sam_vit_b_01ec64.pth> ...
    python scripts/run_model.py --model sam3 ...            # weights via `hf auth login` / HF_TOKEN

Outputs under --out (default outputs/runs/<model>_<variant>/):
    results.jsonl   one line per episode: confusion counts, IoU inputs, timing, diagnostics
    preds/NNNNNN.npz   the predicted mask (bit-packed)
    run_meta.json   what was run. Resumable: finished episodes are skipped.

The band order comes from the episodes file, so it cannot disagree with how episodes were drawn.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.flair_rgb import load_patch  # noqa: E402
from evaluation.episodes import episodes_hash, load_episodes  # noqa: E402
from evaluation.metrics import confusion  # noqa: E402

DEFAULT_VARIANT = {"sam1": "vit_b", "sam2": "small", "sam3": "facebook/sam3", "mock": "colour"}


def pick_device(requested: str) -> str:
    import torch
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=os.path.dirname(__file__), text=True).strip()
    except Exception:
        return "unknown"


PROMPTS = ("fewshot", "oracle", "text")


def build_predictor(model: str, variant: str, checkpoint: str, device: str, prompt: str = "fewshot"):
    """Return fn(supports, query_rgb, query_gt, class_id) -> (bool mask, diagnostics).

    supports is a list of (rgb, bool mask). 'oracle' and 'text' ignore it: oracle prompts come
    from the query's own ground truth (an upper bound), text prompts from the class phrase.
    """
    if prompt == "text" and model not in ("sam3", "mock"):
        raise ValueError(f"text prompts need SAM 3; {model} has no text encoder")
    if model == "mock":
        from models.mock import mock_predict
        def predict(supports, q_rgb, q_gt, class_id):
            if prompt == "fewshot":
                return mock_predict([s for s, _ in supports], [m for _, m in supports], q_rgb)
            return mock_predict([q_rgb], [q_gt], q_rgb)
        return predict
    if model == "sam3":
        from models.sam3_fss import Sam3Fss
        m = Sam3Fss(device, model_id=variant)
        if prompt == "oracle":
            return lambda supports, q_rgb, q_gt, c: m.predict_oracle(q_rgb, q_gt)
        if prompt == "text":
            return lambda supports, q_rgb, q_gt, c: m.predict_text(q_rgb, c)
        return lambda supports, q_rgb, q_gt, c: m.predict(
            [s for s, _ in supports], [k for _, k in supports], q_rgb)
    from models.persam import load_persam_adapter, persam_oracle, persam_predict
    adapter = load_persam_adapter(model, variant, checkpoint, device)
    if prompt == "oracle":
        return lambda supports, q_rgb, q_gt, c: persam_oracle(adapter, q_rgb, q_gt)
    return lambda supports, q_rgb, q_gt, c: persam_predict(
        adapter, [s for s, _ in supports], [k for _, k in supports], q_rgb)


def arm_name(prompt: str, k_shot: int) -> str:
    return f"{k_shot}shot" if prompt == "fewshot" else prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["sam1", "sam2", "sam3", "mock"],
                    help="mock = colour-threshold stand-in used ONLY to test the pipeline; NOT a measurement")
    ap.add_argument("--variant", default=None,
                    help="sam1: vit_b|vit_l|vit_h; sam2: tiny|small|base_plus|large; sam3: HF model id")
    ap.add_argument("--checkpoint", default=None, help="weights file (sam1/sam2). sam3 loads from Hugging Face")
    ap.add_argument("--root", required=True)
    ap.add_argument("--episodes", default="outputs/episodes.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=None, help="first N episodes only (smoke test)")
    ap.add_argument("--per_class_limit", type=int, default=None,
                    help="only the first N episodes of each class (a random subset, since episodes are "
                         "drawn randomly). Used to cap expensive arms such as SAM 3 5-shot")
    ap.add_argument("--k_shot", type=int, default=1, help="supports per episode (fewshot prompt only)")
    ap.add_argument("--prompt", choices=PROMPTS, default="fewshot",
                    help="fewshot = from the supports; oracle = box/points from the QUERY ground truth "
                         "(upper bound, not a method); text = class phrase only (SAM 3)")
    args = ap.parse_args()

    variant = args.variant or DEFAULT_VARIANT[args.model]
    arm = arm_name(args.prompt, args.k_shot)
    tag = f"{args.model}_{variant.replace('/', '-')}_{arm}"
    out = args.out or os.path.join("outputs", "runs", tag)
    os.makedirs(os.path.join(out, "preds"), exist_ok=True)

    ep_file = load_episodes(args.episodes)
    band_order = ep_file["meta"]["band_order"]
    episodes = ep_file["episodes"]
    if args.per_class_limit:
        seen = {}
        kept = []
        for ep in episodes:
            seen[ep["class_id"]] = seen.get(ep["class_id"], 0) + 1
            if seen[ep["class_id"]] <= args.per_class_limit:
                kept.append(ep)
        episodes = kept
    if args.prompt == "fewshot" and args.k_shot > 1:
        short = [e for e in episodes if len(e.get("supports", [e["support"]])) < args.k_shot]
        if short:
            print(f"note: {len(short)} episode(s) hold fewer than {args.k_shot} supports and are skipped")
        episodes = [e for e in episodes if len(e.get("supports", [e["support"]])) >= args.k_shot]
    episodes = episodes[: args.limit]

    sha = episodes_hash(ep_file["episodes"])      # of the FULL list, whatever --limit says
    meta_path = os.path.join(out, "run_meta.json")
    if os.path.exists(meta_path):
        prev = json.load(open(meta_path))
        if prev.get("episodes_sha") not in (None, sha):
            sys.exit(f"ERROR: {out} holds results for a different episode file "
                     f"({prev.get('episodes_sha')} vs {sha}). Delete that directory, do not mix runs.")

    results_path = os.path.join(out, "results.jsonl")
    done = set()
    if os.path.exists(results_path):
        with open(results_path) as fh:
            done = {json.loads(line)["id"] for line in fh if line.strip()}

    device = pick_device(args.device)
    print(f"model={args.model} variant={variant} device={device} band_order={band_order} "
          f"episodes={len(episodes)} (already done: {len(done)})")
    k_max = ep_file["meta"].get("k_shot_max") or 1
    if args.prompt == "fewshot" and args.k_shot > k_max:
        sys.exit(f"ERROR: --k_shot {args.k_shot} but the episodes file stores only {k_max} support(s) "
                 f"per episode. Regenerate episodes with --k_shot_max {args.k_shot}.")
    predict = build_predictor(args.model, variant, args.checkpoint, device, args.prompt)

    with open(meta_path, "w") as fh:
        json.dump({"episodes_sha": sha, "is_measurement": args.model != "mock", "model": args.model,
                   "prompt": args.prompt, "k_shot": args.k_shot if args.prompt == "fewshot" else 0, "arm": arm, "variant": variant, "device": device,
                   "checkpoint": os.path.basename(args.checkpoint) if args.checkpoint else None,
                   "episodes_file": args.episodes, "band_order": band_order,
                   "per_class_limit": args.per_class_limit,
                   "git": git_sha(), "n_episodes": len(episodes)}, fh, indent=1)

    from tqdm import tqdm
    with open(results_path, "a") as results:
        for ep in tqdm(episodes, desc=tag):
            if ep["id"] in done:
                continue
            q_rgb, q_msk = load_patch(args.root, ep["query"], band_order)
            q_bin = q_msk == ep["class_id"]
            supports = []
            if args.prompt == "fewshot":
                for rel in ep.get("supports", [ep["support"]])[: args.k_shot]:
                    s_rgb, s_msk = load_patch(args.root, rel, band_order)
                    supports.append((s_rgb, s_msk == ep["class_id"]))

            t0 = time.time()
            pred, diag = predict(supports, q_rgb, q_bin, ep["class_id"])
            if device == "cuda":
                import torch
                torch.cuda.synchronize()
            dt = time.time() - t0

            rec = {"id": ep["id"], "class_id": ep["class_id"], **confusion(pred, q_bin),
                   "seconds": round(dt, 3), "diag": diag}
            # Mask first, result line second: a crash in between must not leave a result
            # whose mask is missing (the resume logic trusts results.jsonl).
            np.savez_compressed(os.path.join(out, "preds", f"{ep['id']:06d}.npz"),
                                packed=np.packbits(pred), shape=np.array(pred.shape))
            results.write(json.dumps(rec) + "\n")
            results.flush()
    print(f"Done. Results in {out}")


if __name__ == "__main__":
    main()
