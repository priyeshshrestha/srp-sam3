"""Fail-fast checks, each run in the SAME Python environment that will run the model.

    python scripts/preflight.py data  --root ./data
    python scripts/preflight.py model --model sam2 --variant small --checkpoint <file>

Exit status 0 = ready, 1 = not ready (every problem is printed, with the fix).
"""
import argparse
import importlib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def check_data(root: str) -> list:
    problems = []
    if not os.path.isdir(root):
        return [f"data directory {root!r} does not exist (set FLAIR_DATA_DIR)"]
    try:
        importlib.import_module("tifffile")
    except ImportError:
        try:
            importlib.import_module("rasterio")
        except ImportError:
            problems.append("neither rasterio nor tifffile is installed (pip install tifffile)")
    from datasets.flair_rgb import discover_patches
    try:
        rels = discover_patches(root)
        print(f"  found {len(rels)} patches with masks")
    except FileNotFoundError as e:
        problems.append(str(e))
        return problems
    try:                                                       # can one patch actually be decoded?
        from datasets.flair import _read_tif, mask_path_for
        from datasets.flair_rgb import read_label
        img = _read_tif(os.path.join(root, rels[0]))
        lab = read_label(mask_path_for(os.path.join(root, rels[0])))
        print(f"  read {rels[0]}: {img.shape[0]} bands {img.shape[1]}x{img.shape[2]}, label {lab.shape}")
    except Exception as e:
        hint = " (LZW-compressed FLAIR-HUB files need: pip install imagecodecs)" if "imagecodecs" in str(e).lower() \
            or "lzw" in str(e).lower() or "compress" in str(e).lower() else ""
        problems.append(f"cannot read {rels[0]}: {type(e).__name__}: {e}{hint}")
    return problems


def _need(module: str, hint: str) -> list:
    try:
        importlib.import_module(module)
        return []
    except Exception as e:                                     # ImportError, or a broken install
        return [f"cannot import {module!r} ({type(e).__name__}: {e}). {hint}"]


def check_model(model: str, variant: str, checkpoint: str) -> list:
    problems = _need("torch", "pip install torch")
    if model == "mock":
        return problems
    if model in ("sam1", "sam2"):
        if not checkpoint:
            problems.append(f"{model}: no checkpoint path given")
        elif not os.path.isfile(checkpoint):
            problems.append(f"{model}: checkpoint not found at {checkpoint!r}")
        elif os.path.getsize(checkpoint) < 10_000_000:
            problems.append(f"{model}: checkpoint {checkpoint!r} is only {os.path.getsize(checkpoint)} bytes "
                            f"(truncated download?)")
    if model == "sam1":
        problems += _need("segment_anything", "pip install segment-anything")
        if variant not in ("vit_b", "vit_l", "vit_h"):
            problems.append(f"sam1 variant must be vit_b|vit_l|vit_h, got {variant!r}")
    elif model == "sam2":
        problems += _need("sam2", "pip install git+https://github.com/facebookresearch/sam2.git")
        from models.persam import SAM2_CONFIGS
        if variant not in SAM2_CONFIGS:
            problems.append(f"sam2 variant must be one of {list(SAM2_CONFIGS)}, got {variant!r}")
    elif model == "sam3":
        bad = _need("transformers", "pip install -U transformers")
        if not bad:
            try:
                from transformers import Sam3Model, Sam3Processor  # noqa: F401
            except Exception as e:
                bad = [f"this transformers release has no Sam3Model ({type(e).__name__}). pip install -U transformers"]
        problems += bad
        if not bad:
            problems += check_sam3_weights(variant or "facebook/sam3")
    else:
        problems.append(f"unknown model {model!r}")
    return problems


def check_sam3_weights(model_id: str) -> list:
    """Weights must be in the local cache, or reachable with a valid token (gated repo)."""
    try:
        from huggingface_hub import get_token, try_to_load_from_cache
    except ImportError:
        return ["huggingface_hub missing (pip install huggingface_hub)"]
    cached = try_to_load_from_cache(model_id, "model.safetensors")
    if isinstance(cached, str):
        print(f"  {model_id} weights found in the local Hugging Face cache")
        return []
    if not get_token():
        return [f"{model_id}: weights not cached and no Hugging Face login. Run `hf auth login` "
                f"(or export HF_TOKEN) once on a machine with internet, then `bash run.sh prefetch`. "
                f"Compute nodes often have no internet, so prefetch on the login node."]
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
        get_hf_file_metadata(hf_hub_url(model_id, "config.json"), timeout=20)
        print(f"  {model_id}: token accepted (weights will download on first use; "
              f"run `bash run.sh prefetch` first if this node has no internet)")
        return []
    except Exception as e:
        return [f"{model_id}: token present but access failed ({type(e).__name__}). Is access to the gated repo "
                f"approved for this account, and does this node have internet? Try `bash run.sh prefetch` "
                f"on the login node."]


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="what", required=True)
    d = sub.add_parser("data")
    d.add_argument("--root", required=True)
    m = sub.add_parser("model")
    m.add_argument("--model", required=True)
    m.add_argument("--variant", default="")
    m.add_argument("--checkpoint", default="")
    a = ap.parse_args()

    problems = check_data(a.root) if a.what == "data" else check_model(a.model, a.variant, a.checkpoint)
    if problems:
        print(f"PREFLIGHT FAILED ({a.what}):")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print(f"preflight ok ({a.what})")


if __name__ == "__main__":
    main()
