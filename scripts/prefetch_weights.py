"""Download the SAM 3 weights into the local Hugging Face cache (run on a node WITH internet).

    hf auth login            # once; or export HF_TOKEN=...   (never commit a token)
    python scripts/prefetch_weights.py

Fetches only what the transformers route needs (skips the 3.4 GB native sam3.pt).
"""
import sys

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "facebook/sam3"

from huggingface_hub import get_token, snapshot_download  # noqa: E402

if not get_token():
    sys.exit("No Hugging Face login found. Run `hf auth login` (or export HF_TOKEN) first.")
path = snapshot_download(MODEL_ID, allow_patterns=["*.json", "*.txt", "*.safetensors", "LICENSE", "README.md"])
print(f"{MODEL_ID} cached at {path}")
