#!/bin/bash
#SBATCH --job-name=srp_sam_flair
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-user=shresthap@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# ---------------------------------------------------------------------------
# No training happens here. Every model is frozen. The experiment varies only
# WHICH ENCODER is used and WHERE spectral evidence enters the pipeline.
#
# The two axes:
#   BACKENDS   sam (ViT-B) | sam2 (Hiera-S) | native/hf (SAM 3)
#
# ELEVATION IS OFF THE TABLE. The EU UAV target data is RGB + NIR with no DSM,
# so FLAIR's nDSM channel cannot be on the deliverable path. Composites and
# Shannon sources are elevation-free by default; ndvi_dsm_h / rgb_dsm /
# --use_elevation remain only as a FLAIR-only diagnostic of what the payload
# is missing. Set DIAGNOSTIC=1 to append that arm.
#   INSERTION  where NIR acts:
#                --composite cir          input substitution   (MEASURED: harmful)
#                --nir_insertion modulate post-encoder blend    (untested)
#                --nir_insertion prompt   prompt-point choice   (untested)
#
# Measured baseline, SAM 1 + prototype-cosine prompting, 1-shot FLAIR:
#   rgb 0.2860 | cir 0.2535 | ndvi_dsm_h 0.2442
#   Vegetation classes are TIED (0.348/0.347/0.337); RGB wins entirely on
#   albedo-defined classes (impervious 0.340->0.124, plowed 0.374->0.229).
#   Hence: keep RGB at the input, insert spectra after the encoder.
#
# If SLURM is absent (this box has no `module` command and may be a plain
# node), run directly:  CUDA_VISIBLE_DEVICES=0 bash run.sh
# ---------------------------------------------------------------------------
set -euo pipefail

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
mkdir -p logs outputs

# source activate srp_env     # uncomment/adjust for your environment

FLAIR_DATA_DIR="${FLAIR_DATA_DIR:-./data}"
CLASS_INDEX="${CLASS_INDEX:-./outputs/flair_class_index.json}"
K_SHOT="${K_SHOT:-1}"
EPISODES="${EPISODES:-30}"
HOLDOUT_DOMAINS="${HOLDOUT_DOMAINS:-}"

# Which encoders to compare. Drop any that will not load on this machine.
BACKENDS="${BACKENDS:-sam sam2}"

# ---------------------------------------------------------------------------
# 1. Data present?
# ---------------------------------------------------------------------------
if [ ! -d "$FLAIR_DATA_DIR" ] || [ -z "$(find "$FLAIR_DATA_DIR" -name 'IMG_*.tif' -print -quit 2>/dev/null)" ]; then
    echo "ERROR: no FLAIR patches under $FLAIR_DATA_DIR"
    echo "Fetch on a login node first:  bash scripts/download_toy.sh $FLAIR_DATA_DIR"
    exit 1
fi
echo "Found $(find "$FLAIR_DATA_DIR" -name 'IMG_*.tif' | wc -l | tr -d ' ') FLAIR patches"

# ---------------------------------------------------------------------------
# 2. Tests, then verify the band-order assumption against the real files
# ---------------------------------------------------------------------------
echo; echo "=== Stage 2: tests (no GPU, no data) ==="
python tests/test_flair.py
python tests/test_canvas_prompting.py
python tests/test_spectral.py

echo; echo "=== Stage 3: band order + per-class NDVI ==="
# READ THIS OUTPUT. If the channel-order check FAILs, stop -- every NDVI
# downstream is meaningless until it is resolved.
python scripts/inspect_flair.py --root "$FLAIR_DATA_DIR" --samples 50

# ---------------------------------------------------------------------------
# 3. Class index (one full pass over every mask; cached)
# ---------------------------------------------------------------------------
echo; echo "=== Stage 4: class index ==="
if [ -f "$CLASS_INDEX" ]; then
    echo "Already at $CLASS_INDEX -- skipping."
else
    python -c "
import sys; sys.path.insert(0, '.')
from datasets.flair import FLAIRDataset
ds = FLAIRDataset(root='$FLAIR_DATA_DIR')
print(f'Indexing {len(ds)} patches...', flush=True)
idx = ds.build_class_index(save_to='$CLASS_INDEX')
for c, p in sorted(idx.items()): print(f'  class {c:2d}: {len(p):6d} patches')
"
fi

# ---------------------------------------------------------------------------
# 4. The sweep
# ---------------------------------------------------------------------------
run_eval () {   # backend composite insertion weight strategy
    echo; echo "--- backend=$1 composite=$2 insertion=$3 weight=$4 strategy=$5 ---"
    python scripts/evaluate_sam3.py \
        --root "$FLAIR_DATA_DIR" \
        --class_index "$CLASS_INDEX" \
        --model_backend "$1" \
        --composite "$2" \
        --nir_insertion "$3" \
        --nir_weight "$4" \
        --strategy "$5" \
        --k_shot "$K_SHOT" \
        --num_episodes "$EPISODES" \
        ${HOLDOUT_DOMAINS:+--holdout_domains $HOLDOUT_DOMAINS} \
        --output_dir ./outputs \
    || echo "  FAILED (backend unavailable?) -- continuing"
}

for B in $BACKENDS; do
    echo; echo "############ ENCODER: $B ############"

    # (a) Input-substitution axis. ELEVATION-FREE composites only -- the EU UAV
    #     target carries RGB + NIR and no nDSM, so a result depending on canopy
    #     height cannot ship. ndvi_h_nir replaces the old ndvi_dsm_h here.
    for COMP in rgb cir ndvi_h_nir rgb_ndvi; do
        run_eval "$B" "$COMP" none 0.0 ecological
    done

    # (b) Post-encoder insertion, on RGB input. weight=0.0 must reproduce the
    #     rgb/none row exactly -- that identity is the sanity check on the whole
    #     arm, so run it before trusting any non-zero weight.
    run_eval "$B" rgb modulate 0.0 ecological
    for W in 0.2 0.4 0.6; do
        run_eval "$B" rgb modulate "$W" ecological
    done

    # (c) Prompt-selection ablation. Is the ecological box choice doing anything?
    #     Never measured -- all Phase 3 runs used 'ecological' with no control.
    for S in random largest compact; do
        run_eval "$B" rgb none 0.0 "$S"
    done
done

    # (d) OPTIONAL elevation diagnostic. Not a deliverable result -- it answers
    #     "how much would canopy height buy us if the EU UAV payload had it?"
    if [ "${DIAGNOSTIC:-0}" = "1" ]; then
        echo; echo "--- elevation diagnostic (FLAIR-only, does not transfer) ---"
        run_eval "$B" ndvi_dsm_h none 0.0 ecological
    fi

# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------
echo; echo "=== Summary ==="
python -c "
import json, glob
rows = [json.load(open(f)) for f in sorted(glob.glob('./outputs/results_*.json'))]
if not rows:
    print('No results.'); raise SystemExit
print(f\"{'backend':<9}{'composite':<12}{'insert':<10}{'w':>5}{'strategy':<11}{'mIoU':>8}{'FB-IoU':>9}  transfers\")
print('-' * 78)
for r in sorted(rows, key=lambda r: -r.get('miou', 0)):
    t = '' if r.get('transfers_to_rgb_nir', True) else '  NO (elevation)'
    print(f\"{r.get('backend','?'):<9}{r.get('composite','?'):<12}\"
          f\"{r.get('nir_insertion','?'):<10}{r.get('nir_weight',0):>5}\"
          f\"{r.get('strategy','?'):<11}{r.get('miou',0):>8.4f}{r.get('fb_iou',0):>9.4f}{t}\")
"

echo; echo "Job completed at $(date)"
