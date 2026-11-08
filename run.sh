#!/bin/bash
#SBATCH --job-name=sam_flair
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
# Add your partition if your cluster needs one, e.g.:   #SBATCH --partition=<name>

# =============================================================================
# SAM 1 vs SAM 2 vs SAM 3 on FLAIR (RGB only, one-shot, training-free).
#
#   bash run.sh            run everything, end to end     (or:  sbatch run.sh)
#   bash run.sh prefetch   download the SAM 3 weights (do this on a node WITH internet)
#   bash run.sh check      preflight only: report what is missing, run nothing
#   bash run.sh train      the separate training experiment (SAM 3, RGB vs RGB+NIR): see run_train.sh
#
# Output:   outputs/report/summary.md        metrics, baselines, paired CIs, per-class table
#           outputs/report/figures/*.png     10 example episodes, every model on the SAME episode
#           outputs/report/per_class_iou.png
#           outputs/report.tar.gz            the same, ready to scp
#
# What protects you from mistakes:
#   * every problem is checked BEFORE any expensive step, and each one prints its fix
#   * the band order (R,G,B vs B,G,R) is checked on your real files, and the script STOPS
#     if it cannot decide, rather than guessing
#   * one fixed episode list is used by every model; results from different episode lists
#     are refused, never merged
#   * a model that cannot run is skipped with a clear message; the others still run; the
#     exit status is non-zero if anything was skipped or failed
#   * everything is resumable: re-run the same command after any interruption
#
# Configuration (environment variables, all optional):
#   FLAIR_DATA_DIR   dataset root                              [./data]
#   SAM1_CKPT        SAM 1 checkpoint                          [./checkpoints/sam_vit_b_01ec64.pth]
#   SAM2_CKPT        SAM 2.1 checkpoint                        [./checkpoints/sam2.1_hiera_small.pt]
#   RUNS             models to run, "model:variant:checkpoint" separated by spaces. Defaults to
#                    "sam1:vit_b:$SAM1_CKPT sam2:small:$SAM2_CKPT sam3::". Add bigger models, e.g.
#                    RUNS="sam1:vit_h:/p/sam_vit_h_4b8939.pth sam2:large:/p/sam2.1_hiera_large.pt sam3::"
#   ARMS             which prompt settings to run per model          [1shot 5shot oracle text]
#                      1shot / 5shot  few-shot from 1 or 5 labelled support patches (the task)
#                      oracle         prompt from the QUERY's own ground truth: an upper bound, not a method
#                      text           class phrase only (SAM 3 only; skipped for SAM 1/2)
#   K5_PER_CLASS     cap 5-shot to this many episodes per class (5x the compute)     [all]
#   BAND_ORDER       rgb | bgr. Skips the automatic check (set it after reading the montage)
#   N_PER_CLASS      episodes per class                        [100]
#   DOMAIN_MODE      auto | cross | same                       [auto: cross if >=2 domains, else same]
#   SEED             episode sampling seed                     [42]
#   PYTHON           python interpreter                        [python, else python3]
#   PYTHON_SAM1 / PYTHON_SAM2 / PYTHON_SAM3    per-model interpreter if they need separate envs
#   SMOKE            1 = run 2 episodes per model first to catch failures early   [1]
#   SKIP_TESTS       1 = skip the unit tests                   [0]
#   CUDA_VISIBLE_DEVICES   pin a GPU on a shared machine, e.g. CUDA_VISIBLE_DEVICES=3
# =============================================================================
set -uo pipefail

# Under SLURM the script is copied away from the repo; go back to where it was submitted.
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

MODE="${1:-run}"
# The training experiment (SAM 3 fine-tuned, RGB vs RGB+NIR) lives in its own script.
if [ "$MODE" = "train" ]; then shift; exec bash run_train.sh "$@"; fi
FLAIR_DATA_DIR="${FLAIR_DATA_DIR:-./data}"
OUT="${OUT:-./outputs}"
PYTHON="${PYTHON:-$(command -v python || command -v python3 || echo python)}"
SAM1_CKPT="${SAM1_CKPT:-./checkpoints/sam_vit_b_01ec64.pth}"
SAM2_CKPT="${SAM2_CKPT:-./checkpoints/sam2.1_hiera_small.pt}"
RUNS="${RUNS:-sam1:vit_b:$SAM1_CKPT sam2:small:$SAM2_CKPT sam3::}"
BAND_ORDER="${BAND_ORDER:-}"
N_PER_CLASS="${N_PER_CLASS:-100}"
DOMAIN_MODE="${DOMAIN_MODE:-auto}"
SEED="${SEED:-42}"
SMOKE="${SMOKE:-1}"
ARMS="${ARMS:-1shot 5shot oracle text}"
K5_PER_CLASS="${K5_PER_CLASS:-}"
SKIP_TESTS="${SKIP_TESTS:-0}"

mkdir -p "$OUT" logs
LOG="logs/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

say()  { echo; echo "=== $* ==="; }
die()  { echo; echo "ERROR: $*"; echo "(log: $LOG)"; exit 1; }
py_for() {   # interpreter for a model: PYTHON_SAM3 etc., else $PYTHON
    local v="PYTHON_$(echo "$1" | tr '[:lower:]' '[:upper:]')"
    echo "${!v:-$PYTHON}"
}
default_variant() {
    case "$1" in sam1) echo vit_b;; sam2) echo small;; sam3) echo facebook/sam3;; mock) echo colour;; *) echo "";; esac
}

echo "SAM x FLAIR  |  $(date)  |  host $(hostname)  |  job ${SLURM_JOB_ID:-local}"
echo "python: $($PYTHON --version 2>&1)  |  data: $FLAIR_DATA_DIR  |  out: $OUT"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset (all GPUs visible; pin one on a shared machine)}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || echo "nvidia-smi not found (CPU only?)"
# A GPU machine whose torch cannot see the GPU would silently run everything on the CPU: refuse.
if command -v nvidia-smi >/dev/null 2>&1 && [ "${CUDA_VISIBLE_DEVICES-unset}" != "" ]; then
    for py in $(printf '%s\n' "$PYTHON" "${PYTHON_SAM1:-$PYTHON}" "${PYTHON_SAM2:-$PYTHON}" "${PYTHON_SAM3:-$PYTHON}" | sort -u); do
        $py -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null || die "GPUs present but torch in $py cannot use them (torch built for a newer CUDA than the driver?). Use the matching environment, e.g. conda activate srp311"
    done
fi

# ---------------------------------------------------------------------------
if [ "$MODE" = "prefetch" ]; then
    say "Prefetch SAM 3 weights"
    $(py_for sam3) scripts/prefetch_weights.py || die "prefetch failed (login with \`hf auth login\` first)"
    exit 0
fi
[ "$MODE" = "run" ] || [ "$MODE" = "check" ] || die "unknown mode '$MODE' (use: run | check | prefetch | train)"

# ---------------------------------------------------------------------------
if [ "$SKIP_TESTS" != "1" ] && [ "$MODE" = "run" ]; then
    say "1. Unit tests (no data, GPU or weights needed)"
    $PYTHON tests/test_benchmark.py || die "unit tests failed: the code is broken in this environment; nothing was run"
fi

# ---------------------------------------------------------------------------
say "2. Data"
$PYTHON scripts/preflight.py data --root "$FLAIR_DATA_DIR" || die "data check failed (see above)"

# ---------------------------------------------------------------------------
say "3. Band order (are channels 0,1,2 Red,Green,Blue or Blue,Green,Red?)"
if [ -n "$BAND_ORDER" ]; then
    case "$BAND_ORDER" in rgb|bgr) echo "Using BAND_ORDER=$BAND_ORDER from the environment.";; *) die "BAND_ORDER must be rgb or bgr";; esac
else
    rm -f "$OUT/band_order.txt"
    $PYTHON scripts/check_bands.py --root "$FLAIR_DATA_DIR" --out "$OUT/band_check.png" \
        --write_order "$OUT/band_order.txt" || die "band check crashed (see above)"
    if [ ! -s "$OUT/band_order.txt" ]; then
        die "the band order could not be decided automatically.
  1. Open $OUT/band_check.png : the top row reads the channels as RGB, the bottom row as BGR.
     Soil should look brown/beige, vegetation green. Pick the row that looks natural.
  2. Re-run with   BAND_ORDER=rgb bash run.sh    or    BAND_ORDER=bgr bash run.sh"
    fi
    BAND_ORDER="$(tr -d '[:space:]' < "$OUT/band_order.txt")"
    echo "Decided automatically: BAND_ORDER=$BAND_ORDER  (confirm by eye: $OUT/band_check.png)"
fi

# ---------------------------------------------------------------------------
say "4. Model preflight"
OK_RUNS=()
SKIPPED=()
for spec in $RUNS; do
    IFS=: read -r model variant ckpt <<< "$spec"
    [ -n "$variant" ] || variant="$(default_variant "$model")"
    echo "--- $model ($variant)"
    if $(py_for "$model") scripts/preflight.py model --model "$model" --variant "$variant" --checkpoint "$ckpt"; then
        OK_RUNS+=("$model:$variant:$ckpt")
    else
        SKIPPED+=("$model:$variant")
        echo "  -> $model ($variant) will be SKIPPED"
    fi
done
[ "${#OK_RUNS[@]}" -gt 0 ] || die "no model is runnable; fix the problems above"
[ "$MODE" = "check" ] && { echo; echo "check only: ${#OK_RUNS[@]} model run(s) ready, ${#SKIPPED[@]} skipped."; exit 0; }

# ---------------------------------------------------------------------------
say "5. Episodes (one fixed list, shared by every model)"
$PYTHON scripts/make_episodes.py --root "$FLAIR_DATA_DIR" --band_order "$BAND_ORDER" \
    --out "$OUT/episodes.json" --n_per_class "$N_PER_CLASS" --domain_mode "$DOMAIN_MODE" --k_shot_max 5 \
    --seed "$SEED" --counts_cache "$OUT/class_counts.json" || die "episode generation failed (see above)"

# ---------------------------------------------------------------------------
say "6. Models"
DONE_DIRS=()
FAILED=()
for spec in "${OK_RUNS[@]}"; do
    IFS=: read -r model variant ckpt <<< "$spec"
    PY="$(py_for "$model")"
    ckpt_arg=(); [ -n "$ckpt" ] && ckpt_arg=(--checkpoint "$ckpt")
    base="${model}_${variant//\//-}"

    if [ "$SMOKE" = "1" ]; then
        echo "--- smoke test: $base (2 episodes, 1-shot)"
        if ! $PY scripts/run_model.py --model "$model" --variant "$variant" ${ckpt_arg[@]+"${ckpt_arg[@]}"} \
                --root "$FLAIR_DATA_DIR" --episodes "$OUT/episodes.json" \
                --out "$OUT/smoke/$base" --limit 2; then
            echo "  -> smoke test FAILED for $base; skipping this model"
            FAILED+=("$base"); continue
        fi
    fi

    for arm in $ARMS; do
        case "$arm" in
            1shot)  arm_args=(--prompt fewshot --k_shot 1) ;;
            5shot)  arm_args=(--prompt fewshot --k_shot 5)
                    [ -n "$K5_PER_CLASS" ] && arm_args+=(--per_class_limit "$K5_PER_CLASS") ;;
            oracle) arm_args=(--prompt oracle) ;;
            text)   if [ "$model" != "sam3" ] && [ "$model" != "mock" ]; then continue; fi
                    arm_args=(--prompt text) ;;
            *)      echo "  unknown arm '$arm' (use 1shot 5shot oracle text); skipping"; continue ;;
        esac
        tag="${base}_${arm}"
        echo "--- run: $tag   (resumable; re-run this script if interrupted)"
        if $PY scripts/run_model.py --model "$model" --variant "$variant" ${ckpt_arg[@]+"${ckpt_arg[@]}"} \
                --root "$FLAIR_DATA_DIR" --episodes "$OUT/episodes.json" --out "$OUT/runs/$tag" "${arm_args[@]}"; then
            DONE_DIRS+=("$OUT/runs/$tag")
        else
            echo "  -> run FAILED for $tag (partial results are kept and will resume next time)"
            FAILED+=("$tag")
        fi
    done
done
[ "${#DONE_DIRS[@]}" -gt 0 ] || die "no model finished; nothing to report"
ONE_SHOT=(); for d in "${DONE_DIRS[@]}"; do case "$d" in *_1shot) ONE_SHOT+=("$d");; esac; done

# ---------------------------------------------------------------------------
say "7. Report: metrics + figures"
# Main report: the one-shot task, all models, with the 10 figures.
if [ "${#ONE_SHOT[@]}" -gt 0 ]; then
    $PYTHON scripts/report.py --root "$FLAIR_DATA_DIR" --episodes "$OUT/episodes.json" \
        --runs "${ONE_SHOT[@]}" --out "$OUT/report" --n_figures 10 || die "report failed (see above)"
fi
# Every arm side by side (table only), on the episodes all of them completed.
$PYTHON scripts/report.py --root "$FLAIR_DATA_DIR" --episodes "$OUT/episodes.json" \
    --runs "${DONE_DIRS[@]}" --out "$OUT/report/all_arms" --n_figures 0 --n_boot 300 || echo "  (all-arms table failed; main report is unaffected)"
tar -czf "$OUT/report.tar.gz" -C "$OUT" report
echo
sed -n '1,30p' "$OUT/report/summary.md"

# ---------------------------------------------------------------------------
say "Done"
echo "Metrics : $OUT/report/summary.md   (+ per_class.csv)"
echo "Figures : $OUT/report/figures/     ($(ls "$OUT"/report/figures/*.png 2>/dev/null | wc -l | tr -d ' ') images, why each was chosen: manifest.json)"
echo "Bundle  : $OUT/report.tar.gz       (scp this)"
STATUS=0
if [ "${#SKIPPED[@]}" -gt 0 ]; then echo "SKIPPED (preflight): ${SKIPPED[*]}"; STATUS=2; fi
if [ "${#FAILED[@]}" -gt 0 ];  then echo "FAILED  (run)      : ${FAILED[*]}";  STATUS=2; fi
[ "$STATUS" -eq 0 ] || echo "!! The report above covers ONLY the models that finished. Fix the problems listed and re-run this script (finished work is kept)."
exit "$STATUS"
