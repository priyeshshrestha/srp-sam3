#!/bin/bash
# =============================================================================
# Does SAM 3 get better when it is TRAINED to use near-infrared?
# SAM 3 fine-tuned on FLAIR (LoRA + semantic head), RGB vs RGB+NIR, otherwise identical.
#
# Built for a SHARED GPU machine: runs start wherever and whenever a GPU has enough free memory,
# and the whole thing may take days or weeks. Start it so it survives logging out:
#
#   tmux new -s sam3        then   bash run_train.sh        (detach: Ctrl-b d;  back: tmux attach -t sam3)
#   or:   nohup bash run_train.sh > /dev/null 2>&1 &        (progress: tail -f logs/train_*.log)
#
#   bash run_train.sh            run everything (also: bash run.sh train)
#   bash run_train.sh check      preflight only (data, NIR band, split, weights, GPU memory); trains nothing
#   bash run_train.sh status     what is running / waiting / done
#   bash run_train.sh diagnose   does the RGB+NIR model use NIR? (filter sizes, loss curves, NIR ablation; 1 GPU)
#
# Runs:
#   rgb_s<seed>, rgbi_s<seed>   for every seed in SEEDS: train, then evaluate on held-out domains
#   zeroshot_rgb                untrained SAM 3 with the same 12 text prompts (what training buys)
#
# Output:   outputs/train/report/summary.md         mIoU table, RGBI - RGB with 95% CI, per class
#           outputs/train/report/figures/*.png      10 test patches, chosen by rule
#           outputs/train/report/per_class_iou.png
#           outputs/train_report.tar.gz             the same, ready to scp
#
# What protects you from mistakes:
#   * RGB and RGB+NIR runs of a seed see the same images, augmentations and prompts in the same
#     order, start from the same weights (NIR filters at zero = exactly pretrained SAM 3), and
#     train the same parts. The ONLY difference is the NIR channel.
#   * test domains are never seen in training (whole domains are held out)
#   * GPU sharing: memory is measured once (probe), then each run is started only on a GPU with
#     enough FREE memory (+ a margin), one of our runs per GPU, with the per-pass batch chosen to
#     fit (the effective batch, hence the experiment, never changes). If another user's job takes
#     the memory and ours runs out, it is re-queued and resumes from its checkpoint (saved every
#     10 minutes). Other users' processes are never touched.
#   * everything is resumable: re-run the same command after any interruption or reboot
#   * finished runs with different settings or a different split are refused, never mixed
#
# Configuration (environment variables, all optional):
#   FLAIR_DATA_DIR   dataset root (FLAIR #1 or FLAIR-HUB, 4+ bands)          [./data]
#   ARMS             which models to train: rgb | rgbi | rgb_ndvi | rgbi_ndvi   [rgb rgbi]
#                      rgb_ndvi  = VegSAM-style: the ViT sees RGB only; NIR enters ONLY through an NDVI
#                                  adapter added after every ViT block
#                      rgbi_ndvi = raw NIR as a 4th channel AND the NDVI adapter
#   PRIORITY         FLAIR class ids to focus the TRAINING data on, e.g. "5 6 7 8 10" (water, coniferous,
#                    deciduous, brushwood, herbaceous). Patches rich in them are drawn more often and those
#                    classes are prompted first; test set unchanged; reports add the priority-class mIoU   [none]
#   FOCUS_FLOOR      sampling weight of a patch without priority pixels, relative to an all-priority one's 1  [0.1]
#   SEEDS            training seeds                                         [0 1 2]
#   STEPS            optimizer steps per run                                [4000]
#   BATCH / ACCUM    max images per forward pass / passes per step; effective batch = BATCH x ACCUM [2 / 4]
#   LR               learning rate (patch embedding gets 0.1x)              [1e-4]
#   PROMPTS          class prompts per image per step                       [3]
#   LORA_RANK        LoRA rank on every ViT attention projection             [16]
#   TEST_LIMIT       evaluate a fixed random subset of the test set (all runs share it)  [all]
#   SPLIT            auto | hub_csv | official | domain  (auto: official FLAIR-HUB CSVs in <data>/splits if present, else a *test* folder,
#                    else hold out TEST_FRAC of the domains)                [auto]
#   TEST_FRAC        fraction of domains held out when SPLIT=domain         [0.2]
#   GPUS             GPU indices we may use, e.g. "0 3 5"                   [all]
#   MAX_GPUS         at most this many of our runs at once, always on the GPUs with the most free
#                    memory. Change it while running:  echo 1 > outputs/train/max_gpus    [no cap]
#   MARGIN_GB        free memory to leave on a GPU beyond our measured need [1.5]
#   PROBE_MIN_GB     free memory to wait for before the memory probe         [10]
#   WORKERS          data-loader workers per run                            [4]
#   ZERO_SHOT        1 = also evaluate untrained SAM 3                      [1]
#   BAND_ORDER       rgb | bgr  (skips the automatic check)
#   PYTHON           interpreter                                            [python, else python3]
#   SKIP_TESTS       1 = skip the unit tests                                [0]
#   MODEL            sam3 (the experiment) | tiny (pipeline test stand-in, NOT a result)  [sam3]
# =============================================================================
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="${1:-run}"
export FLAIR_DATA_DIR="${FLAIR_DATA_DIR:-./data}"
export OUT="${OUT:-./outputs}"
T="$OUT/train"
export PYTHON="${PYTHON:-$(command -v python || command -v python3 || echo python)}"
export MODEL="${MODEL:-sam3}"
export SEEDS="${SEEDS:-0 1 2}"
export ARMS="${ARMS:-rgb rgbi}"
export PRIORITY="${PRIORITY:-}"
export FOCUS_FLOOR="${FOCUS_FLOOR:-0.1}"
export STEPS="${STEPS:-4000}"
BATCH="${BATCH:-2}"
ACCUM="${ACCUM:-4}"
export LR="${LR:-1e-4}"
export PROMPTS="${PROMPTS:-3}"
export LORA_RANK="${LORA_RANK:-16}"
export TEST_LIMIT="${TEST_LIMIT:-0}"
SPLIT="${SPLIT:-auto}"
TEST_FRAC="${TEST_FRAC:-0.2}"
MARGIN_GB="${MARGIN_GB:-1.5}"
PROBE_MIN_GB="${PROBE_MIN_GB:-10}"
export WORKERS="${WORKERS:-4}"
ZERO_SHOT="${ZERO_SHOT:-1}"
export BAND_ORDER="${BAND_ORDER:-}"
SKIP_TESTS="${SKIP_TESTS:-0}"
export DEVICE="${DEVICE:-auto}"
GPUS="${GPUS:-}"
MAX_GPUS="${MAX_GPUS:-}"
DIAG_N="${DIAG_N:-1000}"

dev_for() { [ "$1" = "cpu" ] && echo cpu || echo cuda; }
gpu_env() { [ "$1" = "cpu" ] && echo "" || echo "$1"; }
common()  { echo --root "$FLAIR_DATA_DIR" --split "$T/split.json" --band_order "$BAND_ORDER"; }

# One run on one GPU: train (resumable) then evaluate (resumable). Exit 3 = out of GPU memory.
run_job() {   # run_job <tag> <gpu> <batch> <accum>
    local tag="$1" g="$2" b="$3" acc="$4" dir="$T/runs/$1" rc
    local dev; dev="$(dev_for "$g")"
    local -a C; read -r -a C <<< "$(common)"
    local -a FOCUS=(); [ -n "$PRIORITY" ] && FOCUS=(--focus "$T/focus.json")
    mkdir -p "$dir"
    if [ "$tag" = "zeroshot_rgb" ]; then
        CUDA_VISIBLE_DEVICES="$(gpu_env "$g")" $PYTHON scripts/eval_sam3.py --zero_shot --model "$MODEL" \
            --channels rgb "${C[@]}" --out "$dir" --limit "$TEST_LIMIT" --workers "$WORKERS" --device "$dev"
        rc=$?; [ $rc -eq 0 ] || return $rc
    else
        local ch="${tag%_s*}" seed="${tag##*_s}"
        CUDA_VISIBLE_DEVICES="$(gpu_env "$g")" $PYTHON scripts/train_sam3.py --model "$MODEL" --steps "$STEPS" \
            --lr "$LR" --prompts "$PROMPTS" --lora_rank "$LORA_RANK" --workers "$WORKERS" "${C[@]}" \
            --channels "$ch" --seed "$seed" --batch "$b" --accum "$acc" --out "$dir" --device "$dev" ${FOCUS[@]+"${FOCUS[@]}"}
        rc=$?; [ $rc -eq 0 ] || return $rc
        CUDA_VISIBLE_DEVICES="$(gpu_env "$g")" $PYTHON scripts/eval_sam3.py --run "$dir" "${C[@]}" \
            --limit "$TEST_LIMIT" --workers "$WORKERS" --device "$dev"
        rc=$?; [ $rc -eq 0 ] || return $rc
    fi
    touch "$dir/DONE"
}

# Internal: the GPU queue calls   bash run_train.sh _job <tag> <gpu> <batch> <accum>
if [ "$MODE" = "_job" ]; then
    [ -n "$BAND_ORDER" ] || { echo "internal error: BAND_ORDER not passed to the job"; exit 1; }
    run_job "$2" "$3" "$4" "$5"
    exit $?
fi
if [ "$MODE" = "diagnose" ]; then
    mkdir -p logs
    [ -n "$BAND_ORDER" ] || BAND_ORDER="$(tr -d '[:space:]' < "$T/band_order.txt" 2>/dev/null)"
    [ -n "$BAND_ORDER" ] || { echo "ERROR: BAND_ORDER unknown (set BAND_ORDER=rgb, or run run_train.sh first)"; exit 1; }
    [ -f "$T/split.json" ] || { echo "ERROR: $T/split.json missing: run the training first"; exit 1; }
    NEED="$($PYTHON -c "import json; m = json.load(open('$T/memory.json')); print(max(4.0, m['eval_gb']) + ${MARGIN_GB:-1.5})" 2>/dev/null || echo 7)"
    if command -v nvidia-smi >/dev/null 2>&1 && [ "$DEVICE" != "cpu" ]; then
        G="$($PYTHON scripts/gpu_queue.py wait --min_gb "$NEED" ${GPUS:+--gpus $GPUS})" || exit 1
        echo "NIR diagnostic on gpu $G ($DIAG_N test patches); log: logs/diagnose.log"
        CUDA_VISIBLE_DEVICES="$G" $PYTHON scripts/nir_diagnostic.py --train_dir "$T" --root "$FLAIR_DATA_DIR" \
            --band_order "$BAND_ORDER" --n_images "$DIAG_N" 2>&1 | tee -a logs/diagnose.log
    else
        $PYTHON scripts/nir_diagnostic.py --train_dir "$T" --root "$FLAIR_DATA_DIR" --band_order "$BAND_ORDER" \
            --n_images "$DIAG_N" --device cpu 2>&1 | tee -a logs/diagnose.log
    fi
    exit "${PIPESTATUS[0]}"
fi
if [ "$MODE" = "status" ]; then
    $PYTHON - "$T/queue_status.json" <<'EOF'
import json, os, sys
p = sys.argv[1]
if not os.path.exists(p):
    sys.exit("no queue has run yet")
s = json.load(open(p))
print(f"updated {s['updated']}  ({s['hours_running']:.1f} h since the queue started)"
      + (f"  |  GPU cap: {s['max_gpus']}" if s.get("max_gpus") else ""))
for t, r in s["runs"].items():
    extra = f"gpu {r['gpu']}, batch {r['batch']}" if r["status"] == "running" else ""
    print(f"  {t:14s} {r['status']:8s} {extra}  out-of-memory restarts {r['ooms']}, errors {r['fails']}")
EOF
    for f in "$T"/runs/*/train_log.jsonl; do
        [ -f "$f" ] && echo "  $(basename "$(dirname "$f")"): $(tail -n 1 "$f")"
    done
    exit 0
fi

NO_TMUX=0; [ -z "${TMUX:-}${STY:-}" ] && [ -t 1 ] && NO_TMUX=1
mkdir -p "$T/logs" logs
LOG="logs/train_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

say()  { echo; echo "=== $* ==="; }
die()  { echo; echo "ERROR: $*"; echo "(log: $LOG)"; exit 1; }

echo "SAM 3 RGB vs RGB+NIR training  |  $(date)  |  host $(hostname)"
echo "python: $($PYTHON --version 2>&1)  |  data: $FLAIR_DATA_DIR  |  out: $T  |  model: $MODEL"
[ "$MODEL" = "sam3" ] || echo "!! MODEL=$MODEL is a pipeline stand-in: its numbers are NOT results"
[ "$MODE" = "run" ] || [ "$MODE" = "check" ] || die "unknown mode '$MODE' (use: run | check | status)"
[ $NO_TMUX -eq 1 ] && [ "$MODE" = "run" ] && echo "!! Not inside tmux/screen: this may run for days. If your SSH session ends, it stops (it resumes when re-run)."

# ---------------------------------------------------------------------------
say "1. Environment and GPUs"
$PYTHON - <<'EOF' || die "torch is not usable in this environment"
import torch
print(f"  torch {torch.__version__}  cuda available {torch.cuda.is_available()}  devices {torch.cuda.device_count()}")
EOF
if [ -z "$GPUS" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && [ "$DEVICE" != "cpu" ]; then
        GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')"
    else
        GPUS="cpu"
    fi
fi
read -r -a GPU_LIST <<< "$GPUS"
ON_GPU=1; [ "${GPU_LIST[0]}" = "cpu" ] && ON_GPU=0
if [ $ON_GPU -eq 1 ]; then
    $PYTHON -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null || die "this machine has GPUs but torch in $PYTHON cannot use them
  (torch $($PYTHON -c 'import torch; print(torch.__version__)' 2>/dev/null); usually a torch build for a newer CUDA than
  the driver supports: see the 'driver ... too old' warning above). Use the environment whose torch matches the
  driver, e.g.  conda activate srp311   or   PYTHON=/path/to/env/bin/python bash run_train.sh check"
fi
echo "  GPUs we may use: ${GPU_LIST[*]}"
[ $ON_GPU -eq 1 ] && $PYTHON scripts/gpu_queue.py free
$PYTHON - <<'EOF' || die "Python packages are missing in $PYTHON (see the pip line above); nothing was run"
import importlib
need = {"numpy": "numpy", "scipy": "scipy", "tifffile": "tifffile", "imagecodecs": "imagecodecs",
        "matplotlib": "matplotlib", "PIL": "pillow", "huggingface_hub": "huggingface_hub", "transformers": "transformers"}
missing = []
for mod, pkg in need.items():
    try:
        importlib.import_module(mod)
    except Exception:
        missing.append(pkg)
if missing:
    print(f"  missing packages: {' '.join(missing)}\n  fix:  {__import__('sys').executable} -m pip install {' '.join(missing)}")
    raise SystemExit(1)
print("  packages ok: " + ", ".join(need.values()))
EOF
if [ "$MODEL" = "sam3" ]; then
    $PYTHON - <<'EOF' || die "transformers has no Sam3Model here: pip install -U 'transformers>=5' (SAM 3 needs a recent release)"
import transformers
from transformers import Sam3Model  # noqa: F401
print(f"  transformers {transformers.__version__}: Sam3Model available")
EOF
fi

# ---------------------------------------------------------------------------
if [ "$SKIP_TESTS" != "1" ] && [ "$MODE" = "run" ]; then
    say "2. Unit tests (no data, GPU or weights needed)"
    $PYTHON tests/test_train.py || die "unit tests failed: the code is broken in this environment; nothing was run"
fi

# ---------------------------------------------------------------------------
say "3. Data"
$PYTHON scripts/preflight.py data --root "$FLAIR_DATA_DIR" || die "data check failed (see above)"

say "4. Band order (are channels 0,1,2 Red,Green,Blue or Blue,Green,Red? NIR is channel 3 either way)"
if [ -n "$BAND_ORDER" ]; then
    case "$BAND_ORDER" in rgb|bgr) echo "Using BAND_ORDER=$BAND_ORDER from the environment.";; *) die "BAND_ORDER must be rgb or bgr";; esac
else
    rm -f "$T/band_order.txt"
    $PYTHON scripts/check_bands.py --root "$FLAIR_DATA_DIR" --out "$T/band_check.png" \
        --write_order "$T/band_order.txt" || die "band check crashed (see above)"
    [ -s "$T/band_order.txt" ] || die "the band order could not be decided automatically.
  Open $T/band_check.png (top row = RGB reading, bottom = BGR), pick the natural-looking row, and
  re-run with   BAND_ORDER=rgb bash run_train.sh   or   BAND_ORDER=bgr bash run_train.sh"
    BAND_ORDER="$(tr -d '[:space:]' < "$T/band_order.txt")"
    echo "Decided automatically: BAND_ORDER=$BAND_ORDER  (confirm by eye: $T/band_check.png)"
fi
export BAND_ORDER
read -r -a COMMON <<< "$(common)"

say "5. Train/test split (whole domains held out)"
$PYTHON scripts/make_split.py --root "$FLAIR_DATA_DIR" --out "$T/split.json" --mode "$SPLIT" \
    --test_frac "$TEST_FRAC" || die "split failed (see above)"

if [ -n "$PRIORITY" ]; then
    say "5b. Training-data focus on priority classes: $PRIORITY"
    $PYTHON scripts/make_focus.py --root "$FLAIR_DATA_DIR" --split "$T/split.json" --priority $PRIORITY \
        --floor "$FOCUS_FLOOR" --out "$T/focus.json" --counts_cache "$T/train_class_counts.json" \
        || die "focus failed (see above)"
fi

if [ "$MODEL" = "sam3" ]; then
    say "6. SAM 3 weights"
    $PYTHON scripts/preflight.py model --model sam3 --variant facebook/sam3 --checkpoint "" \
        || die "SAM 3 weights are not available (run: bash run.sh prefetch, on a node with internet)"
fi
[ "$MODE" = "check" ] && { echo; echo "check only: ready. GPUs we may use: ${GPU_LIST[*]}"; exit 0; }

# ---------------------------------------------------------------------------
TRAIN_ARGS=(--model "$MODEL" --steps "$STEPS" --lr "$LR" --prompts "$PROMPTS" --lora_rank "$LORA_RANK" --workers "$WORKERS")
EFF=$((BATCH * ACCUM))
MEM="$T/memory.json"
PROBE_ARM=rgbi; case " $ARMS " in *" rgb_ndvi "*) PROBE_ARM=rgb_ndvi;; esac
case " $ARMS " in *" rgbi_ndvi "*) PROBE_ARM=rgbi_ndvi;; esac                     # probe the largest arm
MEM_KEY="$MODEL p$PROMPTS r$LORA_RANK b$BATCH eff$EFF $PROBE_ARM"

say "7. Memory probe (3 training steps + 2 test images; measures what one run needs)"
if [ -f "$MEM" ] && grep -q "\"key\": \"$MEM_KEY\"" "$MEM"; then
    echo "  reusing $MEM (measured earlier for the same settings):"; cat "$MEM"; echo
else
    probe() {   # probe <gpu> <batch> -> exit code of the probe
        rm -rf "$T/probe"
        CUDA_VISIBLE_DEVICES="$(gpu_env "$1")" $PYTHON scripts/train_sam3.py "${TRAIN_ARGS[@]}" "${COMMON[@]}" \
            --channels "$PROBE_ARM" --seed 0 --batch "$2" --accum 1 --out "$T/probe" --probe 3 --device "$(dev_for "$1")"
    }
    need=$PROBE_MIN_GB
    while :; do
        if [ $ON_GPU -eq 1 ]; then
            G0="$($PYTHON scripts/gpu_queue.py wait --min_gb "$need" --gpus "${GPU_LIST[@]}")" || die "no usable GPU"
        else
            G0=cpu
        fi
        echo "  probing on gpu $G0 at batch 1"
        probe "$G0" 1; rc=$?
        [ $rc -eq 0 ] && break
        [ $rc -eq 3 ] && [ "${need%.*}" -lt 15 ] && { need=$((${need%.*} + 2)); echo "  -> out of memory; waiting for a GPU with $need GB free"; continue; }
        die "the training probe crashed (exit $rc, see above); nothing was trained"
    done
    B1="$($PYTHON -c "import json; print(json.load(open('$T/probe/probe.json'))['peak_gb'])")"
    SPP="$($PYTHON -c "import json; print(json.load(open('$T/probe/probe.json'))['s_per_step'] / 1)")"; PB=1
    B2=null; MAXB=1
    if [ "$BATCH" -ge 2 ]; then
        echo "  probing on gpu $G0 at batch 2"
        if probe "$G0" 2; then
            B2="$($PYTHON -c "import json; print(json.load(open('$T/probe/probe.json'))['peak_gb'])")"; MAXB=$BATCH
            SPP="$($PYTHON -c "import json; print(json.load(open('$T/probe/probe.json'))['s_per_step'])")"; PB=2
        else
            echo "  -> batch 2 did not fit on gpu $G0 right now; runs will use batch 1 (x accum $EFF)"
        fi
    fi
    rm -rf "$T/probe"
    CUDA_VISIBLE_DEVICES="$(gpu_env "$G0")" $PYTHON scripts/eval_sam3.py --zero_shot --model "$MODEL" --channels rgb \
        "${COMMON[@]}" --out "$T/probe_eval" --limit 2 --workers 0 --device "$(dev_for "$G0")" \
        || die "the evaluation probe crashed (see above); nothing was trained"
    EV=0; [ -f "$T/probe_eval/eval_mem.json" ] && EV="$($PYTHON -c "import json; print(json.load(open('$T/probe_eval/eval_mem.json'))['peak_gb'])")"
    rm -rf "$T/probe_eval"
    case "$B1" in nan) B1=0;; esac; case "$B2" in nan) B2=null;; esac; case "$EV" in nan) EV=0;; esac
    case "$SPP" in nan) SPP=null;; esac
    printf '{"key": "%s", "train_b1_gb": %s, "train_b2_gb": %s, "eval_gb": %s, "max_batch": %s, "effective_batch": %s, "s_per_pass": %s, "pass_batch": %s}\n' \
        "$MEM_KEY" "$B1" "$B2" "$EV" "$MAXB" "$EFF" "$SPP" "$PB" > "$MEM"
    echo "  memory model: $(cat "$MEM")"
fi
$PYTHON - "$MEM" "$STEPS" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
if m.get("s_per_pass"):
    h = m["s_per_pass"] * m["effective_batch"] / m["pass_batch"] * int(sys.argv[2]) / 3600
    print(f"  estimated training time per run: ~{h:.0f} h on a GPU shared like right now "
          f"({m['s_per_pass']:.1f} s per pass of {m['pass_batch']} images; test-set evaluation comes on top)")
EOF

# ---------------------------------------------------------------------------
say "8. Training + evaluation (queued onto whichever GPU has room; logs in $T/logs/)"
JOBS=()
for s in $SEEDS; do for ch in $ARMS; do
    case "$ch" in rgb|rgbi|rgb_ndvi|rgbi_ndvi) ;; *) die "unknown arm '$ch' in ARMS (use: rgb rgbi rgb_ndvi rgbi_ndvi)";; esac
    JOBS+=("${ch}_s${s}")
done; done
[ "$ZERO_SHOT" = "1" ] && JOBS+=("zeroshot_rgb")
echo "  runs: ${JOBS[*]}"
# Existing runs (finished or not) must have been started with the current settings and split.
$PYTHON - "$T/runs" "$T/split.json" "$MODEL" "$STEPS" "$LR" "$PROMPTS" "$LORA_RANK" "$EFF" "$BAND_ORDER" "${PRIORITY:+$T/focus.json}" <<'EOF' || die "existing runs were made with different settings (listed above). Re-run with the old settings, or move $T/runs away to start fresh. Runs are never mixed."
import glob, json, os, sys
runs, split, model, steps, lr, prompts, rank, eff, band = sys.argv[1:10]
focus = sys.argv[10] if len(sys.argv) > 10 else ""
want = {"model": model, "steps": int(steps), "lr": float(lr), "prompts": int(prompts), "lora_rank": int(rank),
        "effective_batch": int(eff), "band_order": band, "split_sha": json.load(open(split))["sha"],
        "focus_sha": json.load(open(focus))["sha"] if focus else ""}
bad = 0
for cfg_p in sorted(glob.glob(os.path.join(runs, "*", "config.json"))):
    cfg = json.load(open(cfg_p))
    diff = {k: (cfg.get(k), v) for k, v in want.items() if (cfg.get(k) if cfg.get(k) is not None else "") != v}
    if diff:
        bad += 1
        print(f"  {os.path.basename(os.path.dirname(cfg_p))}: different settings (was, now) {diff}")
sys.exit(1 if bad else 0)
EOF
echo "  progress at any time:   bash run_train.sh status"
if [ -n "$MAX_GPUS" ]; then
    echo "$MAX_GPUS" > "$T/max_gpus"
    echo "  GPU cap: at most $MAX_GPUS run(s) at once, on the GPUs with the most free memory"
    echo "           (change while running:  echo <n> > $T/max_gpus)"
else
    rm -f "$T/max_gpus"
fi
if [ $ON_GPU -eq 1 ]; then
    $PYTHON scripts/gpu_queue.py run --tags "${JOBS[@]}" --cmd "bash $(pwd)/run_train.sh _job" --mem "$MEM" \
        --runs_dir "$T/runs" --logs_dir "$T/logs" --status "$T/queue_status.json" \
        --gpus "${GPU_LIST[@]}" --margin_gb "$MARGIN_GB" ${MAX_GPUS:+--max_parallel "$MAX_GPUS"} \
        --max_parallel_file "$T/max_gpus"
else
    for tag in "${JOBS[@]}"; do      # CPU (tests only): one after another
        [ -f "$T/runs/$tag/DONE" ] && continue
        run_job "$tag" cpu "$BATCH" "$ACCUM" >> "$T/logs/$tag.log" 2>&1 || echo "  FAILED $tag (see $T/logs/$tag.log)"
    done
fi

DONE_DIRS=(); FAILED=()
for tag in "${JOBS[@]}"; do
    if [ -f "$T/runs/$tag/DONE" ]; then DONE_DIRS+=("$T/runs/$tag"); else FAILED+=("$tag"); fi
done
[ "${#DONE_DIRS[@]}" -gt 0 ] || die "no run finished; see $T/logs/"

# ---------------------------------------------------------------------------
say "9. Report (re-predicts the 10 figure patches, so it waits for a GPU with room)"
if [ $ON_GPU -eq 1 ]; then
    NEED_R="$($PYTHON -c "import json; m = json.load(open('$MEM')); print(max(4.0, m['eval_gb']) + $MARGIN_GB)")"
    GR="$($PYTHON scripts/gpu_queue.py wait --min_gb "$NEED_R" --gpus "${GPU_LIST[@]}")" || die "no usable GPU"
else
    GR=cpu
fi
CUDA_VISIBLE_DEVICES="$(gpu_env "$GR")" $PYTHON scripts/train_report.py --runs "${DONE_DIRS[@]}" "${COMMON[@]}" \
    --out "$T/report" --dates "$T/season/patch_dates.json" --device "$(dev_for "$GR")" || die "report failed (see above)"
tar -czf "$OUT/train_report.tar.gz" -C "$T" report

say "Done"
echo "Metrics : $T/report/summary.md   (+ per_class.csv, summary.json)"
echo "Figures : $T/report/figures/     ($(ls "$T"/report/figures/*.png 2>/dev/null | wc -l | tr -d ' ') images, why each was chosen: manifest.json)"
echo "Bundle  : $OUT/train_report.tar.gz   (scp this)"
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "FAILED: ${FAILED[*]}  (see $T/logs/<run>.log; re-run this script to resume them, finished work is kept)"
    exit 2
fi
exit 0
