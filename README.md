# SAM 1 vs SAM 2 vs SAM 3 on FLAIR (few-shot), and SAM 3 trained with RGB vs RGB+NIR

Three parts: (1) which Segment Anything model is best at one-shot land-cover segmentation on FLAIR
aerial imagery (frozen models, identical episodes, RGB only); (2) does SAM 3 get better when trained
with NIR; (3) does that NIR gain survive a change of camera, and how well does the model adapt to a
new region from a few labelled patches.

Student Research Project 2025/26 · Group 7 · Universität Hildesheim, ISMLL.

## Run it

```bash
pip install -r requirements.txt        # + sam2 from GitHub, see requirements.txt
hf auth login                          # once, for the gated SAM 3 weights (never commit a token)
bash run.sh prefetch                   # on a node WITH internet: caches SAM 3
bash run.sh check                      # preflight only: lists everything that is missing
bash run.sh                            # (or: sbatch run.sh) the whole experiment
```

Defaults assume `./data` (FLAIR), `./checkpoints/sam_vit_b_01ec64.pth` and
`./checkpoints/sam2.1_hiera_small.pt`. Everything is configurable by environment variable; the
full list is at the top of [run.sh](run.sh). Example with bigger models and explicit paths:

```bash
FLAIR_DATA_DIR=/path/to/flair SAM1_CKPT=/p/sam_vit_b_01ec64.pth SAM2_CKPT=/p/sam2.1_hiera_small.pt \
RUNS="sam1:vit_h:/p/sam_vit_h_4b8939.pth sam2:large:/p/sam2.1_hiera_large.pt sam3::" bash run.sh
```

**You get** `outputs/report/summary.md` (mIoU, FB-IoU, empty-prediction rate, baselines, paired
confidence intervals, per-class table), `outputs/report/figures/` (10 episodes, every model on the
same episode, correct/false-positive/missed pixels coloured), `outputs/report/all_arms/summary.md`
(every arm side by side) and `outputs/report.tar.gz`.

**Arms** (`ARMS="1shot 5shot oracle text"`): `1shot`/`5shot` are the few-shot task; `oracle` takes its
prompt from the query's own ground truth (an upper bound, never a method); `text` is SAM 3 with a
fixed class phrase. `K5_PER_CLASS=10` caps the 5x-more-expensive 5-shot arm.

**NDVI read-out and the deck** (after a run):

```bash
python scripts/ndvi_indicators.py --root ./data --run outputs/runs/sam3_facebook-sam3_1shot --out outputs/ndvi/sam3_1shot.json
python presentation/build_deck.py --root ./data          # -> presentation/out/SRP_Group7_Final.pptx
python presentation/preview.py presentation/out/SRP_Group7_Final.pptx   # layout QA without Keynote
bash presentation/render_keynote.sh presentation/out/SRP_Group7_Final.pptx   # .key/.pdf (macOS)
```

The deck is computed entirely from `outputs/runs` and `outputs/ndvi`: models or arms that have not run
are marked pending, never filled in.

## NIR robustness and few-shot adaptation (the contribution, 2026-11-17)

The RGB+NIR model relies on NIR for trees, so a drone camera with a different NIR response could break it. These stages
test that with simulated cameras, add NIR-robust training and label-free NIR calibration, and measure K-shot adaptation
to unseen regions (plan and decision rule: top of [PROJECT_STATE.md](PROJECT_STATE.md)).

```bash
ARMS=rgbiaug ZERO_SHOT=0 MAX_GPUS=1 bash run_train.sh   # NIR-robust training next to the existing runs
MAX_GPUS=1 bash run_train.sh shift                      # all finished runs under simulated NIR cameras (+ calibration)
MAX_GPUS=2 bash run_train.sh adapt                      # K = 1, 5, 20 labelled patches from unseen domains
bash run_train.sh robust_report                         # -> outputs/train/robust_report/summary.md
```

## Training experiment: does SAM 3 improve when trained to use NIR?

SAM 3 fine-tuned on FLAIR (LoRA on the image encoder + its built-in semantic head, one text prompt per
class), once with **RGB** and once with **RGB+NIR**, everything else identical (same seeds, images,
augmentations, prompts, starting weights). The NIR input filters start at zero, so the RGB+NIR model
starts as exactly pretrained SAM 3. Test regions (domains) are never seen in training.

```bash
tmux new -s sam3                       # it can run for days on a shared machine; survive logout
FLAIR_DATA_DIR=/path/to/flair bash run_train.sh check     # preflight only
FLAIR_DATA_DIR=/path/to/flair bash run_train.sh           # (or: bash run.sh train)
bash run_train.sh status                                  # from another shell: running / waiting / done
```

The GPUs are shared, so a memory probe runs first, then each run (`rgb_s0`, `rgbi_s0`, ... and
`zeroshot_rgb`) is started on whichever GPU has enough free memory, and re-queued (resuming from its
checkpoint) if another job takes the memory. Re-running the command resumes everything. Settings are at
the top of [run_train.sh](run_train.sh) (`SEEDS`, `STEPS`, `GPUS`, `TEST_LIMIT`, ...).

**You get** `outputs/train/report/summary.md` (mIoU per arm, the paired RGB+NIR − RGB difference with a
95% CI, per-class IoU, zero-shot reference), `outputs/train/report/figures/` (10 test patches chosen by
rule: RGB, NIR false colour, ground truth, each model) and `outputs/train_report.tar.gz`.

## What run.sh does

1. unit tests → 2. data check → 3. **band order** (RGB or BGR, decided from your files; stops if unsure)
→ 4. per-model preflight → 5. one fixed episode list → 6. smoke test then full run per model
(resumable) → 7. report.

A model that cannot run is skipped with the reason and the exit status is non-zero; the rest
still run. Results computed on different episode lists are refused, never merged.

## Protocol

- **SAM 1 / SAM 2:** identical PerSAM-style front end ([models/persam.py](models/persam.py)): support
  mask → feature prototype → cosine map on the query → point/box prompts → pick the candidate
  closest to the prototype. Only the encoder/decoder differ.
- **SAM 3:** Hugging Face `transformers`, FSS-SAM3 spatial-concatenation canvas with one exemplar
  box on the support half ([models/sam3_fss.py](models/sam3_fss.py)). SAM 3 cannot take a reference
  from another image, so this is its available few-shot recipe. The comparison is therefore
  *model + its best available few-shot recipe*, not bare encoders.
- **Episodes:** class covers 2%–90% of both patches (the cap stops "predict everything" scoring
  ~1.0); support ≠ query; cross-domain when the data has ≥ 2 FLAIR domains, otherwise same-domain
  with a loud warning.
- **Metrics:** class-accumulated mIoU (standard few-shot) as headline; mean-of-episodes mIoU and
  FB-IoU beside it; `all_fg` / `empty` baselines always listed; paired bootstrap on differences.
- **Figures:** 10 episodes chosen by a fixed rule after all runs (largest disagreement / all
  succeed / all fail), never by hand. `figures/manifest.json` records why each was chosen.

## Data

[FLAIR](https://ignf.github.io/FLAIR/) (IGN, France): 512×512 patches at 0.20 m, Open Licence 2.0 /
etalab-2.0 (attribution required). Both FLAIR #1 (`IMG_*/MSK_*`, classes 1–19) and FLAIR-HUB
(`*_AERIAL_RGBI_*` / `*_AERIAL_LABEL-COSIA_*`, labels 0–18 in a different order, remapped on read) are
supported. `scripts/download_toy.sh` fetches the FLAIR-HUB toy set (250 patches, 10 domains). Cite the FLAIR papers
([#1](https://arxiv.org/abs/2211.12979), [#2](https://arxiv.org/abs/2305.14467)).
**Channel order is not assumed** — the IGN dataset card says R,G,B,NIR,nDSM; earlier code assumed
B,G,R. `run.sh` settles it on your files.

## Layout

```
run.sh                       the one command
datasets/flair_rgb.py        RGB-only access; band_order is a required argument
evaluation/                  episodes, metrics, band decision rule, report helpers
models/persam.py             SAM 1 / SAM 2 (adapters + shared front end)
models/sam3_fss.py, canvas.py, prompting.py    SAM 3 canvas pipeline
scripts/                     check_bands, make_episodes, run_model, report, preflight, prefetch_weights
tests/                       no data/GPU/weights needed (test_run_sh.py runs run.sh end to end on synthetic TIFFs)
datasets/flair.py, ecological/   legacy (NDVI/Shannon composites); only helpers are still used
```

SAM licence: SAM 3 is under the SAM License (acknowledgement required in publications).
