# CLAUDE.md

## What this repo is (restart of 2026-09-30)

Two experiments:
1. **Few-shot comparison** (`run.sh`): SAM 1, SAM 2 and SAM 3, one-/five-shot and training-free, on identical
   **FLAIR RGB** episodes — numbers plus 10 qualitative figures.
2. **Training** (`run_train.sh`, added 2026-10-17): SAM 3 fine-tuned (LoRA + semantic head) on FLAIR, **RGB vs
   RGB+NIR**, identical otherwise — does SAM 3 get better when trained to use NIR? The EU UAV target has NIR.
3. **NIR robustness + few-shot adaptation** (added 2026-11-17, the contribution; plan at the top of `PROJECT_STATE.md`):
   NIR-robust training (`ARMS=rgbiaug`), simulated NIR cameras + label-free calibration (`shift`), K-shot adaptation to
   unseen domains (`adapt`), report (`robust_report`).
No elevation, ever. The earlier "ecological priors" direction is parked (history in `PROJECT_STATE.md`).

## Commands

```bash
bash run.sh check        # preflight only
bash run.sh prefetch     # cache SAM 3 weights (needs internet + `hf auth login`)
bash run.sh              # everything: tests -> data -> band order -> episodes -> models -> report
python tests/test_benchmark.py   # no data/GPU/weights;  also: test_flair.py, test_canvas_prompting.py
python tests/test_run_sh.py      # runs run.sh end to end on synthetic TIFFs (needs tifffile; skips otherwise)

bash run_train.sh check          # training experiment: preflight only   (also: bash run.sh train)
bash run_train.sh                # probe memory -> queue runs onto free GPUs -> evaluate -> report  (run in tmux)
bash run_train.sh status         # what is running / waiting / done
bash run_train.sh diagnose       # does the RGB+NIR model use NIR? filter sizes, loss curves, NIR ablation (1 GPU)
MAX_GPUS=2 bash run_train.sh     # cap: at most 2 of our runs at once, on the most-free GPUs (live: echo n > <T>/max_gpus)
ARMS=rgbiaug bash run_train.sh   # NIR-robust arm (random NIR gain/offset/gamma, 20% no NIR); pairs with rgbi_s<seed>
bash run_train.sh shift          # every finished run under simulated NIR cameras (+ calib:), fixed test subset (SHIFT_N)
bash run_train.sh adapt          # K-shot adaptation of ADAPT_RUNS to unseen test domains
bash run_train.sh robust_report  # outputs/train/robust_report/summary.md + gain_curve.png + kshot_curve.png (CPU)
python tests/test_train.py; python tests/test_gpu_queue.py; python tests/test_run_train_sh.py
```

Tests are plain asserts in `__main__`, not pytest. Run scripts from the repo root. **Keep the tests
data-free and GPU-free** — a test needing a download is a test nobody runs. `run.sh` is itself tested
(`test_run_sh.py`, using the explicit `mock` model): if you change `run.sh` or the stage scripts, run it.

## Rules that protect the science

- **`band_order` has no default** (`datasets/flair_rgb.py`). The IGN dataset card says R,G,B,NIR,nDSM;
  the old loader assumed B,G,R with no source. `run.sh` decides from the files (soil/water colour
  statistics + an rgb/bgr montage) and **stops rather than guessing**. A wrong order feeds every model
  a colour-swapped image. All Phase 3 numbers are unverified for this reason.
- **Every model runs the same `episodes.json`.** Eligibility is data-only (class covers 2%-90% of both
  patches, support != query, cross-domain when >= 2 domains). Never filter episodes per model. Each
  run records `episodes_sha`; `run_model.py` and `report.py` refuse to mix different episode files.
- **Always show the `all_fg` baseline** beside model numbers. A class that fills the frame makes
  "predict everything" score ~1.0 (how an earlier 51.64% IoU was meaningless). Also report the
  empty-prediction rate (SAM 3 is instance-based and may return nothing on "stuff" classes).
- **Figures are chosen by rule after all runs** (`evaluation/report_lib.select_figure_episodes`),
  never by hand.
- **Fairness caveat:** SAM 1/2 use PerSAM-style prompting; SAM 3 uses the FSS-SAM3 canvas + one exemplar
  box. The comparison is *model + its best available few-shot recipe*. Say so whenever quoting numbers.
- **Arms:** `1shot`/`5shot` are the task; `oracle` takes its prompt from the query's own ground truth
  (upper bound, never a method); `text` is SAM 3 with a fixed class phrase (`CLASS_PHRASES`, chosen
  before any result). 5-shot uses `supports[:5]` of the same episodes, so 1- and 5-shot stay paired.
- **The deck contains no typed-in results.** `presentation/build_deck.py` computes every number, chart
  and prediction image from `outputs/runs`, `outputs/ndvi` and `outputs/train/report` (training experiment); only the recap numbers (previous
  presentation) and cited literature facts are fixed text. Do not add invented or placeholder results.
- **Training: RGB and RGBI runs differ ONLY in the input channels.** Same seed -> same images, augmentations,
  prompts (`rng([seed, item])` in `train/data.py`) and same LoRA init; NIR filters start at zero (= exactly
  pretrained SAM 3). Test domains are held out whole. `run_train.sh` refuses runs made with other settings.
  Report the paired RGBI - RGB difference with its CI and the zero-shot reference, never one arm alone.
- **NIR robustness:** transforms touch the NIR band only (`train/nir.py`), so the RGB model is the fixed reference in
  every condition. The robust arm draws its NIR augmentation from a separate RNG stream: same images/augmentations/
  prompts as `rgbi_s<seed>`, only NIR differs. Always report the out-of-training-range conditions (gain 0.6 / 1.4)
  separately, and never tune the a-priori settings (augmentation ranges, test cameras, adaptation recipe) on results.
- **The cluster GPUs are shared with other students.** Never assume a free GPU; `scripts/gpu_queue.py` starts
  runs only where free memory fits, and every run must stay resumable (exit 3 = out of memory = re-queue).
- **`mock` / `tiny` are not models.** It exists to test the pipeline; runs are tagged `is_measurement: false` and
  the report prints a warning. Never cite its numbers.
- **Secrets:** never write tokens, passwords, IPs or hostnames into the repo. Hugging Face auth comes
  from `hf auth login` / `HF_TOKEN`; checkpoints come from `--checkpoint` / env vars, never hardcoded.
  The repository is **public**: no student IDs, local paths or cluster addresses either (IDs live in the git-ignored
  `presentation/team_ids.local.txt`). The old commit with a cluster password was removed before the first push (2026-11-08).

## Code map

| File | Role |
|---|---|
| `run.sh` | The one command; preflight, per-model isolation, resumable, non-zero exit if anything was skipped |
| `datasets/flair_rgb.py` | RGB patches + masks, class counts, domains; all paths relative to the data root |
| `evaluation/episodes.py` | Episode generation, `episodes_hash` |
| `evaluation/metrics.py` | IoU, class-accumulated mIoU, FB-IoU, baselines, paired bootstrap |
| `evaluation/bands.py` | `decide_band_order` (conservative; None when unsure) |
| `evaluation/report_lib.py` | Loading runs, figure-episode selection |
| `models/persam.py` | SAM 1 / SAM 2 adapters + shared prototype-matching front end |
| `models/sam3_fss.py`, `canvas.py`, `prompting.py` | SAM 3 via transformers + canvas geometry + exemplar-box choice |
| `scripts/` | `check_bands`, `make_episodes`, `run_model` (arms: `--prompt fewshot/oracle/text`, `--k_shot`), `report`, `ndvi_indicators`, `preflight`, `prefetch_weights`, `download_toy.sh` |
| `run_train.sh` | Training experiment: preflight, memory probe, GPU queue, train + eval per run, report |
| `train/sam3_rgbi.py` | SAM 3 with widened (zero-init NIR) patch embedding, LoRA, semantic head; `TinySeg` stand-in |
| `train/data.py`, `train/metrics.py` | RGBI loading, held-out-domain split, paired training schedule; mIoU, paired bootstrap |
| `scripts/train_sam3.py`, `eval_sam3.py`, `train_report.py`, `gpu_queue.py`, `make_split.py` | Training-experiment stages |
| `train/nir.py` | NIR-only transforms: simulated cameras, robust-training augmentation, moment-matching calibration |
| `scripts/nir_shift.py`, `adapt_kshot.py`, `robust_report.py` | NIR-shift evaluation, K-shot adaptation, robustness report |
| `presentation/build_deck.py` | Builds the final deck (template of the previous presentation) **only from real runs**; `render_keynote.sh` exports .key/.pdf/PNGs |
| `datasets/flair.py`, `ecological/` | Legacy composite/NDVI/Shannon code. Only `_read_tif`, `mask_path_for`, `domain_of`, `FLAIR_CLASSES` are used. |

## FLAIR facts

Two releases are supported and both are read into **FLAIR #1 class ids (1-19)** everywhere:

- **FLAIR #1** (what Phase 3 used): `.../D041_2021/aerial/Z*/IMG_*.tif` + `labels/.../MSK_*.tif`,
  5 channels, masks 1-19 with **no class 0**.
- **FLAIR-HUB** (what `scripts/download_toy.sh` actually fetches): `D069-2020_AERIAL_RGBI/<roi>/
  D069-2020_AERIAL_RGBI_<roi>_<x-y>.tif` (4 bands **R,G,B,NIR**, uint8, LZW -> needs `imagecodecs`)
  + `..._AERIAL_LABEL-COSIA_...` labels. **COSIA labels are 0-18 in a different order** (0 = building,
  12 = deciduous, 13 = coniferous...). `datasets/flair_rgb.HUB_TO_FLAIR1` remaps them (source: class
  table in github.com/IGNF/FLAIR-HUB); `read_label` applies it. Never read HUB labels with `_read_tif`
  directly. The toy set has 250 aerial patches from 10 domains (2019-2021).
- **Band order:** FLAIR-HUB is `rgb`, confirmed on the files (soil ch0-ch2 = +20.5, water -28.8, montage).
  FLAIR #1 is still unverified on files; its dataset card says R,G,B,NIR,nDSM while the legacy loader
  assumed B,G,R, so **Phase 3 very likely ran on colour-swapped images.**
- Domains: `D041_2021` / `D041-2021`; `domain_of` normalises both to `D041_2021`.
- Default evaluation classes are 1-12. NIR (channel 3 in both releases) is used only by
  `scripts/ndvi_indicators.py` and the training experiment (`train/`); height is never used (the EU UAV
  target has NIR but no height).
- Licence: Open Licence 2.0 / etalab-2.0 (attribution). SAM 3: SAM License (acknowledgement required in publications).

## Model facts (verified where marked)

- SAM 3 through Hugging Face `transformers` (`Sam3Model`, `Sam3Processor`) runs on Python 3.11 (verified on a
  Mac/MPS; the cluster has transformers 5.18.0, the same version, torch 2.5.1+cu121, driver CUDA 12.2).
  SAM 3 has **840.6 M** parameters (vision 454 M, text 353.5 M). The only 3-channel layer is the patch
  embedding (Conv2d 3->1024, 14x14, no bias). `Sam3VisionModel` refuses HF gradient checkpointing; `train/`
  wraps layers itself. Its built-in `semantic_seg` head (one map per prompt, 288x288) is what we train.
- Cluster: 8x RTX A4000 16 GB, **shared** (other students' jobs use 1-13 GB per card at any time). The native
  `sam3` package, not used here, reportedly needs Python >= 3.12 / CUDA >= 12.6 (unverified).
- SAM 3 cannot take an exemplar from another image; the canvas (support + query on one 1008x1008 image,
  support gets 60%) is the workaround. The support *mask* never reaches the model, only one box.
- SAM 2 has no public embedding accessor; `persam.py` reads `predictor._features["image_embed"]`
  (verified with sam2 1.1.0). SAM 1/2 need square inputs here (FLAIR patches are).
- Previous result (unverified, possibly band-swapped): SAM 1 ViT-B + PerSAM, 1-shot, mIoU 0.286 on one domain.
