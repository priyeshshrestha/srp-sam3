# PROJECT STATE — full handoff

## PLAN 2026-11-17 — 15 days left (deadline ~2 Dec): NIR robustness + few-shot adaptation

**This is the plan until the deadline. Everything below this section is history.**

### The story (our contribution)
1. Training SAM 3 with NIR helps, mostly vegetation: +0.0095 mIoU [+0.008, +0.011], coniferous +0.06 (Exp. 3, done).
2. But the RGB+NIR model **depends on NIR for trees** (NIR flattened: coniferous −0.21, deciduous −0.23 IoU) — a drone
   camera with a different NIR response could break it (NIR diagnostic, done).
3. **New:** (A) **NIR-robust training** (random NIR gain/offset/gamma + 20% "no NIR") and (B) **label-free NIR
   calibration** (match each new domain's NIR mean/std to FLAIR's, from unlabelled images), evaluated under simulated
   camera changes, plus **K-shot adaptation to unseen regions** (K = 1, 5, 20 labelled patches) — the drone scenario.

Not novel on its own (be honest): SAM + few-shot (ViRefSAM, FSS-SAM3), SAM + NIR/NDVI (VegSAM), the +1-point NIR gain,
the SAM 1/2/3 benchmark. The contribution is measuring the NIR dependence of a NIR-extended SAM 3 and fixing it for
cross-camera / cross-region few-shot transfer. **Literature check still needed** on NIR/channel robustness for SAM-type
models on aerial imagery (teammates, this week).

### What is trustworthy
| Work | Status |
|---|---|
| Band-order bug found + old numbers retired; evaluation protocol (cross-region, all_fg baseline, paired bootstrap, rule-chosen figures) | solid |
| Exp. 3 RGB vs RGB+NIR (official split, 3 seeds, 4k + 12k steps) | solid — main measured result |
| NIR diagnostic (tree reliance) | solid (4k models) |
| NDVI read-out (r = 0.86) / NDVI filter hurts | toy subset only |
| Few-shot SAM 1/2/3 | preliminary (toy, no SAM 1) — optional rerun only if a GPU is idle |
| Never cite | 0.286, 51.64 %, the 0.0709→0.0560 "regression", HRNet recap until 0.580 vs 0.5134 is settled |

### New code (branch `robust-nir-fewshot`)
| File | Role |
|---|---|
| `train/nir.py` | NIR-only transforms: simulated cameras (`SHIFTS`), `random_nir` (robust training), `calibrate` (moment matching), `NIR_MISSING` = 127.5 (normalises to exactly 0 = "no NIR") |
| `train/data.py` | `TrainSchedule(nir_aug=True)`: NIR augmentation from a separate RNG stream -> same images/augmentations/prompts as `rgbi_s<seed>` |
| `scripts/train_sam3.py` | `--nir_aug`; `nir_aug` is a run key (old configs read as `False`) |
| `scripts/nir_shift.py` | one run under all NIR conditions on a fixed test subset (paired), incl. `calib:<condition>` |
| `scripts/adapt_kshot.py` | K-shot adaptation of a run to unseen test domains, conditions normal / camera_b / calib:camera_b |
| `scripts/robust_report.py` | tables (shift, trees, adaptation) with paired 95% CIs + `gain_curve.png`, `kshot_curve.png` |
| `run_train.sh` | `ARMS=rgbiaug`, modes `shift`, `adapt`, `robust_report` (separate GPU queue in `outputs/train/extra`) |
| tests | `tests/test_train.py` (NIR transforms, calibration, pairing, old-config resume), `tests/test_run_train_sh.py` (all new stages end to end with `tiny`) |

Settings fixed a priori (not tuned): augmentation p_missing 0.2, p_clean 0.2, gain U(0.7, 1.3), offset U(−25, 25),
gamma log-U(0.8, 1.25); test cameras gain 0.6/0.8/1.2/1.4, gamma 1.25, camera_b = gain 0.8 + offset 15 + gamma 1.15;
gain 0.6 and 1.4 are OUTSIDE the training range (report them separately). Adaptation: 100 steps, 4 images/step,
lr 5e-5 (patch embedding 0.1×), 4 domains, 200 evaluation patches per domain, supports nested (K=1 ⊂ 5 ⊂ 20).

### Commands, in order (cluster, `conda activate srp311`, inside tmux; `FLAIR_DATA_DIR` as before)
```
python tests/test_train.py && python tests/test_run_train_sh.py              # 3 min, no GPU
# GPU 1: robust training, 4,000 steps, next to the existing 4k runs in outputs/train (~8-15 h per seed)
ARMS=rgbiaug ZERO_SHOT=0 MAX_GPUS=1 bash run_train.sh
# GPU 2, at the same time: the existing 4k models under simulated cameras (DECISION POINT, a few hours)
MAX_GPUS=1 bash run_train.sh shift && bash run_train.sh robust_report
# when the robust runs are done (both GPUs):
MAX_GPUS=2 bash run_train.sh shift        # only the new runs are evaluated; finished ones are skipped
MAX_GPUS=2 bash run_train.sh adapt        # rgb_s0, rgbi_s0, rgbiaug_s0 -> ~6-8 h per NIR run
bash run_train.sh robust_report           # -> outputs/train/robust_report/summary.md + 2 figures
```
Never more than 2 GPUs in total (training queue + shift/adapt queue together).

### Decision rule after the first `shift` report (day 1-2)
- RGB+NIR falls below RGB under a plausible camera (gain 0.8 / camera_b) or without NIR -> the risk is real; main
  claim = robust training and/or calibration remove it while keeping the clean-NIR gain.
- RGB+NIR does NOT fall below RGB even at gain 0.6 / 1.4 -> report that honestly ("the NIR gain is robust to camera
  changes of this size"); the main claim moves to K-shot adaptation to new regions.
Either way the paper has a measured answer. Do not change the a-priori settings after seeing results.

### 15-day schedule
| Days | Dates | Do |
|---|---|---|
| 1 | Tue 17 – Wed 18 Nov | apply patch, run tests; start robust training (GPU 1) + `shift` on existing models (GPU 2); first report = decision point |
| 2–4 | Wed 18 – Sat 21 | robust training finishes (3 seeds; if slower than expected, `SEEDS="0 1"` is the fallback). Optional on the idle GPU: few-shot SAM 1/2/3 rerun on full data |
| 4–5 | Sat 21 – Sun 22 | `shift` on robust runs, `adapt`, `robust_report` |
| 6 | Mon 23 | results freeze; copy `outputs/train/robust_report/` + `report/` to the Mac |
| 7–11 | Tue 24 – Sat 28 | write-up (problem → method → Exp. 3 → diagnostic → robustness → adaptation → limits), figures, slides |
| 12–13 | Sun 29 – Mon 30 | rehearsal, Q&A prep |
| 14–15 | Tue 1 – Wed 2 Dec | buffer |

### Dropped for time
PANet head / fusion strategies A/B/D/D′, ESIA/PCSA adapters, Shannon-as-input arm, ISPRS Potsdam, LoveDA, 7-class split,
NIR-gain retuning, new training lengths. The 12k-step models stay the headline of Exp. 3; robustness uses the 4k models
(the NIR gain was identical at 4k and 12k).

### Open, owned by teammates
1. EU UAV data access (still the biggest risk; the simulated cameras are the fallback — say so in the paper).
2. Literature check on NIR/channel robustness of SAM-type models (decides how strongly we claim novelty).
3. Published FLAIR-HUB baselines (we train on 21 of 58 training domains — state it).
4. ViRefSAM provenance question before the write-up; HRNet recap number.

---

## RESULT 2026-10-28 — 12,000-step runs + NIR diagnostic (4,000-step models)

- **12,000 steps** (`outputs/long/train` on the cluster; report copied to `outputs/long_report/` on the Mac): RGB
  **0.6220 ± 0.0030**, RGB+NIR **0.6314 ± 0.0003**; **Δ = +0.0095 [+0.0082, +0.0108]** — identical to 4,000 steps.
  Both arms +0.016. Per class: coniferous +0.057, deciduous +0.021, herbaceous +0.017; 7/12 significant gains;
  pervious loss gone (+0.001 n.s.); only significant loss building −0.003. RGB+NIR has far lower seed spread both times.
- **NIR diagnostic** (4,000-step models, 1,000 test patches): real NIR 0.630 → flattened 0.588 → swapped 0.571
  (RGB-only 0.625); coniferous −0.21 / deciduous −0.23 IoU with NIR flattened → strong tree reliance on NIR = transfer
  risk for a different NIR camera. Filter-norm section not yet seen. Loss still falling mid-training at 4,000 steps.
- Deck now presents the 12,000-step report as main result (`--train_main_report outputs/long_report`), 4,000-step as
  comparison line; `docs/group_update.md` + `docs/technical_nir_training.md` → PDFs via `docs/md_to_pdf.py`.

## NEXT (2026-10-21) — longer training + NIR diagnostic

- **Diagnostic** (`bash run_train.sh diagnose`, `scripts/nir_diagnostic.py`): NIR vs RGB filter norms of the trained
  patch embedding (NIR started at 0, had lr x0.1), RGB filter drift from pretrained, loss per eighth of training,
  and an NIR ablation on 1,000 fixed test patches (real / flattened / swapped NIR vs the RGB-only model).
- **Longer runs**: same recipe, `STEPS=12000` (~2.5 epochs of the 37,900 train patches vs <1), in `OUT=./outputs/long`
  so the 4,000-step results stay intact; `ZERO_SHOT=0`; `MAX_GPUS=2` (user: never more than the 2 most-free GPUs).
  Changing only the step count keeps "train longer" a clean question; a higher NIR learning rate is a separate run,
  decided by the diagnostic.

## RESULT 2026-10-21 — training experiment finished on the cluster (all 7 runs, 0 errors)

Official FLAIR-HUB split: 37,900 train patches (21 domains, ~100 GB download incl. test) / **50,700 test patches
(16 held-out domains)**; 4,000 steps × effective batch 8; 3 seeds; mIoU over classes 1-12.
- **SAM 3 zero-shot (text prompts, semantic head): 0.1801.**
- **Fine-tuned RGB: 0.6062 ± 0.0021** (0.6092, 0.6048, 0.6047).
- **Fine-tuned RGB+NIR: 0.6157 ± 0.0004** (0.6162, 0.6159, 0.6151).
- **Δ(RGBI − RGB) = +0.0095, 95% CI [+0.0084, +0.0107]**, positive for every seed (+0.0069, +0.0112, +0.0104).
- Per class (mean over seeds): coniferous **+0.065** (0.555→0.620), deciduous +0.018, herbaceous +0.018, plowed
  +0.008, water +0.007, agricultural +0.007, bare soil +0.005; pervious surface **−0.011**; building −0.004,
  vineyard −0.003, impervious −0.001. Per-class CIs not in the first report → added (report-only rerun).
- First report's 10 figures were uninformative (rule picked single-class patches with per-image mIoU 0/1) →
  figure rule now requires ≥3 classes each ≥5% of the patch. Rerun of the report needed; no retraining.
- Speed on shared A4000s: 7.4–13 s per optimizer step; peak 9.2 GB (batch 2) / 5.8 GB (batch 1).
- Report rerun with per-class CIs: **8 of 12 classes gain significantly** (coniferous, deciduous, herbaceous, plowed,
  agricultural, water, bare soil, brushwood); significant losses: pervious −0.011, building −0.004, impervious −0.001;
  vineyard n.s. Report copied to the Mac at `outputs/train/report/`.
- Deck now 22 slides: proposal slide "adaptive SAM 3" replaced (when a training report exists) by Experiment 3 slides
  17–19 (method, results with Δ-per-class CI chart, rule-chosen helped/hurt figures); conclusion + future work updated;
  speaker script rewritten for 15–22 (~21–22 min). Slide 17 shows "—" for training-set size until
  `outputs/train/split.json` is copied from the cluster (or the report is re-run, which now records `setup`).

## UPDATE 2026-10-17 — training experiment: SAM 3 fine-tuned, RGB vs RGB+NIR (`run_train.sh`)

**Question:** does SAM 3 get better when TRAINED to use NIR? Frozen SAM cannot use NIR (Phase 3: CIR
input 0.286 → 0.254; NDVI gate 0.334 → 0.212), but NDVI on SAM 3 segments tracks truth (r = 0.86).
The EU UAV target data **has NIR** (user, 2026-10-17), so an RGBI model transfers to it.

**Design (verified on facebook/sam3, transformers 5.18.0 — same version as the cluster):**
- Patch embedding `vision_encoder.backbone.embeddings.patch_embeddings.projection` (Conv2d 3→1024, 14×14,
  no bias) is the ONLY 3-channel layer. Widened to 4 with NIR filters **at zero**: output identical to
  pretrained SAM 3 (max diff 0.0); NIR filters still get gradient. (Mean-of-RGB init shifts features by up to 236 — not used.)
- Trainable 16.5 M: patch embedding (lr ×0.1), LoRA r=16 on q/k/v/o of all 32 ViT layers (128 layers, B=0 →
  exact no-op at init), DETR encoder, semantic path of the mask decoder. Text encoder (353 M) dropped after
  caching the 12 `CLASS_PHRASES` features.
- Output = SAM 3's built-in `semantic_seg` head, one map per class prompt; loss BCE + Dice; test = argmax over
  12 prompts. My forward (`Sam3Semantic.encode/decode`) equals `Sam3Model(...).semantic_seg` (diff 0.0 on CPU).
- Control: RGB run of the same seed differs ONLY in channels (same images/augmentations/prompts via
  `rng([seed, item])`, same LoRA init — re-seeded after widening). Split = whole held-out domains
  (official `*test*` folder if present, else 20% of domains). 3 seeds; paired bootstrap over test images.
- Zero-shot SAM 3 (same prompts, no training) evaluated as the "what training buys" reference.
- **SAM 3's size is 840.6 M** (454 vision + 353.5 text + ~33 heads), not 860 M — deck/notes/script corrected.

**Shared cluster:** 8× RTX A4000 16 GB, all partly used by other students (free 3.6–15.4 GB on 2026-10-17).
`scripts/gpu_queue.py` starts each run on whichever GPU has enough FREE memory (measured memory model +
1.5 GB margin), one of ours per GPU, batch chosen to fit (effective batch fixed = 8); OOM (exit 3) → re-queue,
resume from checkpoint (every 10 min). Timeline: ~2 months available, runs whenever GPUs free up.
Run inside tmux: `bash run_train.sh`; progress: `bash run_train.sh status`.

**Data on the cluster (checked 2026-10-18):** only FLAIR #1 domain D041 (2,162 patches in `~/srp_phase2/data`,
`~/data/D041_2021`, and the same packed as `~/flair_dataset` tar) — one domain, unusable for a held-out-domain test —
plus the **official FLAIR-HUB split CSVs** (`~/srp_phase2/data/splits/FLAIR-HUB_{TRAIN,VALID,TEST}.csv`).
FLAIR-HUB on Hugging Face (`IGNF/FLAIR-HUB`, public) has per-domain zips: AERIAL_RGBI 74 domains = 241 GB,
AERIAL_LABEL-COSIA = 1.8 GB. `scripts/download_flair_hub.py` fetches all official TEST domains + train domains up to
`--budget_gb` (default 100), and `make_split.py` uses the official CSV split (`hub_csv`, VALID unused). Home is a
shared 40 TB NFS with 5.6 TB free (87% used); the GPU server reaches huggingface.co. Use env `srp311` (the `srp` venv's
torch 2.14+cu130 cannot use the 12.2 driver; run.sh/run_train.sh now refuse that).

**Verified locally:** a real SAM 3 RGBI training step on toy FLAIR (forward + backward + step, MPS). Needed
own gradient checkpointing (HF refuses it for Sam3VisionModel) on ViT **and** DETR-encoder layers (the
DETR encoder's 5184-token attention OOMed a 20 GB Mac otherwise). **Not yet measured:** memory and speed on
A4000s (the probe stage of `run_train.sh` measures it first).

## UPDATE 2026-10-01 — first REAL runs on FLAIR-HUB (Mac), final-phase deck generator

**Data on the Mac now:** `data/FLAIR-HUB_TOY` (from `scripts/download_toy.sh`, 1.5 GB extracted): 250 aerial
RGBI patches from **10 domains** (D006, D012, D032, D038, D044, D051, D068, D069, D083, D085; 2019–2021).
Multi-domain, so episodes are genuinely **cross-domain** (Phase 3's data was one domain).

**Findings, all measured on the files:**
- `download_toy.sh` fetches **FLAIR-HUB**, not FLAIR #1: different layout (`*_AERIAL_RGBI_*` / `*_AERIAL_LABEL-COSIA_*`,
  no IMG/MSK), 4 bands, LZW-compressed. `run.sh` would have failed on it; the loader now supports both.
- **COSIA labels are 0–18 in a different class order** from FLAIR #1's 1–19 (e.g. HUB 12 = deciduous, #1 12 = plowed).
  Remapped on read (`HUB_TO_FLAIR1`, source: IGNF/FLAIR-HUB README), with tests.
- **FLAIR-HUB band order is R,G,B,NIR** (`check_bands.py`: soil ch0−ch2 = +20.5, water −28.8, montage unambiguous).
  FLAIR #1 still unverified on files; the legacy loader's B,G,R assumption means **Phase 3 very likely ran on
  colour-swapped images**. Run `bash run.sh check` against the cluster's FLAIR #1 copy to settle it.
- Episodes: 300 (12 classes × 25), support and query from different domains, 5 supports stored per episode.
  Vineyard (13 eligible patches) and plowed land (12) reuse ~10–11 distinct queries; other classes 17–24.

**Real results (preliminary: FLAIR-HUB toy subset, Apple M1 Pro GPU):** see `outputs/report/summary.md`,
`outputs/report/all_arms/summary.md` and `outputs/ndvi/*.json`. Filled in below once all arms finished.
- SAM 2.1 Hiera-S, 1-shot: class-accumulated mIoU **0.266** (300 episodes).
- NDVI read-out from SAM 2 segments: segment-NDVI vs ground truth r = 0.73 (|err| 0.08); **the support-derived
  NDVI gate hurts** (0.266 → 0.206) — an NDVI range learned in one domain/date does not transfer to another.
- **SAM 1 not run on the Mac** (checkpoint download not permitted here) → run on the cluster.

**New code:** arms in `run_model.py`/`run.sh` (`ARMS="1shot 5shot oracle text"`, `K5_PER_CLASS`),
`scripts/ndvi_indicators.py`, `presentation/build_deck.py` (+ `render_keynote.sh`). The deck is generated from
the runs only — no hand-typed or placeholder results; a model/arm that has not run is marked pending.

## UPDATE 2026-09-30 (evening) — restart: SAM 1 vs SAM 2 vs SAM 3 on FLAIR, RGB only

**Current goal (supersedes the ecological-priors framing below until this is done):** run SAM 1,
SAM 2 and SAM 3 on the same FLAIR RGB episodes, one-shot and training-free, and produce metrics
and qualitative figures that say which model is best on this dataset. Nothing else for now — no
NIR, NDVI, Shannon, composites, or elevation.

### What exists (all tested without data/GPU; `bash run.sh` is tested end to end on synthetic TIFFs)

**Run it on the cluster:** `bash run.sh check` (preflight) → `bash run.sh` (or `sbatch run.sh`). It stops with
instructions if the band order cannot be decided; set `BAND_ORDER=rgb|bgr` after reading `outputs/band_check.png`.
Outputs: `outputs/report/summary.md`, `outputs/report/figures/` (10 images), `outputs/report.tar.gz`.

| File | Role |
|---|---|
| `run.sh` | The one command: tests → data → band order → per-model preflight → episodes → smoke + full runs → report |
| `datasets/flair_rgb.py` | RGB-only loader. `band_order` (`rgb`/`bgr`) is a **required** argument, no default. |
| `scripts/check_bands.py`, `evaluation/bands.py` | Band-order decision: soil/water colour statistics + rgb/bgr montage; None when unsure. |
| `evaluation/episodes.py`, `scripts/make_episodes.py` | One fixed episode list; data-only eligibility; `auto` domain mode; refuses stale files. |
| `models/persam.py` | Shared prototype-matching front end for SAM 1 and SAM 2 (adapters per model). |
| `models/sam3_fss.py` | SAM 3 via Hugging Face `transformers` + the FSS-SAM3 canvas (`models/canvas.py`). |
| `scripts/run_model.py` | One model over the episode list; `results.jsonl` + packed masks; resumable; episode-hash guard. |
| `scripts/preflight.py`, `scripts/prefetch_weights.py` | Fail-fast checks in each model's own Python env; SAM 3 weight caching. |
| `evaluation/metrics.py`, `evaluation/report_lib.py`, `scripts/report.py` | Tables, empty-prediction rate, paired bootstrap, baselines, 10 rule-selected figures. |
| `models/mock.py` | Colour-threshold stand-in to test the pipeline. **Not a model**; flagged in the report. |

**Deleted (recoverable from git history, last in commit `f16b797`):** `scripts/evaluate_sam3.py`,
`scripts/visualize_flair_predictions.py`, `scripts/inspect_flair.py`, `models/spectral.py`, `tests/test_spectral.py`
and the old `run.sh`. Kept as legacy dependencies: `datasets/flair.py`, `ecological/`, `tests/test_flair.py`.

### Verified so far (2026-09-30, on the Mac, synthetic scenes only — NOT FLAIR, not performance)
- Real SAM 2 (Hiera-S) and real SAM 3 (`facebook/sam3` via transformers, Python 3.11, MPS) run through
  `run.sh` together, each from its own interpreter, and produce a combined report.
- SAM 3 returned **no query mask at all** on flat full-height colour stripes (empty-prediction rate 100%) but
  segmented a disk scene correctly (IoU 0.99). Expect SAM 3 to struggle on "stuff" classes; the report shows the
  empty-prediction rate so this is visible rather than buried in the IoU.
- SAM 1: adapter written and covered by the fake-model tests, **never run on real weights** (no checkpoint on the Mac).

### Protocol decisions (and why)
- **One episode file for all models** → comparisons are paired; bootstrap CIs resample episodes within class.
- **Eligibility depends only on the data**: class covers 2%–90% of BOTH patches; support ≠ query;
  `--domain_mode auto` (default) = cross-domain when the data has >= 2 FLAIR domains, else same-domain with a loud warning (also recorded in the report). The old rule that dropped supports
  with no 500 px–60% connected component was model-specific and is gone (SAM 3 falls back to the
  whole-class box and flags `box_fallback`).
- **Metrics**: class-accumulated mIoU (standard few-shot) is the headline; mean-of-episodes mIoU and
  FB-IoU are reported next to it. `all_fg` (predict the whole patch) and `empty` baselines are listed
  in every table — a model that does not beat `all_fg` on a class has shown nothing there.
- **Figures**: chosen by a fixed rule *after* all runs (largest disagreement / all succeed / all fail,
  with a minimum-coverage guard so a tiny target cannot look like a model failure). Not hand-picked.
- **Fair-comparison caveat, stated up front:** SAM 1/2 use PerSAM-style prompting; SAM 3 uses the
  canvas + one exemplar box. The headline therefore compares *model + its best-available few-shot
  recipe*, not the bare encoders.

### Open / unverified
- **BAND ORDER.** The IGN Hugging Face dataset card says aerial channels are "Red, Green, Blue, NIR,
  nDSM"; the repo's old loader assumed B, G, R with no source. If the card is right, **every Phase 3
  number was produced from a blue/red-swapped image** (and its NDVI from the wrong band). Treat all
  Phase 3 numbers below as **unverified pending `check_bands.py`**. The toy set came from FLAIR-HUB,
  whose order may differ from FLAIR #1.
- **SAM 3 on the cluster is unverified.** The Hugging Face route needs only Python >= 3.10 and ran on the Mac
  (Python 3.11, MPS), so the cluster's Python 3.11 is likely fine; its CUDA 12.2 driver with a recent torch wheel
  is the remaining question. The ">= 3.12 / CUDA >= 12.6" requirement seems to belong to the native `sam3`
  package, which this repo does not use.
- SAM 2 and SAM 3 verified on real weights on synthetic scenes only; **SAM 1 never run on real weights.**
- ~~No FLAIR data on the Mac~~ (superseded 2026-10-01: FLAIR-HUB toy, 10 domains, is on the Mac). The cluster
  copy is believed to be FLAIR #1 layout, one domain (D041_2021, 525 patches).
- Hugging Face access to `facebook/sam3` is approved for the team account; log in with
  `hf auth login` / `HF_TOKEN`. Never commit a token.

---

*Everything BELOW this line is the earlier (Phase 3, ecological-priors) handoff, written 2026-09-30 when the team decided to rethink the direction. It is historical: file names and commands in it refer to code that has since been removed, and its FLAIR numbers are unverified because of the band-order question above.*

---

## 0. If you read one section, read this

**The project:** few-shot semantic segmentation of habitat / land-cover classes from
multispectral aerial imagery, using ecological indices (NDVI, local Shannon entropy) as
priors. Student Research Project 2025/26, Group 7, Universität Hildesheim (ISMLL).
Supervisor **Ahmad Bdeir**; examiner **Prof. Dr. Niels Landwehr**.

**Best result to date:** **mIoU 0.2860**, 1-shot, 12 FLAIR classes, **zero training** —
frozen SAM 1 ViT-B with PerSAM-style prototype-cosine prompting, RGB input.
Up from 0.0560 for the previous architecture.

**Status:** the pipeline works. The *contribution* is the open problem. Three of the four
original novelty claims are closed by published work, and the team has decided to rethink
the direction rather than continue straight ahead.

**Hard constraints that bound any new plan:**
1. The EU UAV target data is **RGB + NIR only, no elevation** — and has *never been obtained*.
2. The cluster (8× RTX A4000 16 GB, driver 535 / CUDA 12.2, Python 3.11) **cannot run SAM 3**
   without an environment upgrade (needs Python ≥ 3.12, CUDA ≥ 12.6).
3. ISPRS Potsdam was flown **leaf-off**, so it cannot test an NDVI hypothesis at all.

---

## 1. Where everything lives

```
~/Workspace/  (local machine)
├── SRP/                      Phase 1 ViRefSAM. Has SRP_MASTER_SUMMARY.md + CLAUDE.md.
├── SRP II.zip/               Phase 2 ViRefSAM. A DIRECTORY, not an archive.
├── srp-sam3-flair/           Phase 3+. THE CURRENT REPO. git, 5 commits, NOT pushed.
├── LoveDA/                   Track A benchmarks + all the SAM 2 work + checkpoints.
└── SRP_documents/            Proposals, slide decks, study notes, CBST checkpoint.

~/Downloads/
├── SRP-3RD EXPERIMET/        Where Phase 3 was actually run. Merged into the repo,
│                             EXCEPT PROJECT_DEFENSE_REPORT.md (52 KB) which is only there.
└── 2nd iteration - SRP.zip/  Duplicate of SRP II.
```

**Not on disk anywhere:** the EU UAV data, ISPRS Potsdam data, SAM 3 weights, any ViRefSAM
`.pth` checkpoint. The only model weights present are `LoveDA/sam2_checkpoints/sam2.1_hiera_small.pt`
and `LoveDA/log/sam2_finetune*/`.

---

## 2. Every number measured, with its source

### Track A — LoveDA (supervised, 7 classes). Never few-shot.

| Experiment | mIoU | Source |
|---|---|---|
| HRNet-W32 supervised | 0.5134 | `SRP_documents/SRP Research.pdf` p2 (OA 0.6923, κ 0.5809) |
| DeepLabv2 + CBST (UDA) | 0.3509 | same (OA 0.5141, κ 0.3781) |
| SAM 2 **zero-shot oracle** | 0.8467 | same p4 — **uses ground truth to label regions; a ceiling, not a model** |
| SAM 2 fine-tune Phase 1 | 0.4862 best (ep 14), 0.4746 (ep 20) | `LoveDA/log/sam2_finetune/` |
| SAM 2 fine-tune Phase 2 | 0.4923 (ep 2, only checkpoint saved) | `LoveDA/log/sam2_finetune_p2/` |

Phase 1 curve: 0.4401 → 0.4703 → 0.4634 → 0.4552 → 0.4466 → 0.4684 → **0.4862** → 0.4771 →
0.4696 → 0.4746. Peaks at epoch 14, drifts down — mild overfitting.

Model: `sam2.1_hiera_small.pt`, 184 MB. Scripts: `LoveDA/sam2_{zeroshot,finetune,finetune_phase2}.py`.

> **Four different HRNet numbers circulate in the documents**: 0.5134 (measured), 0.58
> (speaker script), 0.35 (deck slide 11 — a copy-paste of the UDA figure), 0.4979 (the
> published LoveDA paper). Pick one and state its split.

### Track B — ViRefSAM on ISPRS Potsdam (base: impervious/building/car, novel: low-veg/tree)

| Phase | Best novel-class val mIoU | Notes |
|---|---|---|
| Phase 1 (`SRP/`) | 0.0709 | **Not a real measurement** — see §4.2 |
| Phase 2 (`SRP II.zip/`) | 0.0560 (ep 450 of 2000) | Leak-free, seed-fixed, 200 episodes |

### Phase 3 — FLAIR, frozen SAM 1 + PerSAM prompting, 1-shot, zero training

| Composite | mIoU | FB-IoU | Vegetation classes | Built / bare classes |
|---|---|---|---|---|
| **rgb** | **0.2860** | 0.2935 | 0.3484 | **0.2267** |
| cir | 0.2535 | 0.2525 | 0.3473 | 0.1595 |
| ndvi_dsm_h | 0.2442 | 0.2748 | 0.3373 | 0.1455 |

Per class (rgb / cir / ndvi_dsm_h): vineyard 0.570/0.587/0.541 · coniferous 0.449/0.444/0.366 ·
plowed 0.374/0.246/0.229 · deciduous 0.345/0.328/0.349 · impervious 0.340/0.124/0.156 ·
water 0.304/0.263/0.262 · herbaceous 0.254/0.253/0.261 · agricultural 0.253/0.244/0.255 ·
bare_soil 0.210/0.155/0.158 · pervious 0.106/0.155/0.082 · brushwood 0.125/0.126/**0.169** ·
building 0.104/0.117/0.104.

Raw files: `outputs/phase3/results_*.json` (tracked in git).

**The reading that matters:** on vegetation the three composites are **tied**. RGB's overall
win comes entirely from albedo-defined classes — impervious 0.340 → 0.124, plowed 0.374 →
0.229. So input substitution does not "use NIR", it *trades away* the colour priors the
encoder was pretrained on, and the trade is bad.

**Caveats on these numbers, all significant:**
- n = 30 episodes/class → per-class SE ≈ 0.024. Per-class gaps under ~0.05 are noise.
  The vegetation "tie" is definitively noise; the built-class gap is real.
- **No domain holdout.** `--holdout_domains` and `--fold` both defaulted to `None`, so
  support and query can come from the same FLAIR domains. Inflates everything.
- **Toy subset only** (~750 MB). Class index has 14 classes, several nearly empty
  (class 13: 5 patches, class 15: 6, bare_soil: 24). Rare-class numbers are unreliable.
- **The prompt-selection strategy was never ablated.** All three runs used
  `strategy="ecological"` with no random/largest/compact control — the contribution is
  unmeasured.
- **It was SAM 1, not SAM 3** — see §4.1.

---

## 3. What the code does now

`srp-sam3-flair/`, git, 5 commits, **not pushed** (GitHub auth failed; remote is set to
`https://github.com/priyeshshrestha/srp-sam3.git`, repo exists and is empty).

| File | Role |
|---|---|
| `datasets/flair.py` | FLAIR loader. 6 composites, domain splits, class index. |
| `ecological/indices.py` | NDVI + local Shannon entropy. **Border bug fixed** (§4.5). |
| `models/canvas.py` | FSS-SAM3 canvas geometry, verified against the released code. |
| `models/prompting.py` | Ecological box selection for "stuff" classes + candidate re-scoring. |
| `models/spectral.py` | **Post-encoder** spectral insertion (plausibility, modulation, fusion). |
| `scripts/evaluate_sam3.py` | Multi-backend evaluator: sam3-native / hf / sam2 / sam1 / mock. |
| `scripts/visualize_flair_predictions.py` | 6-panel qualitative figures. |
| `scripts/inspect_flair.py` | Verifies the band-order assumption empirically. |
| `run.sh` | SLURM sweep: encoders × composites × insertions × strategies. |
| `tests/` | 20 tests, all passing, **no GPU / no data / no model required**. |

### Design decisions worth preserving

- **Composites are the experimental knob.** A frozen RGB model has 3 input slots; which 3
  signals occupy them is the question. Elevation-free: `rgb`, `cir`, `ndvi_h_nir`, `rgb_ndvi`.
- **Post-encoder insertion has a guaranteed floor.** `modulate_similarity(w=0)` and
  `fuse_similarity(alpha=1)` reproduce the RGB baseline *exactly*. Input substitution has no
  such setting, which is why it could only trade.
- **Backend must be explicit.** The silent fallback to `mock` was removed; results record
  `backend`, `model_id`, `uses_elevation`, `transfers_to_rgb_nir`.
- **Split by domain, never randomly.** FLAIR domains vary by department *and* acquisition
  date, so holding them out tests geographic and seasonal transfer.

---

## 4. Findings — things that are settled

### 4.1 Phase 3 ran on SAM 1, not SAM 3

`--model_backend auto` had a silent chain: sam3-native → HF sam3 → segment_anything → mock.
SAM 3 failed (gated weights + Python/CUDA too old), so it loaded
`sam_vit_b_01ec64.pth` from the old ViRefSAM work. The method is **PerSAM-style**: pool a
prototype, cosine-match against query features, take the similarity peak/trough as
positive/negative point prompts, feed to SAM 1's decoder. The canvas code was never called.

The Phase 3 report's *prose* is honest about this (§291, §536), but the directory is
`srp-sam3-flair`, the script is `evaluate_sam3.py`, and the milestone line says "Phase 3
Ecological Foundation Model Architecture". **Rename to what it is before anyone reads it.**

### 4.2 The Phase 1 → Phase 2 "regression" is not real

0.0709 was recorded at **episode 20 of 500** — a randomly-initialised MLP with γ ≈ 0. The
validation pool was 4 tiles, and simulating the actual sampler gives a **76.3% support/query
leak** with only 4 distinct possible queries, so "50 episodes" has effective n ≈ 4. Both
figures are a max over checkpoints (upward-biased). Simulated under the null of *identical*
true means, the expected apparent drop is **+0.0855**; observed was +0.0149. The artefact
model over-predicts by 5.7×. **Phase 2 may well be better.**

Also: 52% of Phase 2 validation episodes have GT below 1% of the patch (the sampler admits a
class at ≥50 of 262,144 px). Holding the model fixed and changing only that prevalence moves
mIoU by 4.7×. Both absolute numbers are largely statements about the metric.

### 4.3 The "51.64% IoU breakthrough" was degenerate

Measured directly from `SRP II.zip/outputs/phase2/qualitative_low_vegetation_1.png`: the GT
panel contains **exactly one unique pixel value**. The sampler only draws patches with ≥50 px
of the class, so the mask cannot be all-background — the entire query is low vegetation. The
prediction covers ~46% of the frame, so IoU ≈ coverage. A predict-everything baseline scores
1.00. **Never cite this figure.**

### 4.4 ViRefSAM is an existing paper

**Bi et al., arXiv 2507.02294, 3 July 2025** — "ViRefSAM: Visual Reference-Guided Segment
Anything Model for Remote Sensing Segmentation". Authors: Hanbo Bi, Yulong Xu, Ya Li,
Yongqiang Mao, Boyuan Tong, Chongyang Li, Chunbo Lang, Wenhui Diao, Hongqi Wang, Yingchao
Feng, Xian Sun. Reports **53.06 mIoU 1-shot on iSAID-5i**.

The project's code uses the paper's exact component names — `VisualContextualPromptEncoder`
("Visual Contextual Prompt Encoder") and `EcologicalAdapter`, whose docstring reads "Dynamic
Target Alignment Adapter" (the paper's second named component). This is a reimplementation,
not a coincidence. Evaluated on iSAID-5i, LoveDA-2i, COCO-20i.

Three divergences from the paper, each mapping to an observed symptom: the paper uses **many
Voronoi-tessellated local background prototypes** (project: one global inverse-pooled
vector); **64 learnable queries with mask-constrained attention to the query image**
(project: 4 static tokens with *no* query-image interaction); and the adapter sits **inside**
the image encoder (project: after it).

**Unresolved:** whether this was a deliberate uncited reimplementation or a name collision.
Must be settled before the write-up.

### 4.5 A Shannon entropy bug, inherited from the original code

`compute_shannon_entropy_torch` convolved with **zero padding**, so windows at the patch edge
saw phantom pixels belonging to no bin; probabilities stopped summing to 1 and the entropy
formula read the deficit as diversity. On a *uniform* raster the interior correctly returns
~9e-6 but **every border pixel returns ≈ 0.13** — a ring of fabricated heterogeneity of width
`kernel_size // 2`, scaling with window size (~2% of a 512 px patch at 7 px, ~8% at 21 px).

**Fixed in `srp-sam3-flair`** (replicate padding; `padding_mode="zeros"` retained for
reproducing old numbers). **Still live in `SRP/` and `SRP II.zip/`**, and it has affected
every Shannon map they produced.

### 4.6 Trainable-parameter counts in both ViRefSAM reports are wrong

Actual: **402,113** (Phase 1) and **467,649** (Phase 2) — ≈0.50% of the ~94M model, not the
"~263K, <0.3%" both reports state. `train_virefsam.py` prints the true count at startup, so
the run logs contradict the reports.

### 4.7 Two structural defects in the ViRefSAM prompt pathway

- The 4 prompt tokens are concatenated onto an **empty** sparse embedding, but in SAM v1
  every sparse prompt is `PositionalEncodingRandom(coords) + point_embeddings[label]`. The
  tokens carry **zero positional content**, while `TwoWayTransformer` adds `image_pe` to the
  keys and expects positionally-anchored queries. Large, low-frequency, blobby output is the
  direct predicted consequence — and it was present in Phase 1 too.
- A final `LayerNorm` renormalises every episode's tokens, **erasing the prototype's
  magnitude**. Only direction survives, in a space the frozen decoder never saw.

Switching to PerSAM-style *point* prompts (which SAM's decoder was designed for) took mIoU
from 0.056 to 0.286. The diagnosis was correct and the fix worked.

### 4.8 Two hypotheses tested adversarially and REFUTED

- **"The background prototype acts as a negative prompt causing collapse"** — REFUTED.
  `p_bg` is 256 extra input features to a trainable `Linear`, not a labelled token consumed
  by a pretrained suppression operator. `W_bg = 0` exactly recovers Phase 1 and is where
  weight decay pushes. And the symptom is backwards: documented collapse is to **empty**
  masks (mIoU falls 21.9 while FB-IoU falls only 10.5 — the signature of shrinking, not
  inverting), whereas the project's predictions cover 45–58%.
- **"Feed a CIR composite for free NIR"** — POOR, and subsequently confirmed by measurement.

---

## 5. The literature position — be blunt about this

| Original novelty claim | Status |
|---|---|
| SAM + few-shot segmentation | **Closed.** ViRefSAM (2507.02294), SEMPNet (GIScience & RS, Dec 2024), FSS-SAM3 (2604.05433), plus a 2026 survey of the subfield (MDPI Remote Sensing 18(4):637). |
| SAM + NDVI / ecological channels | **Closed.** VegSAM (CVPRW 2026) injects NDVI priors into SAM's encoder via adapter + LoRA. GMBFormer is NDVI-guided. |
| Multi-country UAV evaluation | **Substantially eroded.** ProC-SAM3 reports on UAVid 60.0, UDD5 73.4, VDD 67.5. |
| **Local Shannon entropy as an INPUT prior** | **Genuinely open.** Every ecological work found computes Shannon/Simpson/Pielou as an *output* metric derived from segmentation results. Using it as an input is an inversion with no precedent found. Narrow but real, and the project under-argues it. |

**Also on record:** ProC-SAM3 reports **64.5 mIoU on ISPRS Potsdam with no training at all**.
Different protocol (open-vocabulary all-class vs base/novel few-shot), but state the
difference plainly rather than letting someone else find it.

### Key papers

- **ViRefSAM** — Bi et al., arXiv 2507.02294 (the name collision)
- **FSS-SAM3** — Tsai, Lin & Wang, arXiv 2604.05433, 7 Apr 2026. Training-free FSS from a
  frozen SAM 3 via spatial concatenation. SOTA on PASCAL-5ⁱ (79.6) and COCO-20ⁱ (66.6).
  Code: `github.com/WongKinYiu/FSS-SAM3`. PDF cached at
  `~/.claude/projects/.../tool-results/webfetch-1790617705255-vcokn3.pdf`
- **SAM 3** — arXiv 2511.16719, released 19 Nov 2025
- **VegSAM** — CVPRW 2026 (closes the SAM + NDVI leg)
- **Sherrah 2016** — arXiv 1606.02585. RGB vs CIR on ISPRS Potsdam: ImageNet pretraining gave
  **+1.91% with RGB vs +1.13% with CIR** — the closest real evidence on false-colour input.

### Facts about SAM 3 worth keeping

- **Cross-image exemplar prompting is NOT supported.** Exemplar tokens ROI-pool from the
  query image's own feature map. Verified three ways including the released code. *This is
  why the canvas trick exists* — it's a necessity, not a shortcut.
- **Text carries most of the performance.** FSS-SAM3 Table 5, COCO 1-shot: text-only **74.9**,
  visual-exemplar-only **66.6**, both **75.8**. The box adds +0.9 over text.
- **Negatives are catastrophic**: 66.6 → 54.8 (one negative, −11.8) → 44.7 (≤5). The authors
  disabled them.
- **It is instance/box based with no mechanism for "stuff" classes.** The support *mask* is
  never shown to the model — only a box around one representative instance. For land cover
  the semantic-mask bbox is often the whole tile. `models/prompting.py` fills this gap.
- **Licence:** SAM License (not Apache). Academic + commercial use permitted, but §1b(ii)
  makes **acknowledgement in any publication mandatory**, and derived weights must ship the
  licence.
- Checkpoint 3.45 GB, ~0.9B params, native resolution 1008×1008. SAM 3.1 is a *video*
  release — no image benchmark table at all; use `sam3.pt` for images.

---

## 6. Hard constraints

### The EU UAV target data
**RGB + NIR, no elevation.** And it has **never been obtained** — six months in, `eu_uav.py`
has never been run against a single file. This is the single biggest project risk and the
only blocker that costs *calendar* time rather than compute. Ask Ahmad, repeatedly.

Worth asking: if the campaigns flew with forward/side overlap, a photogrammetric DSM may be
derivable via Structure-from-Motion even though the delivered product is RGB+NIR
orthomosaics. Not assumed anywhere in the code.

### The cluster
The university GPU server (address kept out of the repo). 8× RTX A4000 16 GB. Driver 535.171.04 → **CUDA 12.2**.
**Python 3.11.16.** 251 GB RAM. **No `module` command** — a plain conda box; SLURM presence
unconfirmed (`run.sh` assumes `sbatch`; check `sinfo`). GPUs 1, 2, 6, 7 were in use by others
— it is shared, so pin `CUDA_VISIBLE_DEVICES`.

**SAM 3 needs Python ≥ 3.12 and CUDA ≥ 12.6.** Python is fixable with a conda env; the CUDA
driver is the real blocker (minor-version compatibility *may* carry a cu128 PyTorch wheel on
a 12.2 driver, untested). SAM 2 and SAM 1 have no such problem.

### Datasets
- **ISPRS Potsdam** — flown **leaf-off**. Tree NDVI μ ≈ 0.207, overlapping bare soil. Cannot
  test an NDVI hypothesis; the ablation is answered by the data, not the method. Also only 2
  novel classes, and a pathological base→novel split (rigid man-made → organic).
- **FLAIR** (IGN) — 0.2 m, **B, G, R, NIR, Elevation** (BGR-first; NIR at index **3**), masks
  1–19 with **no class 0**, 55 domains varying by geography *and season*. Licence: Open
  Licence 2.0 / etalab-2.0, attribution only. Toy subset ~750 MB; FLAIR-HUB full is ~750 GB.
- **LoveDA** — already downloaded, 7 classes, but **RGB only, no NIR**. Cannot carry NDVI.

---

## 7. Open questions for the rethink

1. **What is the contribution?** Shannon-as-input-prior is the only surviving leg. Is it
   enough, or does the project need reframing? One candidate that the evidence supports:
   *"Where should spectral information enter a frozen foundation model?"* — a systematic
   study across insertion points (input substitution / similarity modulation / prompt
   selection / candidate re-scoring / dual-stream) and encoders (SAM 1 / 2 / 3). You already
   have a clean negative result for input substitution to anchor it.
2. **Which encoder?** SAM 1 produced 0.286 and is the *weakest* candidate. SAM 2 Hiera-S is
   multi-scale, already downloaded, Apache-2.0, and runs on the current environment. SAM 3 is
   blocked on an environment upgrade. **SAM 2 is untested on the few-shot task and may simply
   win.**
3. **Which dataset is the deliverable?** FLAIR is here and works. EU UAV is the nominal
   target and does not exist on disk. Potsdam cannot test the hypothesis. This is Ahmad's
   call, not the team's.
4. **How is the ViRefSAM name collision handled** in the write-up?
5. **Is "training-free" the right frame?** It is honest, cheap, and gives enormous ablation
   throughput (cache SAM's candidate sets once, sweep ecological rules offline in seconds).
   But it caps the ceiling at what a frozen model can do.

---

## 8. Immediate actions, independent of the rethink

1. **Email Ahmad about the EU UAV data.** Costs calendar time; nothing else unblocks it.
2. **Read arXiv 2507.02294** and settle the ViRefSAM provenance question.
3. **Push the repo.** GitHub auth is failing — no valid token in the keychain, and none of
   the three SSH keys authenticate. Fix with `gh auth login`, a PAT, or by adding
   `~/.ssh/id_ed25519.pub` to GitHub. 5 commits are waiting.
4. **Run the SAM 2 backend** — it is wired and needs no new environment.
5. **Fix the Shannon border bug in `SRP/` and `SRP II.zip/`** if any of their numbers will be
   reported.
6. **Change the cluster password** — one was shared in a chat transcript. (The commit that contained it was removed from git history before the first push.)

---

## 9. Calibration — things I got wrong in the last session

Recorded so the next session weights my confidence appropriately.

- Hypothesised the background prototype was acting as a negative prompt causing Tsai-style
  collapse. **Wrong on mechanism and on symptom.** The symptom was backwards — documented
  collapse is to empty masks, the project's predictions cover 45–58%.
- Recommended the CIR composite as a cheap way to exploit NIR. **Measured worse than RGB.**
- Recommended `shannon_source="elevation"` as strictly better. **It cannot transfer** —
  the EU UAV payload has no DSM.
- Scoped the new repo as SAM-3-only and did not carry SAM 2 forward as a backend, despite
  having read the LoveDA logs and knowing SAM 2 was the project's best-performing model.
- Treated the 0.0709 → 0.0560 drop as real and worth explaining. **There is no drop** — the
  Phase 1 number was measurement noise.

The pattern: mechanistic stories built on small deltas were wrong; **direct measurement and
adversarial checking were right.** Prefer cheap experiments over plausible narratives.
