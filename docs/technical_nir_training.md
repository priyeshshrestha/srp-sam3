# Technical report: training SAM 3 with near-infrared on FLAIR-HUB

Group 7 · SRP 2025/26 · Experiment 3 in technical detail · status of 19 November 2026 (4,000- and 12,000-step runs complete; NDVI-adapter and data-focus extension implemented, first batch running: section 8)

This report documents exactly how Experiment 3 was built: the data, how the NIR band was added to SAM 3, what is trained, how training and evaluation work, how we made the RGB vs RGB + NIR comparison fair, and all results so far, and, in section 8, the extension added in November (NDVI adapter arms, priority-class data focus, seasonality check). Code paths refer to the project repository. For the plain-language overview, see the separate project-status document.

## At a glance

| Item | Value |
|---|---|
| Question | Does SAM 3 segment FLAIR land cover better when trained with NIR as an extra input band? |
| Model | SAM 3 (`facebook/sam3`, Hugging Face `transformers` 5.18.0), 840.6 M parameters |
| Change to the model | patch embedding widened from 3 to 4 input channels, NIR weights initialised to 0 |
| Trained parameters | 16.5 M (patch embedding, LoRA r = 16 on the ViT, DETR encoder, semantic path of the mask decoder) |
| Output | SAM 3's built-in semantic head, one logit map per class prompt; prediction = argmax over 12 prompts |
| Data | FLAIR-HUB aerial RGBI, 0.2 m, 512 × 512 px; official split: 37,900 train patches (21 domains), 50,700 test patches (16 domains) |
| Control | identical run with RGB only (same seed, images, order, augmentations, prompts, initial weights) |
| Repetitions | 3 seeds per arm |
| Main result (12,000 steps) | RGB 0.6220 ± 0.0030, RGB + NIR 0.6314 ± 0.0003 mIoU; Δ = +0.0095, 95% CI [+0.0082, +0.0108] |
| Shorter runs (4,000 steps) | RGB 0.6062 ± 0.0021, RGB + NIR 0.6157 ± 0.0004; Δ = +0.0095, 95% CI [+0.0084, +0.0107] |
| Extension (18 Nov, section 8) | two more ways to use NIR (NDVI adapter, VegSAM-style) and training focused on water/forest/vegetation classes; **running, no results yet** |

## 1. Data

### 1.1 Source and format

We use **FLAIR-HUB** (IGN), downloaded from Hugging Face (`IGNF/FLAIR-HUB`, public, Etalab Open Licence 2.0). FLAIR-HUB stores one zip per domain and modality. We only use two modalities:

- `<domain>_AERIAL_RGBI`: aerial orthophotos, **4 bands in the order R, G, B, NIR**, uint8, 512 × 512 px at 0.2 m, LZW-compressed GeoTIFF (reading needs `imagecodecs`).
- `<domain>_AERIAL_LABEL-COSIA`: per-pixel land-cover labels, same grid.

We never use the height (DEM) band, Sentinel-1/2 or SPOT, because the EU drone data has none of them. Over all 74 domains, the RGBI zips total 241.2 GB and the label zips 1.8 GB.

**Band order was verified on the files**, not assumed. `scripts/check_bands.py` compares channel 0 and channel 2 on soil-like and water-like classes over 200 patches. On our FLAIR-HUB copy: soil-like classes ch0 − ch2 = +14.0 (red > blue), water-like classes −43.0 (blue > red). Both agree on R, G, B order; the pipeline stops if they disagree.

### 1.2 Labels and classes

FLAIR-HUB labels are numbered 0–18 in a different order from FLAIR #1's 1–19 (for example, HUB 12 = deciduous, but FLAIR #1 class 12 = plowed land). `datasets/flair_rgb.py` (`HUB_TO_FLAIR1`, from the class table of the IGNF/FLAIR-HUB repository) translates every label to FLAIR #1 ids on read, so the rest of the code uses one numbering.

We evaluate the standard **12 classes**: building, pervious surface, impervious surface, bare soil, water, coniferous, deciduous, brushwood, vineyard, herbaceous vegetation, agricultural land, plowed land. The rare classes 13–19 (swimming pool, snow, clear cut, mixed, ligneous, greenhouse, other) are:

- **in training:** treated as negatives for every class prompt (they are genuinely none of the 12);
- **in testing:** ignored (their pixels are excluded from the confusion matrix).

### 1.3 Official split and download

We use the **official FLAIR-HUB split** (`FLAIR-HUB_TRAIN.csv`, `FLAIR-HUB_VALID.csv`, `FLAIR-HUB_TEST.csv`). It has 58 train, 57 validation and 16 test domains, and **no domain occurs in both train and test**. The test regions are therefore never seen in training.

`scripts/download_flair_hub.py` downloads, for a size budget (we used 100 GB):

1. every domain that has test patches (the official test set must be complete): 16 domains, 51.6 GB, 50,700 test patches;
2. then training domains in a fixed (seed 0) random order while the total stays within the budget: 21 domains, 37,900 training patches.

Each zip is downloaded, checked for unsafe paths, unpacked and deleted; finished domains are recorded so the download resumes. `scripts/make_split.py` then builds the split from the CSVs, restricted to the patches on disk:

| Split | Patches | Domains |
|---|---|---|
| Train (used) | 37,900 | 21 |
| Test (used) | 50,700 | 16 |
| On disk but unused (official validation set) | 10,175 | — |

The split file stores a hash of its contents; every run records the hash and the scripts refuse to combine runs made on different splits. We do **no model selection** on any held-out data (fixed schedule, no early stopping), so the validation set is not needed and the test set is touched only once, at the end.

### 1.4 Input pipeline

For each image (`train/data.py`):

- read 4 bands as uint8 (R, G, B, NIR); the RGB-only arm simply drops band 4;
- **augmentation:** random 90° rotation (0–3 times) and random horizontal flip (aerial images have no "up");
- **normalisation:** (x / 255 − 0.5) / 0.5 for every band, including NIR. This is SAM 3's own normalisation; using it for NIR too keeps the zero-initialised NIR filter an exact no-op (see 2.1);
- resized from 512 to 1008 px (SAM 3's native input size) with bilinear interpolation on the GPU.

## 2. Model

### 2.1 Adding NIR: widening the patch embedding

SAM 3's image encoder is a ViT (32 layers, width 1024, 16 heads, 14 × 14 patches; 24 × 24 windowed attention, with global attention in layers 7, 15, 23, 31). The image enters through **one convolution**, the patch embedding:

`vision_encoder.backbone.embeddings.patch_embeddings.projection`: Conv2d(3 to 1024, kernel 14, stride 14, no bias), weight shape (1024, 3, 14, 14).

We checked that this is the **only** layer in the whole model that takes the 3 image channels. Adding NIR therefore only requires replacing it (`train/sam3_rgbi.widen_conv`):

1. create Conv2d(4 to 1024, kernel 14, stride 14), weight shape (1024, 4, 14, 14);
2. copy the pretrained R, G, B filters into channels 0–2;
3. set the NIR filters (channel 3) to **exactly zero**.

Because the NIR filters are zero, the 4-channel model computes exactly what pretrained SAM 3 computes, whatever the NIR values. **Verified on the real weights:** maximum absolute difference of the encoder output = 0.0. The NIR filters still receive gradients (the gradient of a weight depends on its input, here the NIR pixels, not on the weight itself), so training can grow them. We also tested the alternative initialisation (NIR filters = mean of the RGB filters): it shifts the encoder output by up to 236 before any training, destroying the pretrained behaviour, so we did not use it.

### 2.2 What is trained

Training all of SAM 3 is impossible on our 16 GB GPUs: weights, gradients and Adam states alone would need more than 13 GB before any activations. We train **16.5 M of 840.6 M parameters**:

| Part | Parameters | Notes |
|---|---|---|
| Patch embedding (widened) | 0.80 M (RGB arm: 0.60 M) | trained in both arms, learning rate × 0.1 |
| LoRA on the ViT | 4.19 M | rank 16, α = 16, on the q, k, v and output projections of all 32 layers (128 adapters); B initialised to 0, so an exact no-op at the start |
| DETR (fusion) encoder | ≈ 9.5 M | 6 layers, width 256; where the text prompt meets the image |
| Mask decoder, semantic path | ≈ 2.0 M | pixel decoder, prompt cross-attention, semantic projection; the instance-mask parts are frozen |
| Text encoder (353.5 M) | 0 | run once for the 12 class phrases, then removed from memory |
| Everything else (rest of the ViT, DETR decoder, box heads) | 0 | frozen |

The RGB arm trains exactly the same parts; only the patch embedding has one channel fewer.

### 2.3 Using SAM 3 as a semantic segmenter

SAM 3 normally returns instance masks for a concept ("every conifer tree"). FLAIR classes are "stuff" (areas, not objects), and in our few-shot experiments SAM 3's instance output returned nothing in 34% of cases. SAM 3 also has a built-in **semantic head** (`semantic_seg`): one dense logit map per prompt, at 288 × 288. We train and use that head.

- Each class has one fixed text prompt, chosen before any result was seen (`models/sam3_fss.CLASS_PHRASES`): building, lawn, paved road, bare soil, water, conifer tree, deciduous tree, shrub, vineyard, grass, crop field, plowed field.
- The 12 text embeddings are computed once with SAM 3's text encoder (padded to 32 tokens) and cached.
- Our forward pass (`Sam3Semantic.encode` / `decode`) runs the image encoder once, then the DETR encoder and the mask decoder once per prompt. It skips the DETR decoder and box heads, which the semantic head does not use. **Verified:** identical to `Sam3Model(...).semantic_seg` (maximum difference 0.0 for a 3-channel model and 2.5 × 10⁻⁵, floating-point noise, for a 4-channel one; both on CPU).
- **Prediction:** all 12 prompts are run (4 at a time, to bound memory), the maps are upsampled to 512 × 512, and each pixel gets the class with the highest logit.

## 3. Training

### 3.1 Objective

Each training image is shown with **3 class prompts**: up to 2 classes present in it (covering at least 0.5% of the patch) and the rest absent. This teaches both "segment this" and "this is not here". For each prompt the target is the binary mask of that class.

Loss per prompt = **binary cross-entropy** (averaged over valid pixels) + **soft Dice** (smoothing 1), with the logits upsampled from 288 to 512 to match the labels. Valid pixels are those with a label in 1–19.

### 3.2 Settings

| Setting | Value |
|---|---|
| Optimizer | AdamW, weight decay 1e-4 |
| Learning rate | 1e-4 (patch embedding 1e-5) |
| Schedule | linear warm-up over 200 steps, then cosine decay to 0 |
| Steps | 4,000; second set of runs: 12,000 |
| Batch | effective 8 images per step (2 × 4 accumulation, or 1 × 8 on GPUs with less free memory) |
| Precision | bf16 autocast on the GPU, fp32 master weights |
| Gradient clipping | global norm 1.0 |
| Memory saving | gradient checkpointing of all 32 ViT layers and all 6 DETR-encoder layers |
| Checkpoints | every 250 steps or every 10 minutes, whichever comes first |

These settings were chosen before training and not tuned on the test set.

**Gradient checkpointing note.** `transformers` refuses to enable gradient checkpointing on `Sam3VisionModel`, so we wrap each layer's forward in `torch.utils.checkpoint` ourselves. The DETR encoder had to be included as well: it attends over all 72 × 72 = 5,184 tokens for every prompt, and without checkpointing it ran out of memory even on a 20 GB machine.

### 3.3 Making the comparison fair

The RGB and RGB + NIR runs of one seed differ **only in the NIR channel**:

- **Same data and order:** the image order comes from a generator seeded with the run seed; every random choice for training item k (rotation, flip, prompted classes) comes from `np.random.default_rng([seed, k])`. It does not depend on the model, the number of channels, the number of data-loader workers, or where a resumed run restarted.
- **Same initial weights:** widening the convolution consumes random numbers in the RGB + NIR arm only, so the random generator is re-seeded right before the LoRA matrices are initialised.
- **Same trained parts, same settings.** `run_train.sh` refuses to run or report if any setting differs from the existing runs.

A good sign that this works: at step 20, the two seed-0 runs had almost identical losses (1.4472 vs 1.4447).

### 3.4 Running on a shared GPU server

The server has 8 × NVIDIA RTX A4000 (16 GB), shared with other students; every card is usually partly in use. We measured our needs with a short probe:

| | Peak GPU memory |
|---|---|
| Training, 1 image per pass | 5.8 GB |
| Training, 2 images per pass | 9.2 GB |
| Evaluation | 4.75 GB |

`scripts/gpu_queue.py` checks free memory every minute and starts a run only on a GPU with enough free memory plus a 1.5 GB margin, one of our runs per GPU, at most a set number of GPUs at once (currently 2, always the most free ones). It picks 2 or 1 images per pass to fit; the effective batch stays 8, so the experiment does not change. If another job takes the memory and ours runs out, the run exits with code 3, goes back in the queue and resumes from its last checkpoint. For the November batches the cap was raised (`MAX_GPUS`, or live by writing a number into `<output>/train/max_gpus`); the placement rule is unchanged, and other students' jobs are never touched. Observed speed: 7.4–13 s per optimizer step depending on how busy the GPU is, so 4,000 steps took roughly 8–15 hours per run.

## 4. Evaluation

- **Test set:** all 50,700 official test patches from 16 unseen domains.
- **Metric:** a 12 × 12 confusion matrix is accumulated over all test pixels with a label in 1–12; IoU per class = TP / (TP + FP + FN); **mIoU** = mean over the 12 classes.
- **Per-image confusion matrices are saved**, which allows paired statistics and later re-analysis without re-running the models.
- **Seeds:** we report the mean ± standard deviation of mIoU over the 3 seeds.
- **Significance:** paired bootstrap over test images. 1,000 times, we resample the 50,700 images with replacement (the same resample for both arms), sum the confusions (seeds pooled) and compute mIoU(RGB + NIR) − mIoU(RGB). The 2.5th and 97.5th percentiles give the 95% confidence interval. Per-class intervals use the same resamples.
- **Zero-shot reference:** untrained SAM 3 with the same 12 prompts and semantic head, to show what training adds.
- **Added for the new batches (section 8):** `TEST_LIMIT` evaluates all runs on the same fixed random subset of the test set (seed 0); the report adds mIoU over the priority classes next to the 12-class mIoU (both with paired intervals), and per-test-domain tables with the flight date.
- **Figures:** chosen by a fixed rule after all runs, among test patches with at least 3 classes each covering at least 5% of the patch: the 4 patches where NIR helped most, 2 where it hurt most, 2 where both models do well, 2 where both fail (seed 0).

## 5. Results (4,000 steps)

### 5.1 Overall

| Model | mIoU (mean ± sd over seeds) | Per seed |
|---|---|---|
| SAM 3 zero-shot (text prompts, untrained) | 0.1801 | — |
| SAM 3 + LoRA, RGB | 0.6062 ± 0.0021 | 0.6092, 0.6048, 0.6047 |
| **SAM 3 + LoRA, RGB + NIR** | **0.6157 ± 0.0004** | 0.6162, 0.6159, 0.6151 |

**mIoU(RGB + NIR) − mIoU(RGB) = +0.0095, 95% CI [+0.0084, +0.0107]** (bootstrap over 50,700 test images, 3 seeds pooled). Per seed: +0.0069, +0.0112, +0.0104.

Training contributes far more than NIR: +0.43 mIoU from zero-shot to fine-tuned, +0.0095 from adding NIR.

### 5.2 Per class

IoU columns are means over the 3 seeds; Δ and its 95% CI are computed on the confusions pooled over seeds. * = CI excludes 0.

| Class | Zero-shot | RGB | RGB + NIR | Δ | 95% CI of Δ |
|---|---|---|---|---|---|
| Building | 0.620 | 0.832 | 0.828 | −0.004 * | [−0.004, −0.003] |
| Pervious surface | 0.003 | 0.545 | 0.534 | −0.011 * | [−0.013, −0.008] |
| Impervious surface | 0.404 | 0.742 | 0.741 | −0.001 * | [−0.002, −0.000] |
| Bare soil | 0.050 | 0.647 | 0.652 | +0.005 * | [+0.000, +0.010] |
| Water | 0.422 | 0.869 | 0.875 | +0.007 * | [+0.003, +0.010] |
| **Coniferous** | 0.065 | 0.555 | 0.620 | **+0.065 *** | [+0.060, +0.070] |
| Deciduous | 0.103 | 0.689 | 0.707 | +0.018 * | [+0.017, +0.020] |
| Brushwood | 0.060 | 0.305 | 0.309 | +0.003 * | [+0.001, +0.005] |
| Vineyard | 0.069 | 0.718 | 0.715 | −0.003 | [−0.007, +0.001] |
| Herbaceous vegetation | 0.081 | 0.493 | 0.512 | +0.018 * | [+0.017, +0.020] |
| Agricultural land | 0.175 | 0.533 | 0.540 | +0.007 * | [+0.005, +0.009] |
| Plowed land | 0.109 | 0.348 | 0.356 | +0.008 * | [+0.002, +0.015] |

NIR significantly improves **8 of 12 classes**, all of them vegetation, soil or water; the largest gain is coniferous forest. Buildings and roads change by less than 0.005. The only loss above 0.005 is pervious surface (lawns, gardens, sports grounds), plausibly because these look like herbaceous vegetation in NIR; the saved confusion matrices allow us to check this.

![Change in IoU per class when NIR is added, 4,000-step runs, with 95% CIs. Green: significant gain; dark grey: significant loss; light grey: not significant.](docs/fig/train_delta_4k.png)

### 5.3 Examples (rule-chosen, seed 0)

![Largest NIR gain: per-patch mIoU 0.287 (RGB) to 0.896 (RGB + NIR). The RGB model labels the meadow as agricultural land; with NIR it is herbaceous vegetation.](outputs/train/report/figures/01_D015-2020_AERIAL_RGBI_AF-S1-17_8-5.png)

![Largest NIR loss: 0.987 to 0.491. With NIR, a plowed field is merged into the neighbouring agricultural land.](outputs/train/report/figures/05_D085-2019_AERIAL_RGBI_AA-S1-28_8-10.png)

Single patches swing strongly when one large region changes class; they illustrate, the aggregate is the evidence.

## 6. Does the model use NIR? Diagnostic

`scripts/nir_diagnostic.py` analyses the finished 4,000-step runs.

### 6.1 NIR ablation

Each RGB + NIR model is evaluated on a fixed subset of 1,000 test patches in three ways: with the real NIR; with **NIR flattened** (every pixel set to the patch's mean NIR, so no spatial NIR information is left); and with **NIR swapped** (the NIR band of a different test patch: realistic values, wrong content). The RGB-only model of the same seed is the reference.

| Seed | RGB-only model | RGB + NIR, real NIR | NIR flattened | NIR swapped |
|---|---|---|---|---|
| 0 | 0.6281 | 0.6296 | 0.5875 | 0.5689 |
| 1 | 0.6254 | 0.6311 | 0.5874 | 0.5738 |
| 2 | 0.6204 | 0.6299 | 0.5893 | 0.5699 |

Per class, seeds pooled (IoU with real NIR, and the change under each ablation):

| Class | Real NIR | Flattened Δ | Swapped Δ |
|---|---|---|---|
| Building | 0.810 | +0.003 | −0.004 |
| Pervious surface | 0.617 | −0.005 | −0.043 |
| Impervious surface | 0.762 | −0.013 | −0.042 |
| Bare soil | 0.759 | −0.013 | −0.051 |
| Water | 0.867 | +0.006 | −0.043 |
| **Coniferous** | 0.648 | **−0.210** | **−0.207** |
| **Deciduous** | 0.706 | **−0.228** | **−0.199** |
| Brushwood | 0.296 | −0.011 | −0.031 |
| Vineyard | 0.660 | −0.022 | −0.012 |
| Herbaceous vegetation | 0.518 | −0.010 | −0.007 |
| Agricultural land | 0.567 | −0.002 | −0.022 |
| Plowed land | 0.352 | −0.000 | −0.050 |

Interpretation:

- Removing NIR information costs 4–6 points of mIoU, much more than NIR adds over the RGB model (about 1 point). The model has **moved part of its decision onto NIR** rather than using it as a small extra.
- The reliance is concentrated on **trees**: coniferous −0.21 and deciduous −0.23 IoU when NIR is flattened, while every other class changes by less than 0.03. Swapped NIR additionally costs 0.03–0.05 on several non-tree classes.
- Caveat: flattened and swapped NIR are inputs never seen in training, which exaggerates the drops. The concentration on the two tree classes, identical across seeds, shows the reliance is real.

### 6.2 Training loss

Mean training loss per eighth of training (4,000 steps):

| Run | First 1/8 | Middle 1/8 | Second-last 1/8 | Last 1/8 |
|---|---|---|---|---|
| rgb_s0 | 0.783 | 0.486 | 0.422 | 0.441 |
| rgb_s1 | 0.761 | 0.471 | 0.431 | 0.424 |
| rgb_s2 | 0.787 | 0.477 | 0.427 | 0.426 |
| rgbi_s0 | 0.779 | 0.478 | 0.418 | 0.429 |
| rgbi_s1 | 0.781 | 0.462 | 0.420 | 0.409 |
| rgbi_s2 | 0.779 | 0.476 | 0.426 | 0.417 |

The loss falls most in the first half and is still decreasing between the middle and the end; the flat ending coincides with the learning rate decaying to 0. With 4,000 steps × 8 images = 32,000 images, training saw less than one pass over the 37,900 training patches. This motivated the longer runs.

The third part of the diagnostic (the size of the learned NIR filters relative to the RGB filters) is computed but not yet included in this report.

## 7. Results with longer training (12,000 steps)

Same recipe, only `STEPS=12000` (≈ 2.5 passes over the training data instead of < 1), 3 seeds × 2 arms, in a separate output folder; the 4,000-step results above are unchanged. Untrained SAM 3 was not re-run (it does not depend on the number of steps).

### 7.1 Overall

| Model | 4,000 steps | 12,000 steps | Per seed (12,000) |
|---|---|---|---|
| SAM 3 + LoRA, RGB | 0.6062 ± 0.0021 | 0.6220 ± 0.0030 | 0.6244, 0.6237, 0.6178 |
| **SAM 3 + LoRA, RGB + NIR** | 0.6157 ± 0.0004 | **0.6314 ± 0.0003** | 0.6310, 0.6315, 0.6318 |
| **Δ (RGB + NIR − RGB)** | +0.0095 [+0.0084, +0.0107] | **+0.0095 [+0.0082, +0.0108]** | +0.0066, +0.0078, +0.0140 |

- Three times longer training improves **both arms by the same amount** (+0.016 mIoU each).
- **The NIR gain is unchanged** (+0.0095 both times), and its confidence interval is almost identical. The NIR effect is therefore not an artefact of under-training.
- In both experiments the RGB + NIR arm varies much less across seeds (sd 0.0003–0.0004) than the RGB arm (0.0021–0.0030).
- Final training loss: 0.35–0.40 at 12,000 steps vs 0.41–0.44 at 4,000 steps.

### 7.2 Per class (12,000 steps)

| Class | RGB | RGB + NIR | Δ | 95% CI of Δ | Δ at 4,000 steps |
|---|---|---|---|---|---|
| Building | 0.841 | 0.838 | −0.003 * | [−0.003, −0.002] | −0.004 * |
| Pervious surface | 0.573 | 0.574 | +0.001 | [−0.001, +0.003] | −0.011 * |
| Impervious surface | 0.754 | 0.756 | +0.002 * | [+0.001, +0.003] | −0.001 * |
| Bare soil | 0.655 | 0.665 | +0.009 * | [+0.004, +0.015] | +0.005 * |
| Water | 0.887 | 0.888 | +0.001 | [−0.002, +0.005] | +0.007 * |
| **Coniferous** | 0.581 | 0.638 | **+0.057 *** | [+0.052, +0.062] | +0.065 * |
| Deciduous | 0.699 | 0.721 | +0.021 * | [+0.019, +0.023] | +0.018 * |
| Brushwood | 0.324 | 0.328 | +0.004 * | [+0.002, +0.006] | +0.003 * |
| Vineyard | 0.748 | 0.748 | +0.000 | [−0.004, +0.004] | −0.003 |
| Herbaceous vegetation | 0.514 | 0.531 | +0.017 * | [+0.015, +0.019] | +0.018 * |
| Agricultural land | 0.551 | 0.559 | +0.008 * | [+0.005, +0.011] | +0.007 * |
| Plowed land | 0.336 | 0.332 | −0.004 | [−0.012, +0.003] | +0.008 * |

- The core pattern is stable: **coniferous, deciduous and herbaceous vegetation** gain clearly at both step counts.
- With longer training, **7 of 12** classes gain significantly; the **pervious-surface loss (−0.011) disappeared** (now +0.001, not significant). The only significant loss left is building (−0.003).
- Smaller effects shift between the two runs (water, plowed land), so single small per-class differences should not be over-interpreted.

![Change in IoU per class when NIR is added, 12,000-step runs, with 95% CIs.](docs/fig/train_delta_12k.png)

### 7.3 Examples (12,000 steps, rule-chosen, seed 0)

![Largest NIR gain (0.364 to 0.903): the RGB model labels the meadow as agricultural land and the crop field as plowed land; with NIR both are correct.](outputs/long_report/figures/01_D076-2019_AERIAL_RGBI_UA-S1-31_2-1.png)

![Largest NIR loss (0.931 to 0.323): with NIR, a plowed field is merged into the neighbouring agricultural land, the same failure as at 4,000 steps.](outputs/long_report/figures/05_D071-2020_AERIAL_RGBI_NA-S1-24_5-3.png)

The NIR diagnostic (section 6) was run on the 4,000-step models; it can be repeated on the 12,000-step models with `OUT=./outputs/long bash run_train.sh diagnose`.

## 8. Extension: NDVI adapter arms and priority-class data focus (added 18 November 2026, in progress)

**Status: implemented and tested; the first batch is running. There are no results from these arms in this report yet.** This section explains what was added and why, how every part works, what was verified, and how the coming numbers must be read.

### 8.1 In plain words

Sections 1–7 answered one question: if SAM 3 receives NIR as a fourth input band, does it segment better? Yes, by about one mIoU point, almost entirely on trees and other vegetation. Two follow-up questions came up:

1. **Is feeding NIR as an image channel the best way to use it?** A recent paper (VegSAM, CVPR Workshops 2026) does it differently for tree segmentation: SAM sees only the RGB image as usual, and a separate small network turns a *vegetation index* computed from NIR and red (NDVI) into hints that are added inside SAM's image encoder. We implemented this idea and can now compare both routes in one experiment.
2. **Does our training focus on what we care about?** Our project is about biodiversity, so water, forest and vegetation matter more than buildings and roads. We can make training spend more of its effort on those classes without touching the test set.

A third point is a caveat we want to measure and not just worry about: **seasonality and acquisition conditions.** NDVI depends on how green the vegetation is on the day of the flight, and FLAIR domains were flown in different campaigns. If the NIR-based models depend on absolute greenness, they could transfer worse to a new region or season. Section 8.7 checks what the metadata says about this, and the report now breaks results down per test domain.

### 8.2 What NDVI is

NDVI (normalised difference vegetation index) is computed per pixel from the red and NIR bands:

`NDVI = (NIR − Red) / (NIR + Red + 0.0001)`, a number between −1 and +1.

Healthy green vegetation reflects a lot of NIR and absorbs red, so its NDVI is high. Water and many artificial surfaces give values near or below 0; bare soil lies in between. Because it is a ratio, a uniform change of brightness (a darker image of the same scene) mostly cancels out. It does not cancel changes in the vegetation itself (leaf-on or leaf-off, crop growth stage, drought) or differences in how the camera data were processed.

Two caveats specific to our data: (1) the FLAIR images are **8-bit digital numbers with a per-campaign contrast stretch, not calibrated reflectance**, so our "NDVI" is a *pseudo-NDVI*: useful as a relative signal inside an image, not comparable as an absolute value across campaigns. (2) The code computes it from the same normalised and resized tensor that SAM receives (`ndvi_from_normalised` in `train/sam3_rgbi.py`: map back from [−1, 1] to [0, 1], then apply the formula).

### 8.3 The NDVI adapter, step by step

An *adapter* is a small trainable module attached to a large frozen network. Ours (`NdviAdapter`, `train/sam3_rgbi.py`) works like this:

1. **Turn NDVI into tokens.** A convolution with one input channel (Conv2d 1 to 1024, kernel 14, stride 14) cuts the NDVI image into 14 × 14 pixel patches, exactly like SAM's own patch embedding does for the colour image. For a 1008 × 1008 input this gives a 72 × 72 grid of 1024-dimensional NDVI tokens, in the same layout as the ViT's hidden states.
2. **For each of the 32 ViT blocks i**, a small two-layer network turns those tokens into a correction: `P_i = Linear_shared( GELU( Linear_i(tokens) ) )`. `Linear_i` (1024 to 64) is **specific to block i**; `Linear_shared` (64 to 1024) is **shared by all blocks**. The 64-dimensional middle is a bottleneck that keeps the adapter small.
3. **Add the correction to the output of block i:** `hidden_i ← block_i(hidden_{i-1}) + P_i`. The ViT therefore receives vegetation information at every depth and not only at the input.

| Part | Parameters |
|---|---|
| NDVI patch embedding (1 to 1024, 14 × 14, with bias) | 201,728 |
| 32 × per-block layer (1024 to 64, with bias) | 2,099,200 |
| shared layer (64 to 1024, with bias) | 66,560 |
| **Total** | **2,367,488 (2.37 M)** |

**Starting point = exactly pretrained SAM 3.** The shared output layer is initialised to **zero**, so `P_i = 0` and the adapter changes nothing at step 0. This is the same idea as the zero-initialised NIR filters in section 2.1 and as LoRA's zero-initialised B matrix. Training then moves the shared layer away from zero first (it receives gradient because the activations before it are not zero); only after that do the earlier layers receive gradient too. This is why the adapter is given a higher learning rate.

Settings, all fixed before any result: learning rate **5 × 10⁻⁴** for the adapter (5 × the base 10⁻⁴), 10⁻⁵ for the patch embedding (0.1 ×), 10⁻⁴ for everything else. The adapter is applied outside the gradient-checkpointed part of each layer. It is created *after* the LoRA matrices with its own seed (`seed + 1`), so the LoRA initialisation is identical in every arm and the arms stay comparable.

**What we took from VegSAM and what we did not.** Taken: the idea of injecting NDVI through an adapter into every block of SAM's image encoder, starting from zero. **Not** taken: its resizing modulator, its 512 px input, its "tree token" and hierarchical decoder, its LoRA settings (rank 4 on Q and V), and its one-class tree task. We train SAM 3's 12-class semantic head as before. We therefore call the arm **"VegSAM-style"**, not a reproduction of VegSAM.

### 8.4 The four arms (a 2 × 2 design)

Two yes/no questions: does NIR enter as an **image channel** of the ViT, and does NIR enter through the **NDVI adapter**?

| Arm | ViT sees | NIR enters via | Parameters on the NIR path | Question it helps answer |
|---|---|---|---|---|
| `rgb` | R, G, B | nowhere | 0 | the control |
| `rgbi` | R, G, B, NIR | channel only (sections 2–7) | 0.2 M (NIR filters) | does NIR as a channel help? |
| `rgb_ndvi` | **R, G, B only** | **adapter only** (VegSAM-style) | 2.37 M | does NIR help when it never touches the pretrained input layer? |
| `rgbi_ndvi` | R, G, B, NIR | channel **and** adapter | 2.57 M | does the adapter add anything on top of the channel? |

The code can run all four. The current batches use `rgb`, `rgbi` and `rgb_ndvi` (cost: `rgbi_ndvi` is a fourth run per seed). Paired comparisons the report builds (same seeds, same test images, same bootstrap as before):

- `rgbi − rgb`: the original question (unchanged);
- `rgb_ndvi − rgb`: NIR through the adapter alone, against no NIR;
- `rgb_ndvi − rgbi`: which route works better, channel or adapter.

`rgb_ndvi` is the arm that matches VegSAM's design most closely: **the patch embedding stays 3-channel (R, G, B) and NIR is read only by the adapter.** This is enforced in code (`PATCH_CHANNELS` is 3 for `rgb_ndvi`, `CHANNELS` is 4) and tested: a unit test checks that changing only the NIR band of the input changes the output of `rgb_ndvi` only through the adapter.

### 8.5 What was verified

- **Exact no-op at step 0, on the real SAM 3 weights (CPU).** The maximum absolute difference between the output of `rgb(RGB)` and `rgb_ndvi(RGB + NIR)` before training is **0.0**. The same holds for `rgbi_ndvi` against `rgbi`.
- **The adapter really is the only NIR path in `rgb_ndvi`.** The patch embedding has 3 input channels. When the adapter output layer is set to non-zero values, flattening the NIR band changes the output (maximum difference 0.0479); with the layer at zero it changes nothing.
- **Parameter count:** the NIR path of `rgb_ndvi` has 2.37 M trainable parameters (table in 8.3). Gradients reach all parts of the adapter with gradient checkpointing on.
- **Fair comparison preserved:** same seed gives the same images, augmentations, prompts and LoRA initialisation in every arm (tests in `tests/test_train.py`).
- **Test suite:** 17 training tests, 5 queue tests, the benchmark and loader tests, and an end-to-end run of `run_train.sh` on synthetic images with all four arms, a focus file, the diagnostic, resume and the refusal to mix settings. All pass.

### 8.6 Data focus on the classes we care about

**The idea.** Instead of showing every training patch equally often, draw patches with probability proportional to a weight that is higher for patches rich in the *priority classes*:

`weight(patch) = floor + (share of the patch's pixels that belong to a priority class)`, with floor = 0.1.

A patch consisting only of priority classes is therefore drawn (1 + 0.1) / 0.1 = **11 times** as often as a patch with none; the floor keeps every patch possible, so nothing is excluded outright. Patches are drawn with replacement. In addition, when choosing which classes to prompt for a patch, **present priority classes are chosen before other present classes** (the loss section 3.1 is unchanged otherwise: up to 2 present classes and the rest absent, 3 prompts per image).

**Priority classes (currently):** water (5), coniferous (6), deciduous (7), brushwood (8), herbaceous vegetation (10). This choice is a *proposal*, made before any result of the new batches, because they are the vegetation and water classes closest to biodiversity and because coniferous and deciduous showed the largest NIR effects in sections 5–7. **The group has not confirmed it;** it can be changed with the `PRIORITY` setting.

**Measured on the real training pool (37,900 patches, from `scripts/make_focus.py`):**

| | Uniform sampling | Focused sampling |
|---|---|---|
| Share of a training patch's pixels in priority classes (average) | 55.1 % | **77.1 %** |
| Patches with ≥ 50 % priority pixels | 54.8 % of the pool | 81.0 % of the draws |

**What is deliberately unchanged:**

- **The test set stays the complete, official one** (a fixed random subset of it for the fast batch, see 8.8). The focus only affects *training*.
- **All 12 classes stay in the task.** We do not merge the other classes into one "other" class: the 12 text prompts, the 12-class mIoU, the zero-shot reference and the earlier results keep the same meaning. Merging would change the task and make the new numbers incomparable.
- **The focus is part of each run's recorded settings** (a hash of the priority list, the floor and the split). `run_train.sh` refuses to continue or report if runs with different focus settings would be mixed.

**Consequences to remember:** (1) every arm in a batch uses the *same* focus, so the RGB vs NIR comparison inside a batch stays fair; (2) absolute numbers differ from sections 5 and 7 and **must not be compared with them**; (3) because non-priority classes see fewer examples, they may score slightly lower; this is why the report prints **both mIoU over all 12 classes and mIoU over the priority classes only**, and the paired differences are given for both.

### 8.7 Seasonality: what the dataset says

FLAIR-HUB ships per-patch **acquisition dates** (`<domain>_AERIAL_MTD_DATES.gpkg`: patch id, date, time, source image, camera). For the whole dataset they sit in one archive on the hub (`GLOBAL_ALL_MTD.zip`, 2.1 GB, public); we need one file from it. `scripts/season_check.py` reads that file for the patches of our split, tabulates the dates per domain, and computes pseudo-NDVI per class and domain from up to 20 patches per domain (a class counts in a patch only with at least 2,000 pixels). The numbers below come from running it on our real split: **21 training and 16 test domains**. They are descriptive statistics, not a test.

**When the images were taken**

- Every domain was flown between **day 124 and day 287 of the year (early May to mid October)**. There are no winter or leaf-off images in our data: "season" here means early- versus late-growing-season conditions, plus differences between campaigns. Some domains were flown over several months (for example D054057-2018: 5 May to 30 Sep; D040-2021: 8 Jun to 30 Sep), so the median date is only a rough label.
- **Training domains** (2017–2021): day 124–264, median day 225 (13 Aug). **Test domains** (2019–2022): day 125–287, median day 208 (27 Jul). On average the test flights are about 2.5 weeks earlier in the season than the training flights, with a wide overlap.
- **Three test domains were flown later than any training domain** (latest training flight: day 264, 21 Sep): D012-2019 (day 272), D022-2021 (day 282) and D064-2021 (day 287, mid October). No test domain is earlier than the earliest training domain. Late-autumn conditions are therefore never seen in training for these three, and the test set also contains one year (2022) that no training domain has. These are the first places to look in the per-domain results (below).

**How NDVI behaves across domains** (mean pseudo-NDVI of a class within a domain; table at the end of this section)

| Class | Domains | Mean over domains, train / test | Spread of domain means (sd) | Lowest to highest domain |
|---|---|---|---|---|
| Deciduous | 37 | 0.259 / 0.265 | 0.070 | 0.132 to 0.417 |
| Herbaceous vegetation | 37 | 0.045 / 0.048 | 0.067 | −0.140 to 0.198 |
| Coniferous | 31 | 0.172 / 0.197 | 0.131 | −0.162 to 0.510 |
| Agricultural land | 34 | 0.045 / 0.011 | 0.128 | −0.311 to 0.239 |

- **The level of a class shifts a lot between domains**, so a fixed NDVI value does not mean the same thing everywhere. Train and test domains have very similar *average* levels for the tree and grass classes; the shift is between individual domains, in both sets.
- **The ordering between classes is stable.** Deciduous trees have a higher NDVI than herbaceous vegetation in **all 37 domains**, by 0.11 to 0.35 (median 0.21), about three times the domain-to-domain spread of the deciduous level (0.07). What the adapter can rely on is therefore the *contrast* between classes inside an image, more than an absolute level.
- **Coniferous and agricultural spreads are inflated by small samples.** Several domains contribute only 1–3 patches with enough conifer pixels (for instance D085-2019: 0.510 from one patch), and "agricultural land" mixes growing crops with harvested and bare fields. The deciduous and herbaceous rows are the reliable ones.

**Does the date explain it?** Pearson correlation between a domain's median day of year and its class mean NDVI:

| Class | Domains | r | two-sided p |
|---|---|---|---|
| Coniferous | 31 | +0.27 | 0.14 |
| Deciduous | 37 | +0.20 | 0.24 |
| Herbaceous vegetation | 37 | +0.12 | 0.48 |
| Agricultural land | 34 | +0.10 | 0.57 |

The relationship is weakly positive (later flights, higher NDVI) but **none is distinguishable from zero**, and even the largest (r = 0.27) means the date accounts for at most about 7 % of the variation between domains. (p-values: t-test on the Pearson r, treating domains as independent, computed by us from the table.) The earlier check on the 10-domain toy subset gave slightly negative or near-zero correlations; the sign change between the two is what noise looks like. The honest summary: **the flight date alone does not explain the differences between domains.** The rest (year, camera and radiometric processing, region, local weather and land management) is mixed together and cannot be separated with this data.

**What this means for the experiment**

1. Seasonality is not the dominant source of NDVI variation here, so it is not an obvious reason for the NDVI adapter to fail.
2. The stable class ordering is what the adapter needs in order to help.
3. The residual risk is concentrated in the three late-flown test domains (D012-2019, D022-2021, D064-2021) and the 2022 domain (D073-2022). If NIR-based models lose there, relative to `rgb`, that would be the signature of a season or campaign dependence.

**Per-domain results in the report.** `train_report.py` adds a *Per test domain* table for every comparison: mIoU of both arms, their paired difference with a 95 % bootstrap CI (domains with fewer than 10 test images get no CI), the priority-class difference, and, with the date file from `season_check.py`, the median flight date and the correlation of the difference with the day of year (only with at least 5 dated domains). With the 2,000-image test subset each test domain contributes roughly 50 to 210 images, in proportion to its size (1,250 to 5,300 test patches). This is descriptive: 16 domains are 16 comparisons and no correction for that is applied.

**Per-domain NDVI table** (mean pseudo-NDVI; "–" = fewer than the minimum patches with that class; day = median day of year of the domain's flights)

| Domain | Split | Day | Decid. | Herb. | Conif. | Agri. |
|---|---|---|---|---|---|---|
| D083-2020 | test | 125 | 0.151 | 0.014 | 0.059 | −0.090 |
| D036-2020 | test | 138 | 0.195 | 0.012 | 0.047 | 0.070 |
| D061-2020 | test | 139 | 0.191 | 0.003 | 0.064 | −0.027 |
| D084-2021 | test | 147 | 0.278 | 0.076 | 0.161 | 0.041 |
| D069-2020 | test | 175 | 0.222 | 0.056 | 0.084 | −0.009 |
| D026-2020 | test | 177 | 0.243 | 0.011 | 0.158 | −0.120 |
| D015-2020 | test | 182 | 0.283 | 0.068 | 0.093 | −0.034 |
| D076-2019 | test | 204 | 0.388 | 0.075 | 0.439 | 0.007 |
| D073-2022 | test | 212 | 0.381 | 0.040 | – | – |
| D071-2020 | test | 221 | 0.293 | −0.023 | 0.378 | −0.119 |
| D085-2019 | test | 245 | 0.417 | 0.138 | 0.510 | 0.154 |
| D075-2021 | test | 249 | 0.230 | 0.037 | – | 0.097 |
| D068-2021 | test | 251 | 0.199 | 0.040 | – | 0.118 |
| D012-2019 | test | 272 | 0.292 | 0.084 | 0.221 | −0.016 |
| D022-2021 | test | 282 | 0.268 | 0.055 | 0.155 | 0.100 |
| D064-2021 | test | 287 | 0.207 | 0.078 | – | −0.011 |
| D033-2018 | train | 124 | 0.290 | 0.172 | 0.264 | 0.127 |
| D054057-2018 | train | 128 | 0.233 | 0.126 | 0.198 | 0.226 |
| D018-2020 | train | 147 | 0.143 | −0.140 | 0.033 | −0.280 |
| D058-2020 | train | 148 | 0.282 | 0.057 | 0.168 | 0.096 |
| D004-2021 | train | 190 | 0.310 | 0.048 | 0.159 | 0.135 |
| D072-2019 | train | 198 | 0.299 | −0.047 | 0.146 | 0.039 |
| D086-2020 | train | 201 | 0.276 | 0.039 | 0.224 | −0.061 |
| D045-2020 | train | 218 | 0.283 | 0.010 | – | 0.092 |
| D011-2021 | train | 224 | 0.132 | −0.064 | −0.162 | – |
| D017-2021 | train | 225 | 0.349 | 0.179 | 0.387 | 0.125 |
| D038-2021 | train | 225 | 0.210 | 0.037 | 0.097 | 0.127 |
| D066-2021 | train | 225 | 0.326 | 0.041 | 0.184 | 0.157 |
| D037-2021 | train | 226 | 0.183 | 0.055 | – | −0.073 |
| D056-2019 | train | 234 | 0.194 | −0.067 | 0.239 | −0.311 |
| D005-2018 | train | 238 | 0.301 | 0.066 | 0.095 | 0.213 |
| D040-2021 | train | 241 | 0.301 | 0.012 | 0.138 | −0.099 |
| D060-2021 | train | 245 | 0.240 | 0.055 | 0.257 | 0.239 |
| D009-2019 | train | 246 | 0.401 | 0.198 | 0.360 | 0.155 |
| D051-2019 | train | 247 | 0.166 | 0.014 | 0.181 | −0.028 |
| D091-2021 | train | 249 | 0.277 | 0.028 | 0.178 | −0.023 |
| D080-2017 | train | 264 | 0.251 | 0.126 | 0.124 | – |

### 8.8 The experiments and how to read the results

| | Fast batch | Long batch |
|---|---|---|
| Purpose | first look; results for the supervisor meeting | the actual measurement |
| Arms | `rgb`, `rgbi`, `rgb_ndvi` | same |
| Seeds | 1 (seed 0) | 3 |
| Steps | 2,000 (16,000 images, about 0.4 passes over the 37,900 training patches) | 12,000 (96,000 images, about 2.5 passes) |
| Training data | focused on the five priority classes | same |
| Test images | fixed random subset of 2,000 (identical for all runs) | same subset |
| Zero-shot reference | not re-run | not re-run |
| Split | official FLAIR-HUB split, hash `7a0f939a07a36b1e` (the same as sections 5–7) | same |
| Rough duration | 6–7 h if three GPUs are free | about a day per run; 9 runs, so the total depends on how many GPUs are free |

**What can be compared with what.**

- Inside one batch: yes, all arms are paired.
- Fast against long: no; different steps.
- Either batch against sections 5 and 7: **no** (different steps, a focused training set, and a 2,000-image subset instead of 50,700 images). Only the *split* is the same.

**How uncertain a number is.** The bootstrap interval in the report describes only the sampling of *test images*. It does not include the randomness of training itself. From the earlier runs we know that this seed-to-seed spread is 0.002–0.003 mIoU for the RGB arm and 0.0003–0.0004 for the RGB + NIR arm. With a single seed (fast batch) **a difference below about 0.005 mIoU should not be called real, whatever its bootstrap interval says.** The long batch with 3 seeds shows the spread directly.

**How to read the three comparisons.**

| Outcome | Reading |
|---|---|
| `rgb_ndvi − rgb` clearly > 0 | NIR helps even when it only enters through the adapter, without changing SAM's input layer |
| `rgb_ndvi − rgbi` > 0 | the adapter route beats the channel route; < 0: the channel route is better; ≈ 0: no preference |
| `rgb_ndvi − rgb` ≈ 0 while `rgbi − rgb` > 0 | the adapter did not learn to use NIR (see the untuned learning rate in section 9) or the channel route is simply better |
| Per-domain differences vary a lot | the effect depends on region or campaign; look at the dates before drawing conclusions |
| Priority-class difference larger than 12-class difference | the gain sits where we focused, as hoped |

### 8.9 Tooling changes

- `scripts/make_focus.py`: counts class pixels per training patch (cached), writes `focus.json` (weights and hash).
- `scripts/season_check.py` and the helpers `hub_patch_id` / `read_aerial_dates` in `datasets/flair_rgb.py`: acquisition dates and pseudo-NDVI per class and domain.
- `train_report.py`: arms `rgb_ndvi` and `rgbi_ndvi`, priority-class mIoU, the three paired comparisons, per-domain tables.
- `run_train.sh`: `ARMS`, `PRIORITY` and `FOCUS_FLOOR` settings; memory probe on the largest arm; the focus hash in the settings guard; the date file is passed to the report automatically if it exists.
- `nir_diagnostic.py`: also analyses the NDVI arms (the NIR ablation, flattened and swapped NIR, runs for every arm whose input has 4 bands).

## 9. Limitations and open points

- **Hyperparameters were fixed a priori**, not tuned. Results could change with a different learning rate, LoRA rank or number of prompts per image, for both arms.
- **The trained model is not few-shot**: it uses 37,900 labelled patches. It answers whether NIR helps SAM 3 once trained, and is the starting point for adapting to the drone data with few labels.
- **Reliance on NIR is a transfer risk.** A different camera (the drones) records NIR with different brightness and calibration. Planned test: train with random gain and offset on the NIR band only, then evaluate with NIR scaled by about ±20%.
- **Class prompts are fixed phrases** chosen before any result; different phrases might help some classes, but tuning them on the test set would be invalid.
- **Prediction is an uncalibrated argmax** over 12 independently trained sigmoid maps.
- **No comparison with published FLAIR-HUB baselines yet**; those numbers must be checked in the FLAIR-HUB paper before being cited.
- **NDVI adapter (section 8):** its learning rate (5 ×) and bottleneck (64) were fixed a priori and not tuned; if the `rgb_ndvi` arm does not beat `rgb`, an untuned adapter is a plausible cause and not proof that the idea fails. The arm is "VegSAM-style" (only the adapter was taken over), not a reproduction of VegSAM.
- **Pseudo-NDVI** is computed from 8-bit digital numbers with a per-campaign stretch, not from reflectance; absolute values are not comparable across campaigns.
- **Priority classes are a proposal** that the group has not confirmed, and the focus changes the absolute numbers, which are therefore not comparable with sections 5 and 7. Non-priority classes may score lower.
- **Seasonality:** all flights fall between early May and mid October, so leaf-off conditions are not covered at all, and three test domains were flown later in autumn than any training domain (section 8.7). The date explains little of the NDVI variation, but its effect on the models is only visible in the per-domain tables, which are descriptive (many comparisons, no correction).
- **The fast batch has one seed and 2,000 steps;** differences below about 0.005 mIoU are not interpretable. **Known limitation of the report script:** it builds its comparison sections only when a seed has both an `rgb` and an `rgbi` run, so `ARMS` must always include `rgbi` (a batch of only `rgb` and `rgb_ndvi` trains and evaluates but is reported as "no paired comparison").

## 10. How to reproduce

Everything runs through one script; each stage checks its requirements first and every stage resumes after an interruption.

```
FLAIR_DATA_DIR=<data> bash run_train.sh check     # environment, packages, data, band order, split, weights
FLAIR_DATA_DIR=<data> bash run_train.sh           # probe, queue, train + evaluate, report
bash run_train.sh status                          # progress of every run
FLAIR_DATA_DIR=<data> bash run_train.sh diagnose  # NIR diagnostic (1 GPU)
OUT=./outputs/long STEPS=12000 ZERO_SHOT=0 MAX_GPUS=2 FLAIR_DATA_DIR=<data> bash run_train.sh
```

Extension batches (section 8). Put every setting on the same command line; `ARMS` must include `rgbi`:

```
# fast batch (3 arms, 1 seed, 2,000 steps, focused training, 2,000 test images)
OUT=./outputs/fast FLAIR_DATA_DIR=<data> ARMS="rgb rgbi rgb_ndvi" SEEDS="0" STEPS=2000 ZERO_SHOT=0 \
  TEST_LIMIT=2000 PRIORITY="5 6 7 8 10" bash run_train.sh
# long batch (3 seeds, 12,000 steps); copy memory.json and train_class_counts.json from the fast batch's train/ folder first
OUT=./outputs/long_focus FLAIR_DATA_DIR=<data> ARMS="rgb rgbi rgb_ndvi" SEEDS="0 1 2" STEPS=12000 ZERO_SHOT=0 \
  TEST_LIMIT=2000 PRIORITY="5 6 7 8 10" bash run_train.sh
# seasonality: acquisition dates (downloads 2.1 GB once, keeps one file) and pseudo-NDVI per class and domain
python scripts/season_check.py --split <output>/train/split.json --fetch_to <folder> --out <output>/train/season \
  --ndvi --root <data> --band_order rgb
```

Run the season check before the report stage so that the report picks up the dates automatically. (Section 8.7 reports the result for our split.)

Data download (only RGBI + labels, official test domains first):

```
python scripts/download_flair_hub.py --splits_dir <folder with FLAIR-HUB_*.csv> --dest <data> --budget_gb 100
```

| File | Role |
|---|---|
| `train/sam3_rgbi.py` | widened patch embedding, LoRA, semantic forward pass, prediction |
| `train/data.py` | RGBI loading, official split, deterministic paired training schedule |
| `train/metrics.py` | confusion, IoU, mIoU, paired bootstrap with per-class intervals |
| `scripts/train_sam3.py` | training loop (resume, checkpoints, out-of-memory exit code) |
| `scripts/eval_sam3.py` | test-set evaluation, per-image confusions |
| `scripts/train_report.py` | tables, intervals, per-class chart, rule-chosen figures |
| `scripts/nir_diagnostic.py` | NIR filter sizes, loss curves, NIR ablation |
| `scripts/gpu_queue.py` | shared-GPU queue (free-memory check, cap, re-queue) |
| `scripts/download_flair_hub.py`, `scripts/make_split.py` | data download and official split |
| `scripts/make_focus.py` | class counts per training patch and the sampling weights of the data focus |
| `scripts/season_check.py` | acquisition dates per domain and pseudo-NDVI per class and domain (section 8.7) |
| `tests/test_train.py`, `tests/test_gpu_queue.py`, `tests/test_run_train_sh.py` | tests without data or GPU, including an end-to-end run of `run_train.sh` on synthetic images |

## References

- Carion et al. SAM 3: Segment Anything with Concepts. arXiv:2511.16719, 2025.
- Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
- IGN. FLAIR-HUB, github.com/IGNF/FLAIR-HUB, data at huggingface.co/datasets/IGNF/FLAIR-HUB; Etalab Open Licence 2.0. FLAIR #1: Garioud et al., arXiv:2211.12979.
- Wu, Li & Hong. VegSAM: Vegetation-aware Adapter for Segment Anything Model in Urban Tree Segmentation. CVPR Workshops 2026 (NDVI injected through an adapter; implemented here as the `rgb_ndvi` and `rgbi_ndvi` arms, section 8; our arms compare it with giving the model the raw NIR band).
- All result numbers come from our own runs (report and diagnostic files in the repository and on the server).
