# Technical report: training SAM 3 with near-infrared on FLAIR-HUB

Group 7 · SRP 2025/26 · Experiment 3 in technical detail · status of 28 October 2026 (4,000- and 12,000-step runs complete)

This report documents exactly how Experiment 3 was built: the data, how the NIR band was added to SAM 3, what is trained, how training and evaluation work, how we made the RGB vs RGB + NIR comparison fair, and all results so far. Code paths refer to the project repository. For the plain-language overview, see the separate project-status document.

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

`scripts/gpu_queue.py` checks free memory every minute and starts a run only on a GPU with enough free memory plus a 1.5 GB margin, one of our runs per GPU, at most a set number of GPUs at once (currently 2, always the most free ones). It picks 2 or 1 images per pass to fit; the effective batch stays 8, so the experiment does not change. If another job takes the memory and ours runs out, the run exits with code 3, goes back in the queue and resumes from its last checkpoint. Observed speed: 7.4–13 s per optimizer step depending on how busy the GPU is, so 4,000 steps took roughly 8–15 hours per run.

## 4. Evaluation

- **Test set:** all 50,700 official test patches from 16 unseen domains.
- **Metric:** a 12 × 12 confusion matrix is accumulated over all test pixels with a label in 1–12; IoU per class = TP / (TP + FP + FN); **mIoU** = mean over the 12 classes.
- **Per-image confusion matrices are saved**, which allows paired statistics and later re-analysis without re-running the models.
- **Seeds:** we report the mean ± standard deviation of mIoU over the 3 seeds.
- **Significance:** paired bootstrap over test images. 1,000 times, we resample the 50,700 images with replacement (the same resample for both arms), sum the confusions (seeds pooled) and compute mIoU(RGB + NIR) − mIoU(RGB). The 2.5th and 97.5th percentiles give the 95% confidence interval. Per-class intervals use the same resamples.
- **Zero-shot reference:** untrained SAM 3 with the same 12 prompts and semantic head, to show what training adds.
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

## 8. Limitations and open points

- **Hyperparameters were fixed a priori**, not tuned. Results could change with a different learning rate, LoRA rank or number of prompts per image, for both arms.
- **The trained model is not few-shot**: it uses 37,900 labelled patches. It answers whether NIR helps SAM 3 once trained, and is the starting point for adapting to the drone data with few labels.
- **Reliance on NIR is a transfer risk.** A different camera (the drones) records NIR with different brightness and calibration. Planned test: train with random gain and offset on the NIR band only, then evaluate with NIR scaled by about ±20%.
- **Class prompts are fixed phrases** chosen before any result; different phrases might help some classes, but tuning them on the test set would be invalid.
- **Prediction is an uncalibrated argmax** over 12 independently trained sigmoid maps.
- **No comparison with published FLAIR-HUB baselines yet**; those numbers must be checked in the FLAIR-HUB paper before being cited.

## 9. How to reproduce

Everything runs through one script; each stage checks its requirements first and every stage resumes after an interruption.

```
FLAIR_DATA_DIR=<data> bash run_train.sh check     # environment, packages, data, band order, split, weights
FLAIR_DATA_DIR=<data> bash run_train.sh           # probe, queue, train + evaluate, report
bash run_train.sh status                          # progress of every run
FLAIR_DATA_DIR=<data> bash run_train.sh diagnose  # NIR diagnostic (1 GPU)
OUT=./outputs/long STEPS=12000 ZERO_SHOT=0 MAX_GPUS=2 FLAIR_DATA_DIR=<data> bash run_train.sh
```

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
| `tests/test_train.py`, `tests/test_gpu_queue.py`, `tests/test_run_train_sh.py` | tests without data or GPU, including an end-to-end run of `run_train.sh` on synthetic images |

## References

- Carion et al. SAM 3: Segment Anything with Concepts. arXiv:2511.16719, 2025.
- Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
- IGN. FLAIR-HUB, github.com/IGNF/FLAIR-HUB, data at huggingface.co/datasets/IGNF/FLAIR-HUB; Etalab Open Licence 2.0. FLAIR #1: Garioud et al., arXiv:2211.12979.
- Wu, Li & Hong. VegSAM: Vegetation-aware Adapter for Segment Anything Model in Urban Tree Segmentation. CVPR Workshops 2026 (related approach: NDVI via an adapter; we give the model the raw NIR band instead).
- All result numbers come from our own runs (report and diagnostic files in the repository and on the server).
