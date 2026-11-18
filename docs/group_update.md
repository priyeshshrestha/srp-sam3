# Project status: SAM on FLAIR, and teaching SAM 3 to use near-infrared

Group 7 · SRP 2025/26 · Few-Shot Learning for Biodiversity Assessment · status of 28 October 2026

This document explains everything we did in this phase, what we are doing now and what comes next, in plain language. Numbers marked **preliminary** come from a small test subset and will be redone on the full data. A glossary of terms is at the end.

## In one page

**What we wanted to know.** Which Segment Anything model (SAM 1, SAM 2 or SAM 3) is best at segmenting land cover in aerial images when it gets only one or five labelled examples per class? And does the near-infrared band (NIR), which vegetation reflects strongly, make the models better?

**What we found so far.**

- **SAM 3 is the best of the three for few-shot segmentation.** With one labelled example it reaches 0.334 mIoU, against 0.266 for SAM 2, and the difference is statistically clear. (Preliminary, small subset, SAM 1 not yet run.)
- **Few-shot is still hard.** Simply labelling the whole image as the class already scores 0.293. SAM 2 does not beat that; SAM 3 only just does.
- **NIR used as a hand-made rule hurts.** Filtering SAM 3's masks by vegetation index (NDVI) drops it from 0.334 to 0.212.
- **When we train SAM 3, everything changes.** Fine-tuned on FLAIR, SAM 3 reaches **0.62 mIoU** on 16 regions it has never seen (untrained: 0.18).
- **Training SAM 3 with NIR adds a small but real gain: +0.0095 mIoU** (95% confidence interval +0.008 to +0.011), positive in all three repetitions, and **exactly the same after 4,000 and after 12,000 training steps**. The gain is concentrated on vegetation: **coniferous forest about +0.06**, deciduous forest and grassland about +0.02 each.
- **The trained model relies heavily on NIR for trees.** If we take its NIR away, coniferous and deciduous forest each lose about 0.21–0.23 IoU. That is good news (it really learned to use NIR) and a warning (a different NIR camera, like the drones', could confuse it).

**Longer training is done.** Training three times longer (12,000 instead of 4,000 steps) improved both models by about 0.016 (RGB 0.622, RGB + NIR 0.631) and left the NIR gain unchanged.

**What comes next.** Redo the few-shot comparison on the real data including SAM 1; test whether the NIR model survives a change of camera; and, as soon as we have access, move to the EU drone data.

## 1. Where we started this phase

In the previous phase we worked on **LoveDA**, a Chinese land-cover dataset. Our results there (from the previous presentation):

| Approach | Setting | mIoU |
|---|---|---|
| HRNet-W32 | supervised, all labels | 0.580 |
| DeepLabv2 + CBST | domain adaptation, rural to urban | 0.351 |
| SAM 2 Hiera-S + FPN head | frozen encoder, 1.5 M trained parameters | 0.4746 |
| SAM 2 zero-shot | "oracle": uses the true labels to name SAM's masks | 0.846 |

Note: the HRNet number appears as 0.580 on the slides but as 0.5134 in our notes. We still need to check which one is right before we show it again.

Three things made us change course:

- **LoveDA has no near-infrared**, so the ecological part of our plan (NDVI, vegetation diversity) was impossible there.
- The **ISPRS Potsdam** images were taken in winter, without leaves: tree NDVI was about 0.21, the same as bare soil.
- Training a prompt head on top of a frozen SAM stalled at 0.06 mIoU on new classes.

So we moved to **FLAIR** and restarted with a clean, controlled comparison.

## 2. Why FLAIR

FLAIR is a large French land-cover dataset from IGN (the French mapping agency). It is the closest public stand-in for our EU drone data:

- **Every image has red, green, blue and near-infrared** (plus a height band that we deliberately do not use, because the drones have none).
- **0.2 m per pixel**, European landscapes.
- **Many regions and dates** ("domains"), so we can always test on regions the model has never seen. That measures real generalisation, not memorisation.
- **Ecological classes**: coniferous vs deciduous forest, brushwood, vineyard, grassland, crops, plowed land, and so on. We evaluate the 12 main classes.

There are two releases. **FLAIR #1** (used in an earlier phase) and the newer **FLAIR-HUB**. They store files differently, and FLAIR-HUB numbers its classes in a different order (for example, class 12 means "deciduous" in FLAIR-HUB but "plowed land" in FLAIR #1). Our code reads both and translates FLAIR-HUB labels into the FLAIR #1 numbering, so nothing gets mixed up.

## 3. Two mistakes we found and fixed

**Colour channels were swapped in our earlier FLAIR results.** Our old code assumed the image bands were stored blue-green-red. The dataset documentation says red-green-blue, and when we checked the actual files (water is bluer than it is red, bare soil redder than blue) they are red-green-blue. So the earlier FLAIR results (for example SAM 1 at 0.286) were very likely computed on colour-swapped images. We retired those numbers. Now, every run checks the channel order on the real files first, and **stops instead of guessing** if it cannot tell.

**The download script fetched a different dataset version than the code expected.** The toy download was FLAIR-HUB, but the code only understood FLAIR #1. Now both work.

## 4. Experiments 1 and 2: few-shot SAM 1 vs SAM 2 vs SAM 3

### The setup in simple words

An **episode** is one small test: we pick a class (say, vineyard), give the model one image where vineyard is labelled (the **support**), and ask it to find vineyard in a second image (the **query**). The support and query always come from **different regions**, so the model cannot just recognise the same field.

- 300 episodes: 12 classes × 25. **Every model runs exactly the same episodes.**
- The class must cover between 2% and 90% of both images. (If a class fills the whole image, "label everything" scores almost 1.0 and the test means nothing. An earlier result of 51.64% IoU turned out to be exactly that.)
- **No training anywhere.** All models are used as they come.

**How each model uses the example.** SAM 1 and SAM 2 cannot take an example image directly. We compute an average "fingerprint" of the class from the support, find the most similar spots in the query, and turn those into point and box prompts (the PerSAM method). SAM 3 understands concepts but also cannot take an example from another image, so we paste the support and query into one picture, draw a box on the support part, and keep SAM 3's answer for the query part (the canvas method of Tsai and colleagues, 2026). So the comparison is really "each model with its best available few-shot method".

### Results (preliminary: 250-image FLAIR-HUB subset, run on a laptop GPU)

| Setting | mIoU |
|---|---|
| "Label everything" baseline | 0.293 |
| SAM 2, 1 example | 0.266 |
| SAM 2, 5 examples (120 episodes) | 0.345 |
| SAM 2, ideal prompt from the true answer (upper bound) | 0.601 |
| **SAM 3, 1 example** | **0.334** |
| SAM 3, falling back to SAM 2 when SAM 3 returns nothing | 0.371 |

![Few-shot results on identical episodes. Preliminary.](presentation/out/fig/fewshot.png)

What this tells us:

- **SAM 3 beats SAM 2 by +0.068**, with a 95% confidence interval of +0.024 to +0.117, so the difference is real. SAM 3 is best on 7 of 12 classes (for example vineyard 0.736, plowed land 0.631, building 0.456).
- **SAM 2's mask drawing is fine** (0.601 with an ideal prompt). What fails is *finding* the class in an image from another region.
- **SAM 3 returns no mask at all in 34% of episodes.** It is built to find objects, and land cover like "grassland" is not an object. Using SAM 2's answer in those cases gives 0.371, better than either model alone.
- **SAM 3 is about five times slower** (2.8 s vs 0.5 s per episode on a laptop GPU).

Not yet done: SAM 1 (needs the cluster), and SAM 3 with five examples, with ideal prompts, and with the class name only. All of this will be rerun on the full data.

## 5. Near-infrared after segmentation: measuring ecology

Before training anything, we tried a simple use of NIR. SAM 3 segments from colour only; then we compute, inside each predicted segment:

- **NDVI**, a standard vegetation index: (NIR − red) / (NIR + red). Healthy vegetation scores high.
- **Shannon entropy** of NDVI in small windows of 1.4 m, as a measure of how mixed the vegetation is.

| Measured on predicted segments, vs the true segments | SAM 2 | SAM 3 |
|---|---|---|
| NDVI error (average) | 0.082 | **0.059** |
| NDVI correlation with the truth | 0.73 | **0.86** |
| Covered area error | 30.8 points | 21.2 points |
| Diversity (Shannon) correlation | 0.68 | **0.80** |
| mIoU: no filter / with NDVI filter | 0.266 / 0.206 | 0.334 / 0.212 |

So NIR is very useful for **measuring** vegetation once we have segments. But using it as a **hard filter** (keep only pixels whose NDVI looks like the example's) hurts badly: an NDVI range from one region or season does not fit another.

That raised the next question: instead of a hand-made rule, **what if the model learns to use NIR itself?**

## 6. Experiment 3: training SAM 3 with near-infrared

### The idea in simple words

SAM 3 was trained on ordinary colour photos, so it expects three channels. We **widen its very first layer from three inputs to four**, so it can take NIR as well. The new NIR connections **start at exactly zero**, which means that at the start of training, the model is exactly the original SAM 3 (we verified this: the outputs are identical). Training then decides by itself how much NIR to use.

We do not retrain all of SAM 3 (841 million parameters would not fit on our GPUs). Instead we train about **16.5 million parameters**:

- the widened first layer,
- small **LoRA** adapters inside the image encoder (a standard way to fine-tune big models cheaply),
- the part of SAM 3 that combines the class name with the image, and its built-in "semantic" output head.

For each of the 12 classes, the model gets the class name as a text prompt (for example "conifer tree") and produces a map; each pixel gets the class with the highest score.

### The control: why we trust the result

We train the **same model twice**: once with RGB only, once with RGB + NIR. Everything else is identical: the same images in the same order, the same random flips and rotations, the same class prompts, the same starting weights. **The NIR channel is the only difference.** We repeat this with three different random seeds.

- **Data:** the official FLAIR-HUB split. 37,900 training images from 21 regions, and **50,700 test images from 16 regions that never appear in training**.
- **Training:** 4,000 steps of 8 images each, on the shared GPUs of the university server.
- We also measured **untrained SAM 3** with the same class names, to see how much training itself adds.

### Results (first runs, 4,000 training steps)

| Model | mIoU (mean ± spread over 3 seeds) |
|---|---|
| SAM 3, untrained (class names only) | 0.180 |
| SAM 3 fine-tuned, RGB | 0.6062 ± 0.0021 |
| **SAM 3 fine-tuned, RGB + NIR** | **0.6157 ± 0.0004** |

**Adding NIR: +0.0095 mIoU, 95% confidence interval +0.0084 to +0.0107.** Every seed agrees (+0.007, +0.011, +0.010). That is about one point: small, but clearly not chance.

![Change in IoU per class when NIR is added (first runs, 4,000 steps), with 95% confidence intervals. Green = significant gain, dark grey = significant loss, light grey = not significant.](docs/fig/train_delta_4k.png)

| Class | RGB | RGB + NIR | Change | Real? |
|---|---|---|---|---|
| Coniferous forest | 0.555 | 0.620 | **+0.065** | yes |
| Deciduous forest | 0.689 | 0.707 | +0.018 | yes |
| Grassland (herbaceous) | 0.493 | 0.512 | +0.018 | yes |
| Plowed land | 0.348 | 0.356 | +0.008 | yes |
| Water | 0.869 | 0.875 | +0.007 | yes |
| Crops (agricultural land) | 0.533 | 0.540 | +0.007 | yes |
| Bare soil | 0.647 | 0.652 | +0.005 | yes |
| Brushwood | 0.305 | 0.309 | +0.003 | yes |
| Vineyard | 0.718 | 0.715 | −0.003 | no |
| Impervious surface (roads) | 0.742 | 0.741 | −0.001 | yes, tiny |
| Building | 0.832 | 0.828 | −0.004 | yes, tiny |
| Pervious surface (lawns, gardens) | 0.545 | 0.534 | **−0.011** | yes |

In short: **NIR helps 8 of 12 classes, mostly vegetation**, exactly where ecology says it should. Buildings and roads barely change, because colour already separates them. The only loss worth mentioning is lawns and gardens, probably because in NIR they look like grassland; with longer training (section 8) this loss disappeared.

### Two examples

These were chosen automatically (the test images where NIR helped and hurt the most, among images with at least three classes), not by hand. Single images always exaggerate; the evidence is the table above.

![Where NIR helped most: the RGB model calls the whole meadow cropland; with NIR it correctly sees grassland.](outputs/train/report/figures/01_D015-2020_AERIAL_RGBI_AF-S1-17_8-5.png)

![Where NIR hurt most: with NIR, the model merges a plowed field into the neighbouring crop field.](outputs/train/report/figures/05_D085-2019_AERIAL_RGBI_AA-S1-28_8-10.png)

### Does the model really use NIR? The diagnostic

We tested the trained RGB + NIR models on 1,000 test images in three ways: with the real NIR, with NIR flattened (every pixel set to the image's average NIR, so no detail is left), and with NIR taken from a different image.

| Seed | RGB-only model | RGB + NIR, real NIR | NIR flattened | NIR from another image |
|---|---|---|---|---|
| 0 | 0.628 | 0.630 | 0.588 | 0.569 |
| 1 | 0.625 | 0.631 | 0.587 | 0.574 |
| 2 | 0.620 | 0.630 | 0.589 | 0.570 |

- **Taking NIR away costs 4–6 points**, much more than the 1 point NIR adds. So the model does not use NIR as a small extra: it has moved part of its work onto NIR.
- **That part is trees.** With NIR flattened, **coniferous drops by 0.21 and deciduous by 0.23**, while every other class changes by less than 0.03. With NIR from another image, several other classes (roads, bare soil, water, plowed land, lawns) also lose about 0.04–0.05: the model uses NIR lightly for those too.
- One caveat: flattened or swapped NIR is something the model never saw in training, which exaggerates the drops. But the effect is concentrated on exactly the tree classes and identical across seeds, so the reliance is real.

**Why this matters for the drones.** The tree classes now depend on what NIR looks like. A drone camera measures NIR with a different brightness and calibration than FLAIR's aircraft camera. If the values shift, the trees could be misclassified. A model trained only on RGB does not have this weakness. We plan to test exactly this (see section 9).

The training loss was still slowly falling in the middle of training (from about 0.78 at the start to 0.47 at mid-training and 0.42 at the end), which is why we are now training longer.

## 7. How the code works (for anyone who wants to run it)

Everything is in one repository. Two commands do everything, and both check every requirement **before** spending GPU time:

- `bash run.sh`: the few-shot comparison (Experiments 1 and 2). Checks the data, decides the colour channel order from the files, builds one shared list of episodes, runs every model, and writes the metrics and 10 example figures.
- `bash run_train.sh`: the training experiment (Experiment 3). Checks the environment and data, makes the official train/test split, measures how much GPU memory one run needs, then queues the runs, evaluates them and writes the report.
- `bash run_train.sh status` shows what is running; `bash run_train.sh diagnose` runs the NIR diagnostic.

Rules built into the code that protect the results:

- **Nothing is guessed:** if the colour order is unclear, or a file is missing, the script stops and says how to fix it.
- **Results from different settings are never mixed.** Every run records its exact settings and data split; the scripts refuse to combine runs that differ.
- **Example figures are chosen by a fixed rule after all runs**, never by hand.
- **The presentation is generated from the result files.** No number on the slides is typed in by hand (except the recap of the previous phase and published literature facts).
- **Everything resumes after an interruption:** re-running the same command continues where it stopped.
- There are automatic tests for all of this that need no data and no GPU, including a full test run of both scripts on synthetic images.

**The GPU server is shared with other students.** None of its 8 GPUs (16 GB each) is ever completely free. So the training script measures how much memory a run needs (about 6 GB with 1 image at a time, 9 GB with 2), and a small queue starts each run only on a GPU that has enough free memory at that moment, with a safety margin. If another student's job takes the memory and ours crashes, it goes back in the queue and continues from its last save (saved every 10 minutes). We never touch other people's jobs, and we can cap how many GPUs we use at once (currently 2).

**Data on the server.** We downloaded about 100 GB of FLAIR-HUB from Hugging Face: only the colour + NIR images and the labels, for all 16 official test regions and 21 training regions. (The full dataset with all satellite data would be about 750 GB.)

## 8. Longer training: done

We trained the same RGB and RGB + NIR models **three times longer** (12,000 steps instead of 4,000, about 2.5 passes over the training data), 3 seeds each. Only the number of steps changed.

| Model | 4,000 steps | 12,000 steps |
|---|---|---|
| Fine-tuned, RGB | 0.606 | 0.622 |
| Fine-tuned, RGB + NIR | 0.616 | 0.631 |
| NIR gain (95% confidence interval) | +0.0095 (+0.008 to +0.011) | +0.0095 (+0.008 to +0.011) |

- **Both models improve by the same amount** (about 1.6 points), so more training helps, but only modestly.

![Change in IoU per class when NIR is added, 12,000-step runs, with 95% confidence intervals.](docs/fig/train_delta_12k.png)

- **The NIR gain does not change.** That makes it a solid result: it was not caused by too little training.
- Per class, the picture stays the same: coniferous forest +0.057, deciduous +0.021, grassland +0.017. Seven of twelve classes gain significantly. The small loss on lawns and gardens seen in the shorter runs **disappeared**; the only remaining significant loss is buildings (−0.003, negligible).

## 9. Next steps

1. **Redo the few-shot comparison (Experiments 1 and 2) on the real data, with SAM 1.** The current few-shot numbers come from a 250-image subset on a laptop. The full data is now on the server. This is the biggest weakness in what we can show today.
2. **Test robustness to a different NIR camera.** Train with random brightness and contrast changes on the NIR band only, so the model learns NIR patterns rather than exact values. Then test with NIR deliberately scaled up and down by about 20%, to see whether trees survive a change of camera. This directly tests whether our model can transfer to the drones.
3. **Compare with published FLAIR-HUB baselines** on the same official split, so we can say whether 0.63 mIoU is good. (We must check those numbers in the FLAIR-HUB paper first.)
4. **Look at where NIR still hurts** in the saved results: plowed fields merged into crops (the most-hurt test images in both runs), and lawns vs grassland in the shorter runs.
5. **Get access to the EU drone data.** This is still our biggest risk for the final phase. Any help here would make a big difference.
6. Smaller to-dos: check the HRNet recap number (0.580 vs 0.5134); save the code to git.

## 10. What to be careful about when presenting

- The **few-shot numbers are preliminary** (300 episodes, small subset, SAM 1 missing).
- The few-shot comparison is "each model with its best available prompting method", not identical prompting.
- The **trained model is not few-shot**: it learned from 37,900 labelled images. It answers a different question (does NIR help once SAM 3 is trained), and is the starting point for adapting to the drones with few labels.
- The two example images are illustrations. The **evidence is the aggregate**: +0.0095 with its confidence interval, and the per-class table.
- The NIR diagnostic uses inputs the model never saw, so the size of the drops is exaggerated; the pattern (trees) is the finding.

## Glossary

**IoU (intersection over union):** how well a predicted area overlaps the true area: 1 = perfect, 0 = no overlap.

**mIoU:** the average IoU over all classes. Our main score.

**Few-shot:** the model gets only a few labelled examples (here 1 or 5) of each class.

**Zero-shot / untrained:** the model is used as it comes, with no examples or training on our data.

**Fine-tuning:** continuing to train a pretrained model on our data.

**LoRA:** a cheap way to fine-tune a huge model by training small add-on matrices instead of all its weights.

**NIR (near-infrared):** light just beyond red, invisible to us. Healthy leaves reflect it very strongly.

**NDVI:** (NIR − red) / (NIR + red). A standard index of how green and healthy vegetation is.

**Domain / region:** one area of France photographed at one date. Testing on unseen domains means testing on places and dates the model never saw.

**Seed:** the random starting point of a training run. Repeating with several seeds shows whether a result is stable or luck.

**95% confidence interval:** the range in which the true value very likely lies. If it does not include zero, the difference is real, not chance.

**Oracle:** an upper bound that cheats by using the true answer; it shows the best possible result, not a usable method.

## Sources

- Kirillov et al., Segment Anything, ICCV 2023. Ravi et al., SAM 2, arXiv:2408.00714, 2024. Carion et al., SAM 3: Segment Anything with Concepts, arXiv:2511.16719, 2025.
- Tsai, Lin & Wang, Few-Shot Semantic Segmentation Meets SAM3, arXiv:2604.05433, 2026. Zhang et al., PerSAM, arXiv:2305.03048, 2023.
- Hu et al., LoRA, ICLR 2022. Wu, Li & Hong, VegSAM, CVPR Workshops 2026.
- IGN, FLAIR #1 (arXiv:2211.12979) and FLAIR-HUB (github.com/IGNF/FLAIR-HUB); data under Etalab Open Licence 2.0.
- All result numbers in this document come from our own runs (result files in the repository).
