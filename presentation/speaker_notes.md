# Speaker notes — final presentation (20 minutes, 20 slides)

Timing budget: about 1 minute per slide. Content slides (10, 12, 14, 17) get a little more; recap and
references get less. **Bold** = the sentence to land. *[Pause]* marks a moment to let a number sink in.

Every result number below comes from the runs in `outputs/runs`. They are **preliminary**: FLAIR-HUB toy
subset, 300 episodes, 10 domains, run on a laptop GPU. If you rebuild the deck after more runs finish,
re-check the numbers in these notes against the slides.

---

## 1. Title (0:00 – 0:30)

- Good morning. We are Group 7, and this is the final presentation of our project on few-shot learning
  for biodiversity assessment.
- Last time we showed baselines on LoveDA. Today is about three things:
  1. why we changed dataset;
  2. a controlled comparison of SAM 1, SAM 2 and SAM 3 for few-shot habitat segmentation;
  3. what near-infrared can add on top of the segmentation.
- **Everything is training-free.** No model was fine-tuned for any result in this talk.

## 2. Recap — problem and research question (0:30 – 1:30)

- The motivation hasn't changed. Biodiversity monitoring needs habitat maps across many regions.
  UAVs make the imagery cheap: 10 to 30 cm per pixel.
- **The bottleneck is labels.** Pixel-level annotation needs ecologists, so a real project might have a
  handful of labelled images per habitat class.
- That gives the question on the right: given K = 1 or 5 labelled examples of a class (the support set),
  can a model segment that class in a new image from an unseen region (the query)?
- At the end of the last phase the plan was a SAM 2 encoder, prototype-based few-shot, NDVI and
  Shannon channels, then transfer to the EU UAV data.

## 3. Recap — results, and what had to change (1:30 – 2:45)

- On the left are the LoveDA numbers from last time: supervised HRNet 0.58, domain adaptation 0.351,
  SAM 2 with a trained FPN decoder 0.47. The 0.85 SAM 2 "zero-shot" number is an oracle: it uses the
  ground truth to label SAM's masks. It's an upper bound, not a model.
- On the right is what we learned, and each point forced a change:
  - **LoveDA is RGB only.** No near-infrared, so NDVI and Shannon over NDVI cannot even be computed.
    The ecological part of our plan was impossible on that dataset.
  - On ISPRS Potsdam, the other aerial benchmark we tried, the imagery was flown leaf-off. Tree NDVI
    sits around 0.21, overlapping bare soil, so NDVI cannot separate vegetation there.
  - Training a small prompt head on a frozen SAM stalled at 0.06 novel-class mIoU.
- So we changed two things: a dataset with NIR, and prompting frozen foundation models without
  training.

## 4. Why we moved to FLAIR (2:45 – 3:45)

- This table compares the candidate datasets with our actual target, the EU UAV data.
- **FLAIR is the only public dataset here with NIR on every patch, sub-metre resolution, and European
  landscapes.** At 0.2 m it is in the same range as UAV imagery.
- It covers many départements, flown on different dates. That matters twice:
  - season becomes a variable instead of a fixed handicap like Potsdam's leaf-off flight;
  - we can put the support image and the query image in **different regions or dates**. That tests
    transfer rather than memorising neighbouring tiles.
- Its classes are ecological: coniferous vs deciduous, brushwood, vineyard, herbaceous vegetation.
  LoveDA and Potsdam use cartographic classes like "tree" vs "low vegetation".

## 5. The FLAIR dataset (3:45 – 4:40)

- FLAIR is produced by IGN, the French mapping agency: about 93,000 patches of 512 by 512 pixels at
  20 cm, from 55 spatio-temporal domains, with 19 land-cover classes.
- Each patch has red, green, blue and near-infrared, plus a height model. **We deliberately do not use
  height: the EU UAV sensor has none**, so anything built on it couldn't transfer.
- The images on the right are real patches from our evaluation set, with labels in the official colours.
- Two practical findings while setting this up:
  - the toy download is actually FLAIR-HUB, whose label numbering differs from the original FLAIR. We
    remap it explicitly, otherwise deciduous forest would silently be read as plowed land;
  - **we checked the band order on the files.** Our earlier loader assumed blue-green-red; the files
    are red-green-blue. Our previous FLAIR numbers were very likely computed on colour-swapped images,
    so we no longer use them.
- For today we evaluate 12 classes on 250 patches from 10 domains.

## 6. SAM 1 → SAM 2 → SAM 3 (4:40 – 5:40)

- Three generations of Segment Anything, all frozen in our work.
- **SAM 1** (2023): ViT image encoder trained on 11 million images and 1.1 billion masks. It takes points,
  boxes and masks as prompts and returns masks with no class label. We use ViT-B, 91 million parameters.
- **SAM 2** (2024): hierarchical Hiera encoder with multi-scale features, and a memory mechanism for
  video. We use Hiera-S, 46 million parameters, the same encoder as last phase.
- **SAM 3** (late 2025) is a different kind of model. It is ten to twenty times larger: 841 million
  parameters, about 40% of that a text encoder. **It doesn't just segment the object you point at; it
  segments every instance of a concept**, given a text phrase or example boxes.
- The last row is the important one. SAM 1 and 2 need a hand-built prompting pipeline for few-shot.
  SAM 3 has an example-based prompt built in.

## 7. SAM 3 — Promptable Concept Segmentation (5:40 – 6:40)

- How SAM 3 works, simplified. The prompt can be a text phrase like "vineyard", an example box drawn on
  the image, or both.
- The image goes through the Perception Encoder, shared between an image detector and a video tracker.
- A fusion encoder combines image and prompt. A DETR-style decoder then proposes instances, and a
  **presence head** first decides "is this concept in the image at all?" before localising it. That cuts
  false positives.
- It was trained on SA-Co, a data engine with 4 million unique concept labels including hard negatives.
- **One limitation drives our design: example boxes must be in the same image you segment.** You cannot
  hand it a support image and a different query image. Next slide shows how we get around that.

## 8. How each model does few-shot (6:40 – 7:50)

- To compare fairly, every model gets **exactly the same episodes**. An episode is one class, one or
  five labelled supports, and one query.
- **SAM 1 and SAM 2: prototype prompting, PerSAM-style.**
  1. Encode the support and average its features under the mask: that's the class prototype.
  2. Encode the query and compute a cosine-similarity map against the prototype.
  3. Take the most similar point as positive, the least similar as negative, plus a box around the
     high-similarity region, and let SAM's decoder produce the mask.
  The pipeline is identical for both, so any difference is the model.
- **SAM 3: the spatial-concatenation canvas** from Tsai et al., 2026. Paste support on top and query
  below in one image, draw the example box on the support half, run SAM 3, and crop the query half.
  On the right is one real canvas from our data.
- Protocol: 300 episodes, 12 classes × 25. **Support and query always come from different domains.**
  The class must cover 2 to 90% of both patches, because if a class fills the frame, predicting
  everything scores almost perfectly.
- We report class-accumulated mIoU, the standard few-shot metric, plus a predict-all baseline and
  confidence intervals from paired bootstrapping.
- Fairness caveat to say out loud: we compare **each model with its best available few-shot recipe**,
  not bare encoders.

## 9. Experiment 1 — upper bounds (7:50 – 8:50)

- Before the few-shot task: is the model or the prompting the bottleneck?
- **Oracle** means the prompt is taken from the query's own ground truth: a box around the largest true
  region, a positive point inside and a negative point outside. That's cheating on purpose; it measures
  how well the model can segment when told exactly where to look.
- **SAM 2 with oracle prompts reaches 0.601. With its own few-shot prompts it gets 0.266.** *[Pause]*
- So SAM 2's decoder is not the problem. More than half of the performance is lost in finding the
  class across domains: prototype matching on SAM 2's features doesn't transfer between regions.
- The dashed line is the predict-all baseline at 0.293, labelling the entire patch as the class.
- If the SAM 3 oracle and text-only bars are on the slide, add: the text-only arm tells us whether the
  class name alone is enough, or whether one labelled example is worth more.

## 10. Experiment 2 — few-shot results (8:50 – 10:20)

- The main result.
- **One-shot: SAM 3 0.334, SAM 2 0.266.** The difference is +0.068 with a 95% confidence interval of
  +0.024 to +0.117, so it is significant: the interval excludes zero.
- Now the uncomfortable number: **SAM 2's one-shot result is below the predict-all baseline of 0.293.**
  Labelling the whole patch as the class beats SAM 2's prototype prompting. SAM 3 is above the baseline.
  This is why we always show that baseline: without it, 0.27 sounds like a result.
- Five shots help SAM 2 a lot: +0.044 on the same 120 episodes, to 0.345. Averaging five prototypes
  stabilises the matching. SAM 3 five-shot is shown if it finished; otherwise it's marked pending.
- Two columns to mention:
  - **SAM 3 returns no mask at all on 34% of episodes.** It's an instance detector, and on diffuse
    classes it often decides the concept isn't there;
  - SAM 3 is about five times slower: 2.8 versus 0.5 seconds per episode on our laptop GPU.
- SAM 1 is marked pending: it runs on the university cluster and will be added there.
- Caveat: toy subset, 300 episodes, so preliminary.

## 11. Per-class IoU (10:20 – 11:10)

- One-shot IoU per class. The black tick on each class is the predict-all baseline. A bar under its tick
  means the model did worse than labelling the whole patch.
- **SAM 3 wins 7 of 12 classes**, and its wins are large on object-like or strongly textured classes:
  vineyard 0.74 vs 0.44, plowed land 0.63 vs 0.35, buildings 0.46 vs 0.20.
- **SAM 2 wins the other 5**: coniferous, agricultural land, water, bare soil, brushwood. These are big,
  homogeneous areas, where a prototype-similarity map works well and SAM 3 sometimes returns nothing.
- Count the bars under the ticks: SAM 2 is below the baseline on 8 of 12 classes, SAM 3 on 5.
  Diffuse "stuff" classes like pervious surface and brushwood are hard for everyone.

## 12. Examples — where the models disagree most (11:10 – 12:10)

- These are not hand-picked. After all runs, a fixed rule picked the five episodes with the largest gap
  between models.
- How to read each row: the support with its mask in magenta, the query, the ground truth in magenta,
  then each model's prediction. Green is correct, red a false positive, blue missed.
- Each row label shows the class and which domain the support and query came from: always different.
- The typical pattern: **SAM 2 locks onto a wrong region and gets nearly zero, while SAM 3 finds the
  whole field** with IoU above 0.9. Look at the vineyard and plowed-land rows: SAM 3 has recognised the
  concept, not just copied colours.

## 13. Examples — where all models succeed or fail (12:10 – 13:00)

- The same rule picked three episodes where every model does well and two where every model fails.
- Successes: large, distinctive regions such as coniferous forest, buildings and herbaceous fields.
- Failures: small, scattered targets, buildings and pervious surfaces covering a few percent of the
  patch. **SAM 2 segments something else; SAM 3 returns no mask at all.** That is the 34% from before,
  made visible.
- This is the honest part of the picture: few-shot on small, diffuse land-cover targets is unsolved by
  all three.

## 14. Why we continue with SAM 3 (13:00 – 14:15)

- Four cards, all computed from the runs.
  1. **Accuracy:** +0.068 mIoU over SAM 2 in one-shot, significant.
  2. **Classes:** best on 7 of 12.
  3. If present, **more shots:** how much SAM 3 gains from 1 to 5 examples.
  4. **Cost:** about five times slower than SAM 2, still around a second per episode on a 16 GB GPU, so
     feasible on our cluster.
- And it accepts example images and text natively, so we don't need a hand-built prompting pipeline.
- The trade-off is the empty masks. A simple fix needs no labels and no tuning: **when SAM 3 returns
  nothing, use SAM 2's prediction.** That combination reaches 0.371, better than either alone.
  *[Pause]* The two models fail in complementary ways.

## 15. Next: adaptive SAM 3 with an NDVI adapter (14:15 – 15:15)

- This slide is a proposal, not a result. **We haven't trained it yet, and the slide says so.**
- The idea comes from VegSAM, Wu, Li and Hong, CVPR Workshops 2026. They adapt SAM to multispectral
  urban-tree segmentation by injecting an NDVI prior through a lightweight adapter inside the image
  encoder, with LoRA for parameter-efficient tuning. They report about one IoU point over their baseline
  on Potsdam and Vaihingen.
- Our version: keep SAM 3 frozen, add LoRA to the encoder and a small NDVI adapter fed by NIR and red.
- To stay few-shot, the adapters are trained only on base classes and evaluated on novel classes,
  rotating which classes are novel.
- It needs only RGB and NIR, so it would run on the EU UAV sensor. Its baseline is the SAM 3 one-shot
  number we just showed.

## 16. SAM 3 + NDVI — segment, then measure (15:15 – 16:05)

- This is the architecture we can run today, and it uses NIR in a different way.
- **SAM 3 does the segmentation from RGB only, untouched.** Near-infrared enters only afterwards:
  1. compute NDVI from NIR and red;
  2. compute local Shannon entropy of NDVI over a 7-pixel window, 1.4 metres, as a heterogeneity
     measure;
  3. summarise both per predicted segment: mean NDVI (vegetation vigour), cover, and heterogeneity.
- The output is what an ecologist actually wants: a habitat map with vigour and diversity indicators,
  each traceable to a segment.
- We also tested an optional NDVI filter: keep only predicted pixels whose NDVI falls in the range seen
  in the support image. It uses no query labels.

## 17. Results — ecology from the segments (16:05 – 17:20)

- Real examples on the left: the RGB patch, SAM 3's segment, the NDVI map, NDVI inside the segment, and
  the Shannon heterogeneity map.
- The table compares indicators from predicted segments with those from the ground-truth segments of the
  same patch, vegetation classes only:
  - NDVI of the segment: **SAM 3 is off by 0.059 on average, correlation 0.86.** SAM 2: 0.082 and 0.73;
  - vegetation cover is harder: off by 21 percentage points for SAM 3, 31 for SAM 2. Area depends
    directly on mask quality;
  - local heterogeneity correlates at 0.80 for SAM 3 and 0.68 for SAM 2.
- **Negative result, and an important one:** the NDVI filter hurts, SAM 3 0.334 to 0.212 and SAM 2
  0.266 to 0.206. An NDVI range taken from a support in one region or date doesn't fit a query from
  another; NDVI shifts with season and site. Relevant for the EU UAV phase, where sites and dates differ
  even more.
- Conclusion for this slide: **NIR is most useful after segmentation, for measuring ecology, not as a
  hard filter.**

## 18. Conclusion (17:20 – 18:30)

- Five take-aways.
  1. FLAIR made the ecological question testable: NIR, 20 cm, European landscapes, 10 domains in our set.
  2. On identical cross-domain episodes, **SAM 3 beats SAM 2 significantly, 0.334 vs 0.266**, and only
     SAM 3 beats predicting the whole patch.
  3. SAM 2's decoder is fine, 0.60 with oracle prompts. Its weakness is cross-domain matching. SAM 3's
     weakness is returning nothing on a third of episodes; falling back to SAM 2 there gives 0.371.
  4. NIR adds value after segmentation: segment NDVI correlates 0.86 with the ground truth, while an NDVI
     filter hurts.
  5. Everything uses only RGB and NIR, no height, so it transfers to the EU UAV sensor.
- Limits, said plainly: toy subset, 300 episodes, SAM 1 still pending on the cluster.

## 19. Future work — the EU UAV data (18:30 – 19:30)

- The last phase in four steps:
  1. ingest the EU UAV data from Portugal, Germany, the UK and Austria, with the same band-order check
     that caught our earlier mistake;
  2. run few-shot SAM 3 with one or five labelled tiles per habitat;
  3. train the adaptive SAM 3 with the NDVI adapter;
  4. the ecological read-out per habitat and country.
- Open questions: do supports from FLAIR transfer to EU habitats, or must they come from the same
  country? How different is the UAV's NIR response, which affects NDVI calibration? What do we do on
  sites without NIR?
- Biggest risk: **access to the EU UAV data is still pending.** We'd appreciate help with that.

## 20. References (19:30 – 20:00)

- Don't read the list. Point out that the deck relies on the SAM papers, the few-shot canvas method by
  Tsai et al., VegSAM, and the FLAIR dataset from IGN.
- SAM 3's licence requires acknowledgement in publications, and it is cited here.
- Thank you. We're happy to take questions.

---

## Likely questions and short answers

- **"Is the SAM 2 vs SAM 3 comparison fair? They use different prompting."** Each model uses its best
  available few-shot recipe on identical episodes. SAM 2's oracle number (0.60) shows its decoder isn't
  being handicapped. The gap is in how the reference is used.
- **"Why is the baseline so high?"** Many land-cover classes cover large parts of a patch. Predicting
  everything scores the class's coverage, so anything below 0.29 has learned nothing on average. We cap
  coverage at 90% to keep it from dominating.
- **"Why should we believe 300 episodes?"** The SAM 3 vs SAM 2 difference has a 95% confidence interval
  from paired bootstrapping that excludes zero. Per-class differences are noisier; vineyard and plowed
  land reuse about 10 distinct queries.
- **"Why not fine-tune?"** Training-free keeps the measurement clean: the only thing changing is the
  model. The adaptive SAM 3 is the planned trained step, evaluated on novel classes only.
- **"What about your earlier FLAIR results?"** They were computed with a wrong band order assumption,
  very likely on colour-swapped images, so we've retired them.
- **"Why is SAM 3 empty so often?"** It's an instance detector with a presence head. On diffuse classes
  it often concludes the concept isn't present at the default 0.5 confidence threshold. A threshold
  chosen on held-out domains is a planned test; it must not be tuned on the test episodes.
