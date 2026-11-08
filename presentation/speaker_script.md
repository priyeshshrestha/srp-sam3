# Word-for-word speaker script

Final presentation · Group 7 · Few-Shot Learning for Biodiversity Assessment
22 slides · about 21–22 minutes · roughly 140 words per minute

How to use this script: read the plain text aloud as written. Lines in [brackets] are stage directions; don't read them. **Bold** words get a little extra stress. [PAUSE] means stop for one full breath.

Numbers match the deck as built on 21 October 2026: few-shot results on the FLAIR-HUB toy subset (300 episodes, SAM 1 pending); training results (Experiment 3) on the official FLAIR-HUB split, 50,700 test patches, 12,000-step runs (4,000-step runs as comparison). If you rebuild the deck with more runs, check the numbers on slides 9, 10, 14, 18 and 20 again.

## 1 · Title · 0:00

[Stand still, look at the audience, wait two seconds before speaking.]

Good morning, everyone. We are Group 7, and this is the final presentation of our project on few-shot learning for biodiversity assessment.

Last time, we showed you our baselines on LoveDA. Today, we want to show you three things. First, why we changed our dataset. Second, a controlled comparison of three generations of the Segment Anything Model: SAM 1, SAM 2 and SAM 3. And third, what near-infrared light can add on top of the segmentation.

One thing to keep in mind for the whole talk: **nothing you see today was trained.** Every model is frozen. [PAUSE]

[Click.]

## 2 · Recap: problem and research question · 0:35

Let me quickly remind you of the problem.

Biodiversity monitoring needs habitat maps, across many regions and many landscapes. Drones make the images cheap. We get ten to thirty centimetres per pixel, at scale.

But the bottleneck is **labels**. Marking every pixel needs an ecologist. So in a real project, you might have a handful of labelled images per habitat class. Not thousands. A handful.

That leads to our research question, on the right. If we give a model one, or five, labelled examples of a class, can it find that class in a **new image, from a region it has never seen?**

At the end of the last phase, our plan was: a SAM 2 encoder, prototype-based few-shot learning, the ecological channels NDVI and Shannon entropy, and then transfer to the EU drone data.

[Click.]

## 3 · Recap: results, and what had to change · 1:30

On the left are the numbers you saw last time on LoveDA. The supervised HRNet reached 0.58. Domain adaptation reached 0.35. And SAM 2 with a trained decoder reached 0.47. The 0.85 at the bottom uses the ground truth to label SAM's masks. So it is an upper bound, not a usable model.

On the right is what we learned. And every point forced a change.

[Point to the grey box.]

First: LoveDA is **RGB only**. It has no near-infrared. So NDVI could not even be computed. The ecological half of our plan was impossible on that dataset.

Second: the other aerial benchmark we tried, ISPRS Potsdam, was flown in winter, with no leaves on the trees. Tree NDVI sits around 0.21, the same as bare soil. So NDVI cannot separate vegetation there either.

Third: training a small prompt network on top of a frozen SAM got stuck at 0.06 on new classes.

So we changed two things. We needed a dataset with near-infrared. And we stopped training, and instead prompted frozen foundation models directly.

[Click.]

## 4 · Why we moved to FLAIR · 2:40

This table compares the candidate datasets with our real target, the EU drone data, on the far right.

[Point to the FLAIR column.]

FLAIR is the only one that has all three things we need. It has **near-infrared on every image**. It has twenty-centimetre resolution, which is drone resolution. And it shows **European landscapes**.

It also covers many regions of France, flown on different dates. That matters for two reasons. First, the season becomes something we can study, instead of a fixed handicap, like Potsdam's winter flight. Second, we can take the labelled example from one region and test on another region, or another year. So we measure real transfer, not memory of neighbouring tiles.

And its classes are ecological. Coniferous versus deciduous forest. Brushwood. Vineyard. Not just "tree" versus "low vegetation".

[Click.]

## 5 · The FLAIR dataset · 3:40

FLAIR comes from IGN, the French national mapping agency. About ninety-three thousand image patches, each 512 by 512 pixels at twenty centimetres, from fifty-five regions and dates, with nineteen land-cover classes.

Every patch has red, green, blue and near-infrared, plus a height model. **We deliberately do not use the height.** The EU drone sensor has no height channel. So anything built on height could never transfer to our target.

[Point to the images.]

These are real patches from our evaluation set, with their labels in the official colours.

Two things we found while setting this up. First, the version we downloaded numbers its classes in a different order from the original FLAIR. Without fixing that, deciduous forest would silently have been read as plowed land. Second, we checked the colour channels against the actual files. Our earlier code assumed blue-green-red. The files are red-green-blue. So our previous FLAIR numbers were very likely computed on colour-swapped images. **We no longer use them.**

For today, we evaluate twelve classes on two hundred and fifty patches from ten regions.

[Click.]

## 6 · SAM 1 → SAM 2 → SAM 3 · 4:45

Now the three models. All three are frozen in our work.

**SAM 1**, from 2023. A vision transformer, trained on eleven million images and over a billion masks. You give it a point or a box, and it returns a mask. But it doesn't know what the mask is. We use the base version, with ninety-one million parameters.

**SAM 2**, from 2024. A hierarchical encoder that sees several scales at once, plus a memory for video. We use the small version, forty-six million parameters. That is the same encoder as in our last phase.

And **SAM 3**, from late 2025. This one is a different kind of model. It has about eight hundred and forty million parameters, roughly eighteen times more than our SAM 2. And it doesn't just segment the thing you point at. [PAUSE] **It segments every instance of a concept**, given a text phrase or an example.

[Point to the last row.]

The last row is the one that matters for us. For SAM 1 and SAM 2, we have to build our own few-shot pipeline. SAM 3 has example-based prompting built in.

[Click.]

## 7 · SAM 3: Promptable Concept Segmentation · 5:50

Here is how SAM 3 works, simplified.

[Follow the diagram left to right with your hand.]

The prompt can be a short phrase, like "vineyard". Or it can be an example box drawn on the image. Or both.

The image goes through the Perception Encoder, a backbone shared by an image detector and a video tracker. A fusion encoder combines the image with the prompt. Then a detection decoder proposes every instance. And a **presence head** first asks: "is this concept in the image at all?" Only then does it localise it. That cuts false positives.

It was trained on four million unique concepts, including hard negatives.

But there is one limitation that shapes our whole design. [PAUSE] **The example box must be in the same image you want to segment.** You cannot give SAM 3 a labelled support image and a separate query image.

On the next slide, I'll show how we get around that.

[Click.]

## 8 · How each model does few-shot · 6:55

To make the comparison fair, every model gets **exactly the same test cases**. We call them episodes. One episode is one class, one or five labelled examples, and one new image to segment.

[Point to the top row.]

For SAM 1 and SAM 2, we use prototype prompting. We encode the labelled example and average its features under the mask. That gives us a fingerprint of the class. Then we compare that fingerprint with every location in the new image. The most similar point becomes a positive click, the least similar a negative click, and SAM draws the mask. The pipeline is identical for both models. So any difference comes from the model itself.

[Point to the bottom row and the image.]

For SAM 3, we use a trick from a 2026 paper by Tsai and colleagues. We paste the labelled example on top and the new image below, into one picture. We draw the example box on the top half, run SAM 3, and keep only the bottom half. On the right is a real canvas from our data.

Three hundred episodes, twelve classes, twenty-five each. And the example and the test image **always come from different regions**. One fair caveat: we compare each model with its best available few-shot method, not the bare networks.

[Click.]

## 9 · Experiment 1: upper bounds · 8:15

Before the real task, one diagnostic question. When a model fails, is it the model, or the prompting?

"Oracle" means we cheat on purpose. We take the prompt from the answer itself: a box around the true region, a click inside, and a click outside. That shows how well the model can draw the mask when it is told exactly where to look.

[Point to the first bar.]

With oracle prompts, SAM 2 reaches **0.60**. With its own few-shot prompts, it reaches **0.27**. [PAUSE]

So SAM 2 can draw the mask. That is not the problem. More than half of the performance is lost in **finding** the class across regions. The fingerprint matching does not carry over from one region to another.

The dashed line is our baseline: simply labelling the whole image as the class. It scores 0.29. Keep that line in mind, because it matters on the next slide.

[Click.]

## 10 · Experiment 2: few-shot results · 9:15

This is our main result.

With one labelled example, **SAM 3 reaches 0.334, and SAM 2 reaches 0.266.** The difference is plus 0.068, and the ninety-five percent confidence interval runs from plus 0.02 to plus 0.12. It does not include zero, so the difference is significant.

Now, an uncomfortable number. [PAUSE] **SAM 2's one-shot result is below the baseline.** Simply labelling the entire image as the class, 0.293, beats SAM 2's prompting. SAM 3 is above the baseline. This is exactly why we always show that line. Without it, 0.27 sounds like a result.

With five examples, SAM 2 improves a lot, to 0.345. Averaging five fingerprints makes the matching more stable. SAM 3 with five examples is still running.

Two more columns. SAM 3 returns **no mask at all in thirty-four percent** of the episodes. It is a detector, and on diffuse land cover it often decides the concept isn't there. And SAM 3 is about five times slower: under three seconds per episode on our laptop GPU.

SAM 1 runs on the university cluster and will be added. And all of this is on a small subset, so please read it as preliminary.

[Click.]

## 11 · Per-class IoU · 10:50

Here is the same result, class by class. The black tick on each class is the baseline. A bar below its tick did worse than labelling the whole image.

[Point to vineyard, plowed land and building.]

SAM 3 wins **seven of the twelve classes**. Its wins are big on classes with a clear shape or texture. Vineyard: 0.74 versus 0.44. Plowed land: 0.63 versus 0.35. Buildings: 0.46 versus 0.20.

[Point to coniferous and agricultural land.]

SAM 2 wins the other five, such as coniferous forest and agricultural land. These are large, uniform areas, where a similarity map works well, and where SAM 3 sometimes returns nothing.

If you count the bars below the ticks: SAM 2 falls below the baseline on eight classes, SAM 3 on five. And the diffuse classes, like gardens and brushwood, are hard for everyone.

[Click.]

## 12 · Examples: where the models disagree most · 11:50

These examples are **not hand-picked**. After all runs, a fixed rule selected the five episodes where the models disagree the most.

[Point along one row.]

Each row shows the labelled example with its mask in pink, the new image, the true answer in pink, and then each model's prediction. Green is correct. Red is a false alarm. Blue is missed. The small label tells you that the example and the test image come from different regions.

And the pattern is clear. [PAUSE] **SAM 2 locks onto the wrong region and scores almost zero. SAM 3 finds the whole field, above 0.9.** Look at the vineyard and the plowed land. SAM 3 has recognised the concept, not just copied colours.

[Click.]

## 13 · Examples: where all models succeed or fail · 12:45

The same rule also picked three cases where every model succeeds, and two where every model fails.

[Point to the top rows.]

The successes are large, distinctive areas: coniferous forest, a block of buildings, a meadow.

[Point to the bottom rows.]

The failures are small, scattered targets, a few buildings or a few gardens. Here SAM 2 segments something else, and **SAM 3 returns nothing at all**. That is the thirty-four percent you saw earlier, made visible.

So, to be honest: few-shot segmentation of small, scattered land cover is not solved by any of the three models.

[Click.]

## 14 · Why we continue with SAM 3 · 13:40

So why do we continue with SAM 3? Every number here is computed from our runs.

[Point to each card in turn.]

**Accuracy**: plus 0.068 over SAM 2 with one example, and significant.

**Classes**: best on seven of twelve.

**Cost**: about five times slower than SAM 2. But that is still around a second per episode on a sixteen-gigabyte GPU. So it is feasible on our cluster.

And it accepts examples and text out of the box, so we no longer need a hand-built prompting pipeline.

The trade-off is the empty masks. And here is a simple fix, with no labels and no tuning. [PAUSE] **When SAM 3 returns nothing, we use SAM 2's answer.** That combination reaches **0.371**, better than either model alone. The two models fail in different ways, so they complement each other.

[Click.]

## 15 · SAM 3 + NDVI: segment, then measure · 14:55

Now, near-infrared. We used it in two ways. The first needs no training at all.

[Follow the top row, then the bottom row.]

**SAM 3 does the segmentation from RGB alone, untouched.** Near-infrared only comes in afterwards. We compute NDVI from near-infrared and red, and a local Shannon entropy of NDVI over a window of one point four metres, as a measure of how mixed the vegetation is. Then we summarise both inside each predicted segment: how green it is, how much area it covers, and how mixed it is.

That gives an ecologist a habitat map with vigour and diversity values traceable to each segment.

We also tested an optional filter: keep only predicted pixels whose NDVI looks like the labelled example's.

[Click.]

## 16 · Results: ecology from the segments · 15:45

[Point to the table.]

Measured on SAM 3's segments, NDVI is off by only 0.06 on average, with a correlation of **0.86** to the true segments. Covered area is harder, off by about twenty-one percentage points, because area depends directly on mask quality.

But, an important negative result. [PAUSE] **The NDVI filter hurts.** SAM 3 drops from 0.33 to 0.21. An NDVI range from one region or season does not fit another.

So near-infrared as a hand-made rule fails. [PAUSE] Which raises the obvious question: **what if the model learns to use near-infrared itself?**

[Click.]

## 17 · Experiment 3: training SAM 3 with NIR · 16:30

That is our third experiment. Here, for the first time, we **train**.

[Follow the diagram from left to right.]

We give SAM 3 four bands instead of three: red, green, blue, and near-infrared. The very first layer of SAM 3 is built for three colours, so we widen it to four. And the near-infrared filters start at **zero**. That means at the start of training, the model is exactly the original SAM 3. Training alone decides how much near-infrared to use.

We keep the big encoder frozen and train small LoRA adapters inside it, plus the parts that turn a class name into a map. That is about sixteen million of eight hundred and forty million parameters. For each of the twelve classes, the model gets the class name as a prompt and returns a map; the class with the highest score wins each pixel.

[Point to the first bullet.]

The key to this experiment is the control. [PAUSE] We train the **same model twice**: once with RGB only, once with RGB plus near-infrared. Same images, same order, same augmentations, same seeds. **The near-infrared channel is the only difference.**

We used the official FLAIR-HUB split: about thirty-eight thousand training patches from twenty-one regions, and **fifty thousand seven hundred test patches from sixteen regions the model has never seen**. Three seeds per arm, on the shared GPUs of our cluster.

[Click.]

## 18 · Experiment 3: does NIR help? · 17:50

[Point to the table.]

First, training itself. SAM 3 without training, given only the class names, gets 0.18. After fine-tuning on RGB, **0.62**, on regions it has never seen. That is the big step, and far above the few-shot numbers from before.

[Point to the box.]

Now our question. Adding near-infrared gives **plus 0.0095**. [PAUSE] About one point of mIoU. It is small. But it is real: the confidence interval runs from plus 0.008 to plus 0.011, and **every one of the three seeds** shows a gain.

[Point to the last bullet.]

And it is stable. We first trained for four thousand steps, then three times longer. Both models improved by about one and a half points, and the near-infrared gain stayed **exactly the same**: plus 0.0095 both times.

[Point to the chart on the right.]

And it lands exactly where ecology says it should. **Coniferous forest gains almost six points**, deciduous and grassland about two each. In total, seven of twelve classes improve significantly. Buildings and roads barely move; colour already separates them. In the shorter runs, gardens and lawns lost about a point; with longer training that loss disappeared.

[Click.]

## 19 · Experiment 3: where NIR helps and hurts · 19:00

Two examples, chosen automatically: the test patch where near-infrared helped most, and where it hurt most.

[Point to the top row.]

At the top, a meadow with trees. The RGB model calls the whole meadow cropland. With near-infrared, it correctly sees grassland. In the false-colour image you can see why: the meadow reflects near-infrared differently.

[Point to the bottom row.]

At the bottom, the opposite. A plowed field next to a crop field. RGB gets it right; with near-infrared, the model merges the plowed field into the crop.

Single patches always exaggerate. [PAUSE] The evidence is the table on the previous slide.

[Click.]

## 20 · Conclusion · 19:40

Let me bring it together.

One. On identical test cases from different regions, without training, **SAM 3 beats SAM 2 significantly, 0.33 against 0.27**.

Two. Near-infrared after segmentation is good for **measuring** vegetation, but as a hard filter it hurts.

Three. **Training changes everything**: SAM 3 goes from 0.18 to 0.62 on unseen regions. And near-infrared **inside** the trained model adds a small but consistent gain, mostly for vegetation, with coniferous forest gaining the most.

And four. Everything uses only RGB and near-infrared, no height. [PAUSE] So it transfers directly to the EU drone sensor, which has near-infrared.

Our limits: the few-shot comparison ran on a small subset, and SAM 1 is still pending.

[Click.]

## 21 · Future work: the EU drone data · 20:30

For the last phase, four steps.

[Point to each box.]

Bring in the EU drone data with the same colour-channel check. Run few-shot SAM 3 where we have only a handful of labels. Then fine-tune our FLAIR-trained near-infrared model with the few EU labels we have. And measure vegetation and diversity per habitat and per country.

The open question is whether a model trained in France carries over to habitats in other countries. And our biggest risk is simple: **we still do not have access to the EU drone data.**

[Click.]

## 22 · References · 21:15

These are our main sources: the SAM papers, the few-shot method by Tsai and colleagues, VegSAM, LoRA, and the FLAIR datasets from IGN.

[Step forward, smile.]

Thank you for your attention. We are happy to take your questions.

## Answers to likely questions (say these if asked)

**"Is the comparison fair? The models use different prompting."**
"Each model uses its best available few-shot method, on exactly the same test cases. And SAM 2's oracle score of 0.60 shows that its mask decoder is not being held back. The gap is in how each model uses the labelled example."

**"Why is the baseline so high?"**
"Many land-cover classes cover a large part of the image. So labelling everything already scores that class's coverage. That's why we cap coverage at ninety percent, and why we show the baseline on every chart."

**"Are three hundred episodes enough?"**
"For the overall comparison, yes: the confidence interval for SAM 3 minus SAM 2 does not include zero. For single classes, it is noisier, especially vineyard and plowed land, which have only about ten different test images."

**"Why didn't you fine-tune the few-shot models?"**
"In Experiments 1 and 2 we kept everything frozen, so the only thing changing between results is the model. Fine-tuning is Experiment 3, and there the RGB-only control is trained identically, so the near-infrared effect is isolated."

**"Is plus 0.0095 meaningful? It's tiny."**
"Overall it is about one point, yes. But the confidence interval excludes zero, all three seeds agree, and the gain is concentrated where it matters ecologically: coniferous forest gains almost six points. And it is the same gain after four thousand and after twelve thousand training steps. For a biodiversity map, that is the class that counts."

**"Why not an NDVI adapter like VegSAM?"**
"We gave the model the raw near-infrared band and let training decide how to use it. NDVI is one fixed formula from that band; the network can learn it, or something better. An adapter is a possible next comparison."

**"Is the fine-tuned model still few-shot?"**
"No, and we don't claim it is. It uses thirty-eight thousand labelled patches. It answers a different question: whether near-infrared helps SAM 3 once it is trained. For the EU data, we would start from this model and adapt it with the few labels we have there."

**"What happened to your earlier FLAIR results?"**
"They assumed the wrong colour channel order, so they were very likely computed on colour-swapped images. We checked the files and retired those numbers."

**"Why does SAM 3 return nothing so often?"**
"It is a detector with a presence check. On diffuse land cover, it often decides the concept is not there, at its default confidence threshold. Lowering that threshold is a planned test, chosen on separate regions, never on the test cases."
