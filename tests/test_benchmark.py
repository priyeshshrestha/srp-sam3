"""Tests for the SAM 1/2/3 comparison pipeline. No data, no GPU, no model weights.

    python tests/test_benchmark.py
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import datasets.flair_rgb as flair_rgb  # noqa: E402
from datasets.flair_rgb import optical_to_rgb_uint8  # noqa: E402
from evaluation.episodes import make_episodes, save_episodes  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    baseline_record, class_miou, confusion, episode_iou, fb_iou, paired_bootstrap,
)
from evaluation.report_lib import load_pred, select_figure_episodes  # noqa: E402
from models.persam import persam_predict  # noqa: E402
from models.sam3_fss import support_box  # noqa: E402

SIZE = 64


def test_band_order():
    print("[1/13] band_order decides which channel is red...")
    raw = np.zeros((5, 4, 4), dtype=np.float32)
    raw[0], raw[1], raw[2] = 200, 100, 50                     # ch0=200, ch1=100, ch2=50
    rgb = optical_to_rgb_uint8(raw, "rgb")
    bgr = optical_to_rgb_uint8(raw, "bgr")
    assert rgb.shape == (4, 4, 3) and rgb.dtype == np.uint8
    assert tuple(rgb[0, 0]) == (200, 100, 50)
    assert tuple(bgr[0, 0]) == (50, 100, 200), "bgr must reverse the first three channels"
    for bad in ("", "RGB", "auto", None):
        try:
            optical_to_rgb_uint8(raw, bad)
            raise AssertionError(f"band_order {bad!r} should have raised")
        except ValueError:
            pass
    # Channels 3 and 4 (NIR, nDSM) must never leak into the image.
    raw[3], raw[4] = 255, 30
    assert tuple(optical_to_rgb_uint8(raw, "rgb")[0, 0]) == (200, 100, 50)
    print("      --> Passed!")


def _counts():
    # 6 patches over 3 domains; class 1 big in all, class 2 only in D1, class 3 fills the frame.
    full = 512 * 512
    counts, domains = {}, {}
    for d in ("D001_2020", "D002_2020", "D003_2021"):
        for k in range(2):
            p = f"{d}/p{k}.tif"
            counts[p] = {1: 20000, 2: 20000 if d == "D001_2020" else 0, 3: full, 4: 10}
            domains[p] = d
    return counts, domains


def test_episode_generation():
    print("[2/13] Episodes are deterministic, cross-domain and respect coverage filters...")
    counts, domains = _counts()
    a = make_episodes(counts, domains, [1, 2, 3, 4], 10, seed=7, domain_mode="cross")
    b = make_episodes(counts, domains, [1, 2, 3, 4], 10, seed=7, domain_mode="cross")
    assert a == b, "same seed must give identical episodes"
    assert a != make_episodes(counts, domains, [1, 2, 3, 4], 10, seed=8, domain_mode="cross")

    eps = a["episodes"]
    assert all(e["support"] != e["query"] for e in eps)
    assert all(e["support_domain"] != e["query_domain"] for e in eps), "cross mode leaked a same-domain pair"
    assert {e["class_id"] for e in eps} == {1}, "only class 1 is eligible with cross-domain pairs"
    assert 3 in a["skipped"] and 4 in a["skipped"], "full-frame and speck classes must be skipped"
    assert 2 in a["skipped"], "class 2 exists in one domain only -> no cross-domain pair"

    same = make_episodes(counts, domains, [1, 2], 10, seed=1, domain_mode="same")["episodes"]
    assert same and all(e["support_domain"] == e["query_domain"] for e in same)
    assert {e["class_id"] for e in same} == {1, 2}
    assert [e["id"] for e in eps] == list(range(len(eps))), "ids must be 0..N-1"
    print("      --> Passed!")


def _bin(rect):
    m = np.zeros((SIZE, SIZE), dtype=bool)
    y0, y1, x0, x1 = rect
    m[y0:y1, x0:x1] = True
    return m


def test_metrics():
    print("[3/13] IoU, accumulation, FB-IoU, baselines and paired bootstrap...")
    gt = _bin((0, 32, 0, 32))
    perfect, empty = confusion(gt, gt), confusion(np.zeros_like(gt), gt)
    assert episode_iou(perfect) == 1.0 and episode_iou(empty) == 0.0
    half = confusion(_bin((0, 32, 0, 16)), gt)
    assert np.isclose(episode_iou(half), 0.5)

    # All-foreground IoU equals the class coverage -- the degenerate-score guard.
    allfg = baseline_record(gt, "all_fg")
    assert np.isclose(episode_iou(allfg), gt.mean())
    assert episode_iou(baseline_record(gt, "empty")) == 0.0

    # Accumulated and per-episode means differ when episode sizes differ.
    big = {"class_id": 1, "fg_inter": 900, "fg_union": 1000}
    small = {"class_id": 1, "fg_inter": 0, "fg_union": 10}
    acc, _ = class_miou([big, small], accumulate=True)
    epi, _ = class_miou([big, small], accumulate=False)
    assert np.isclose(acc, 900 / 1010) and np.isclose(epi, 0.45)

    assert 0.0 <= fb_iou([perfect, half]) <= 1.0
    assert np.isclose(fb_iou([perfect]), 1.0)

    # Bootstrap: identical models -> difference exactly 0 and not significant.
    recs = {i: {"class_id": i % 3, "fg_inter": 50 + i, "fg_union": 100 + i} for i in range(60)}
    same = paired_bootstrap(recs, recs, n_boot=200)
    assert same["diff"] == 0.0 and not same["significant"]
    # A model that is clearly better on every episode must be flagged significant.
    worse = {i: {**r, "fg_inter": r["fg_inter"] // 2} for i, r in recs.items()}
    res = paired_bootstrap(recs, worse, n_boot=200)
    assert res["diff"] > 0 and res["ci_lo"] > 0 and res["significant"]
    assert res["n_episodes"] == 60
    print("      --> Passed!")


class FakeAdapter:
    """Left half of the image is red, right half is blue; features are just downsampled colour."""
    device = "cpu"

    def embed(self, rgb):
        t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
        feat = torch.nn.functional.adaptive_avg_pool2d(t, (16, 16))
        return torch.cat([feat, torch.ones(1, 1, 16, 16) * 0.1], dim=1)

    def decode(self, points, labels, box):
        red, blue = np.zeros((SIZE, SIZE), bool), np.zeros((SIZE, SIZE), bool)
        red[:, : SIZE // 2], blue[:, SIZE // 2:] = True, True
        self.last = (points, labels, box)
        # The model's OWN confidence prefers the wrong (blue) mask; prototype matching must override.
        return np.stack([red, blue, red | blue]), np.array([0.5, 0.9, 0.1])


def test_persam_frontend():
    print("[4/13] PerSAM front end: prototype drives prompts and candidate choice...")
    img = np.zeros((SIZE, SIZE, 3), np.uint8)
    img[:, : SIZE // 2, 0] = 255      # red half
    img[:, SIZE // 2:, 2] = 255       # blue half
    support_mask = np.zeros((SIZE, SIZE), bool)
    support_mask[:, : SIZE // 2] = True                        # the target is 'red'

    ad = FakeAdapter()
    pred, diag = persam_predict(ad, img, support_mask, img)
    assert pred.dtype == bool and pred.shape == (SIZE, SIZE)
    assert pred[:, : SIZE // 2].all() and not pred[:, SIZE // 2:].any(), "chose the wrong candidate"
    assert diag["candidate"] == 0
    pts, labels, box = ad.last
    assert pts[0][0] < SIZE // 2 and pts[1][0] >= SIZE // 2, "positive point must be on red, negative on blue"
    assert labels.tolist() == [1, 0]
    assert box is not None and box[2] < SIZE // 2 + 2, "box should hug the red half"

    # Non-square input is rejected instead of being silently mis-registered.
    try:
        persam_predict(ad, img[:, :40], support_mask[:, :40], img[:, :40])
        raise AssertionError("non-square input should have raised")
    except ValueError:
        pass
    # An empty support mask must raise, not produce a meaningless prototype.
    try:
        persam_predict(ad, img, np.zeros((SIZE, SIZE), bool), img)
        raise AssertionError("empty support should have raised")
    except ValueError:
        pass
    print("      --> Passed!")


def test_sam3_support_box():
    print("[5/13] SAM 3 box: largest component, whole-class fallback is flagged...")
    m = np.zeros((512, 512), bool)
    m[50:110, 50:110] = True          # 3600 px
    m[300:450, 300:400] = True        # 15000 px -> largest
    box, fb = support_box(m)
    assert box == (300, 300, 100, 150) and fb is False
    tiny = np.zeros((512, 512), bool)
    tiny[10:20, 10:20] = True          # below min area -> no component
    box, fb = support_box(tiny)
    assert box == (10, 10, 10, 10) and fb is True
    huge = np.zeros((512, 512), bool)
    huge[:, :400] = True               # one component over 60% of the patch
    box, fb = support_box(huge)
    assert fb is True and box == (0, 0, 400, 512)
    print("      --> Passed!")


def test_figure_selection():
    print("[6/13] Figure episodes are chosen by rule, deterministically...")
    ious = {0: {"a": 0.9, "b": 0.1}, 1: {"a": 0.8, "b": 0.75}, 2: {"a": 0.0, "b": 0.05},
            3: {"a": 0.5, "b": 0.45}, 4: {"a": 0.02, "b": 0.0}, 5: {"a": 0.95, "b": 0.92}}
    class_of = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
    gt_frac = {0: 0.3, 1: 0.3, 2: 0.3, 3: 0.3, 4: 0.001, 5: 0.3}   # episode 4 is a tiny target
    picked = select_figure_episodes(ious, class_of, gt_frac, n_total=4, n_disagree=1, n_succeed=2)
    ids = [p["id"] for p in picked]
    assert ids[0] == 0, "largest disagreement goes first"
    assert 5 in ids and 1 in ids, "highest worst-model IoU episodes are the 'succeed' picks"
    assert 4 not in ids, "a tiny target must not be shown as a model failure"
    assert len(ids) == len(set(ids)) == 4 and 2 in ids
    assert picked == select_figure_episodes(ious, class_of, gt_frac, n_total=4, n_disagree=1, n_succeed=2)
    print("      --> Passed!")


def test_end_to_end_with_fake_model():
    print("[7/13] run_model -> report works end to end (fake model, fake TIFFs)...")
    import scripts.run_model as run_model
    import scripts.report as report
    real_build = run_model.build_predictor
    root = tempfile.mkdtemp()
    try:
        rng = np.random.default_rng(0)
        patches = {}
        for d in ("D001_2020", "D002_2021"):
            for k in range(3):
                rel = os.path.join(d, "aerial", "Z1_AA", f"IMG_{d[1:4]}{k:03d}.tif")
                full = os.path.join(root, rel)
                os.makedirs(os.path.dirname(full), exist_ok=True)
                os.makedirs(os.path.join(root, d, "labels", "Z1_AA"), exist_ok=True)
                msk = os.path.join(root, d, "labels", "Z1_AA", os.path.basename(rel).replace("IMG", "MSK"))
                open(full, "wb").close()
                open(msk, "wb").close()
                img = rng.integers(0, 255, (5, SIZE, SIZE)).astype(np.float32)
                mask = np.full((1, SIZE, SIZE), 7, np.uint8)
                mask[:, :, : SIZE // 2] = 1 + (k % 2)          # class 1 or 2 on the left half
                patches[full], patches[msk] = img, mask

        def fake_read(path):
            return patches[path]
        import datasets.flair as flair_mod
        flair_mod._read_tif = fake_read
        flair_rgb._read_tif = fake_read

        rels = flair_rgb.discover_patches(root)
        assert len(rels) == 6
        doms = flair_rgb.domains_of(rels)
        counts = flair_rgb.build_class_counts(root, rels)
        payload = make_episodes(counts, doms, [1, 2], 4, seed=3, patch_pixels=SIZE * SIZE)
        assert len(payload["episodes"]) == 8, payload["skipped"]
        ep_path = os.path.join(root, "episodes.json")
        save_episodes(ep_path, payload, {"band_order": "rgb", "domain_mode": "cross",
                                         "min_frac": 0.02, "max_frac": 0.9, "seed": 3})

        def make_fake(offset):
            def build(model, variant, checkpoint, device, prompt="fewshot"):
                def predict(supports, q_rgb, q_gt, class_id):
                    pred = np.zeros(q_gt.shape, bool)
                    pred[:, : SIZE // 2 - offset] = True       # model A is exact, B under-segments
                    return pred, {"box_fallback": False}
                return predict
            return build

        out_dirs = []
        for name, off in (("fakeA", 0), ("fakeB", 10)):
            run_model.build_predictor = make_fake(off)
            out = os.path.join(root, "runs", name)
            sys.argv = ["run_model", "--model", "sam2", "--root", root, "--episodes", ep_path,
                        "--out", out, "--device", "cpu"]
            run_model.main()
            out_dirs.append(out)
            lines = [json.loads(line) for line in open(os.path.join(out, "results.jsonl"))]
            assert len(lines) == 8 and all("fg_inter" in r for r in lines)

        # Resume: a second invocation must not duplicate results.
        run_model.main()
        assert sum(1 for _ in open(os.path.join(out_dirs[0], "results.jsonl"))) == 8

        pred = load_pred(out_dirs[0], 0)
        assert pred.shape == (SIZE, SIZE) and pred[:, : SIZE // 2].all() and not pred[:, SIZE // 2:].any()

        rep = os.path.join(root, "report")
        sys.argv = ["report", "--root", root, "--episodes", ep_path, "--runs", *out_dirs,
                    "--out", rep, "--n_figures", "4", "--n_boot", "50"]
        report.main()
        summary = open(os.path.join(rep, "summary.md")).read()
        assert "baseline:all_fg" in summary and "fakeA" in summary and "Paired bootstrap" in summary
        assert os.path.exists(os.path.join(rep, "per_class.csv"))
        assert os.path.exists(os.path.join(rep, "per_class_iou.png"))
        figs = [f for f in os.listdir(os.path.join(rep, "figures")) if f.endswith(".png")]
        assert len(figs) == 4, figs
        # fakeA is exact (IoU 1.0) and fakeB under-segments, so A must rank above B.
        assert summary.index("| fakeA |") < summary.index("| fakeB |")
        print("      --> Passed!")
    finally:
        run_model.build_predictor = real_build
        shutil.rmtree(root)


def test_existing_loader_still_untouched():
    print("[8/13] The original loader module is unchanged and importable...")
    from datasets.flair import FLAIRDataset, FLAIR_CLASSES
    assert len(FLAIR_CLASSES) == 19 and FLAIRDataset.ELEVATION_FREE
    print("      --> Passed!")


def test_flair_hub_labels_and_layout():
    print("[9/13] FLAIR-HUB: COSIA labels remapped to FLAIR #1 ids, file pairing, domain names...")
    from datasets.flair import FLAIR_CLASSES, domain_of, mask_path_for
    from datasets.flair_rgb import HUB_TO_FLAIR1
    # The table must be a bijection 0..18 -> 1..19, and spot-check the classes whose order differs.
    assert sorted(HUB_TO_FLAIR1) == list(range(19)) and sorted(HUB_TO_FLAIR1.values()) == list(range(1, 20))
    names = {0: "building", 4: "pervious_surface", 8: "herbaceous_vegetation", 11: "vineyard",
             12: "deciduous", 13: "coniferous", 14: "brushwood", 6: "water"}
    for hub, name in names.items():
        assert FLAIR_CLASSES[HUB_TO_FLAIR1[hub]] == name, (hub, FLAIR_CLASSES[HUB_TO_FLAIR1[hub]], name)

    root = tempfile.mkdtemp()
    try:
        rel_img = "D069-2020_AERIAL_RGBI/UA-S1-5/D069-2020_AERIAL_RGBI_UA-S1-5_3-4.tif"
        rel_lab = rel_img.replace("_AERIAL_RGBI", "_AERIAL_LABEL-COSIA")
        for rel in (rel_img, rel_lab):
            os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
            open(os.path.join(root, rel), "wb").close()
        assert mask_path_for(os.path.join(root, rel_img)) == os.path.join(root, rel_lab)
        assert domain_of(rel_img) == "D069_2020", domain_of(rel_img)
        assert domain_of("x/D041_2021/aerial/Z1_AA/IMG_1.tif") == "D041_2021"

        hub = np.array([[12, 13, 0, 4]], dtype=np.uint8)          # deciduous, coniferous, building, pervious
        orig = flair_rgb._read_tif
        flair_rgb._read_tif = lambda p: hub[None]
        try:
            got = flair_rgb.read_label(os.path.join(root, rel_lab))
            assert got.tolist() == [[7, 6, 1, 2]], got.tolist()
            # A FLAIR #1 mask must pass through untouched.
            assert flair_rgb.read_label("x/MSK_1.tif").tolist() == hub.astype(np.int64).tolist()
            flair_rgb._read_tif = lambda p: np.array([[[19]]], np.uint8)
            try:
                flair_rgb.read_label(os.path.join(root, rel_lab))
                raise AssertionError("HUB value 19 must be rejected")
            except ValueError:
                pass
        finally:
            flair_rgb._read_tif = orig
        assert flair_rgb.discover_patches(root) == [rel_img]
    finally:
        shutil.rmtree(root)
    print("      --> Passed!")


def test_k_shot_episodes_and_prototypes():
    print("[10/13] K supports per episode; 1-shot support is the first 5-shot support...")
    counts, domains = _counts()
    for k in range(2):                                   # more patches so 3 supports fit cross-domain
        for d in ("D004_2020", "D005_2021"):
            p = f"{d}/q{k}.tif"
            counts[p] = {1: 20000}
            domains[p] = d
    one = make_episodes(counts, domains, [1], 6, seed=2, k_shot_max=1)["episodes"]
    three = make_episodes(counts, domains, [1], 6, seed=2, k_shot_max=3)["episodes"]
    assert all(len(e["supports"]) == 3 and e["supports"][0] == e["support"] for e in three)
    assert all(len(set(e["supports"])) == 3 and e["query"] not in e["supports"] for e in three)
    assert all(domains[s] != domains[e["query"]] for e in three for s in e["supports"])
    from evaluation.episodes import episodes_hash
    changed = [dict(e) for e in three]
    changed[0]["supports"] = list(reversed(changed[0]["supports"]))
    assert episodes_hash(three) != episodes_hash(changed), "hash must cover every support"
    assert one and episodes_hash(one) != episodes_hash(three)

    # Averaged prototype: two supports whose masks cover red and blue -> query red half still wins
    # when the red support dominates, and a list of one equals the single-image call.
    img = np.zeros((SIZE, SIZE, 3), np.uint8)
    img[:, : SIZE // 2, 0] = 255
    img[:, SIZE // 2:, 2] = 255
    red = np.zeros((SIZE, SIZE), bool)
    red[:, : SIZE // 2] = True
    ad = FakeAdapter()
    single, _ = persam_predict(ad, img, red, img)
    listed, d = persam_predict(ad, [img, img], [red, red], img)
    assert (single == listed).all() and d["k_shot"] == 2

    from models.persam import oracle_prompts, persam_oracle
    gt = np.zeros((SIZE, SIZE), bool)
    gt[10:30, 5:25] = True
    gt[50:52, 50:52] = True                              # a small second component
    pts, labels, box = oracle_prompts(gt)
    assert box.tolist() == [5, 10, 24, 29], box.tolist()
    assert labels.tolist() == [1, 0] and gt[int(pts[0][1]), int(pts[0][0])], "positive point must be inside"
    assert not gt[int(pts[1][1]), int(pts[1][0])], "negative point must be outside"
    pred, d = persam_oracle(ad, img, gt)
    assert pred.shape == gt.shape and d["oracle"]
    print("      --> Passed!")


def test_majority_vote():
    print("[11/13] SAM 3 K-shot majority vote...")
    from models.sam3_fss import CLASS_PHRASES, majority_vote
    a = np.array([[1, 1, 0, 0]], bool)
    b = np.array([[1, 0, 1, 0]], bool)
    c = np.array([[1, 0, 0, 0]], bool)
    assert majority_vote([a, b, c]).tolist() == [[True, False, False, False]]
    assert majority_vote([a]).tolist() == a.tolist()
    assert set(CLASS_PHRASES) == set(range(1, 13)), "one phrase per evaluated class"
    print("      --> Passed!")


def test_run_model_arms():
    print("[12/13] run_model: 5-shot needs 5 stored supports; oracle/text arms run...")
    import scripts.run_model as run_model
    assert run_model.arm_name("fewshot", 5) == "5shot" and run_model.arm_name("oracle", 1) == "oracle"
    try:
        run_model.build_predictor("sam2", "small", "x", "cpu", prompt="text")
        raise AssertionError("text prompts must be refused for SAM 2")
    except ValueError:
        pass
    print("      --> Passed!")


def test_ndvi_indicators():
    print("[13/13] NDVI: red band follows band order, support-derived gate...")
    import scripts.ndvi_indicators as ni
    raw = np.zeros((4, 2, 2), np.float32)
    raw[0], raw[2], raw[3] = 50, 150, 200              # ch0=50, ch2=150, NIR=200
    orig = ni._read_tif
    ni._read_tif = lambda p: raw
    try:
        rgb = ni.ndvi_of("x.tif", "rgb")               # red = ch0 -> (200-50)/(250) = 0.6
        bgr = ni.ndvi_of("x.tif", "bgr")               # red = ch2 -> (200-150)/(350) = 0.143
    finally:
        ni._read_tif = orig
    assert np.allclose(rgb, 0.6, atol=1e-4) and np.allclose(bgr, 50 / 350, atol=1e-4)

    ndvi_s = np.linspace(-0.2, 0.8, 100).reshape(10, 10)
    lo, hi = ni.support_band(ndvi_s, np.ones((10, 10), bool))
    assert lo < hi and -0.2 <= lo < -0.15 and 0.75 < hi <= 0.8
    ndvi_q = np.array([[0.5, -0.5], [0.95, 0.1]])
    pred = np.ones((2, 2), bool)
    assert ni.gate(pred, ndvi_q, (0.0, 0.9)).tolist() == [[True, False], [False, True]]
    assert not ni.gate(np.zeros((2, 2), bool), ndvi_q, (0.0, 0.9)).any(), "gate may only remove pixels"
    assert np.isnan(ni.pearson([1, 2], [1, 2]))
    print("      --> Passed!")


if __name__ == "__main__":
    print("\n--- SAM comparison pipeline tests (no data / GPU / weights) ---")
    test_band_order()
    test_episode_generation()
    test_metrics()
    test_persam_frontend()
    test_sam3_support_box()
    test_figure_selection()
    test_end_to_end_with_fake_model()
    test_existing_loader_still_untouched()
    test_flair_hub_labels_and_layout()
    test_k_shot_episodes_and_prototypes()
    test_majority_vote()
    test_run_model_arms()
    test_ndvi_indicators()
    print("\nAll pipeline tests passed.\n")
