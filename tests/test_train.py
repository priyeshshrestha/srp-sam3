"""Tests for the training experiment (SAM 3, RGB vs RGB+NIR). No data, GPU or weights needed.

    python tests/test_train.py
"""
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train.data import make_split, sample_prompts, split_hash
from train.data import test_subset as subset_of_test  # noqa: E402
from train.metrics import confusion, iou_per_class, miou, paired_bootstrap_delta  # noqa: E402
from train.sam3_rgbi import (  # noqa: E402
    CLASSES, LoRALinear, TinySeg, add_lora, load_trainable, trainable_state, widen_conv,
)


def test_widen_conv_is_exact_noop_with_zero_nir():
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 8, 14, 14, bias=False)
    wide = widen_conv(conv, 4)
    x = torch.randn(2, 3, 56, 56)
    nir = torch.randn(2, 1, 56, 56)
    assert wide.weight.shape == (8, 4, 14, 14)
    assert torch.equal(wide(torch.cat([x, nir], 1)), conv(x))
    out = wide(torch.cat([x, nir], 1)).sum()
    out.backward()
    assert wide.weight.grad[:, 3].abs().sum() > 0, "zero-initialised NIR filters must still get gradient"


def test_lora_is_exact_noop_at_init_and_trains():
    torch.manual_seed(0)
    block = nn.Sequential(nn.Linear(16, 16), nn.ReLU())
    block.q_proj = nn.Linear(16, 16)
    x = torch.randn(4, 16)
    ref = block.q_proj(x)
    assert add_lora(block, names=("q_proj",), rank=4) == 1
    assert isinstance(block.q_proj, LoRALinear)
    assert torch.allclose(block.q_proj(x), ref)
    block.q_proj(x).sum().backward()
    assert block.q_proj.lora_b.grad.abs().sum() > 0


def test_split_holds_out_whole_domains():
    rels = [f"D{d:03d}_2021/aerial/Z1/IMG_{d}{k}.tif" for d in range(1, 11) for k in range(5)]
    s = make_split(rels, mode="auto", test_frac=0.2, seed=0)
    assert s["mode"] == "domain" and not s["domain_overlap"]
    assert len(s["test_domains"]) == 2 and len(s["test"]) == 10
    assert make_split(rels, mode="domain", seed=0)["sha"] == s["sha"], "split must be deterministic"
    assert split_hash(s) == s["sha"]
    off = ["train/D001_2021/IMG_1.tif", "train/D002_2021/IMG_2.tif", "flair_1_aerial_test/D003_2021/IMG_3.tif"]
    o = make_split(off, mode="auto")
    assert o["mode"] == "official" and o["test"] == ["flair_1_aerial_test/D003_2021/IMG_3.tif"]
    sub = subset_of_test(s, 4)
    assert len(sub) == 4 and sub == subset_of_test(s, 4)


def test_official_hub_csv_split():
    tmp = tempfile.mkdtemp()
    hdr = "patch_id;AERIAL_LABEL-COSIA;AERIAL_RGBI\n"

    def row(dom, k):
        return f"{dom}_AA-S1-1_1-{k};x;../{dom}_AERIAL_RGBI/AA-S1-1/{dom}_AERIAL_RGBI_AA-S1-1_1-{k}.tif\n"
    with open(os.path.join(tmp, "FLAIR-HUB_TRAIN.csv"), "w") as fh:
        fh.write(hdr + "".join(row("D001-2020", k) for k in range(3)) + row("D009-2020", 0))
    with open(os.path.join(tmp, "FLAIR-HUB_TEST.csv"), "w") as fh:
        fh.write(hdr + "".join(row("D002-2021", k) for k in range(2)))
    with open(os.path.join(tmp, "FLAIR-HUB_VALID.csv"), "w") as fh:
        fh.write(hdr + row("D001-2020", 7))
    on_disk = [f"{d}_AERIAL_RGBI/AA-S1-1/{d}_AERIAL_RGBI_AA-S1-1_1-{k}.tif"
               for d, ks in (("D001-2020", (0, 1, 2, 7)), ("D002-2021", (0, 1))) for k in ks]
    from datasets.flair import domain_of
    assert domain_of("D054057-2018_AERIAL_RGBI/AA-S1-1/D054057-2018_AERIAL_RGBI_AA-S1-1_1-1.tif") == "D054057_2018"
    s = make_split(on_disk, mode="auto", csv_dir=tmp)
    assert s["mode"] == "hub_csv", s["mode"]
    assert len(s["train"]) == 3 and len(s["test"]) == 2, "D009 is not on disk; the VALID patch is unused"
    assert s["n_unassigned"] == 1 and s["test_domains"] == ["D002_2021"]
    assert make_split(on_disk, mode="auto", csv_dir=None)["mode"] == "domain", "no CSVs -> domain split"

    from scripts.download_flair_hub import select_domains
    per = {"train": {"A": 1, "B": 1, "C": 1, "T": 1}, "test": {"T": 1}}
    test, train, total = select_domains(per, {"A": 4.0, "B": 4.0, "C": 1.0, "T": 3.0}, budget_gb=8.0)
    assert test == ["T"] and "T" not in train and total <= 8.0 and len(train) >= 1
    try:
        select_domains(per, {"A": 1.0, "T": 9.0}, budget_gb=8.0)
        raise AssertionError("a budget smaller than the test set must be refused")
    except ValueError:
        pass


def test_nir_weight_report():
    import scripts.nir_diagnostic as nd
    from train.sam3_rgbi import PATCH_EMBED
    tmp = tempfile.mkdtemp()
    torch.manual_seed(0)
    w0 = torch.randn(8, 3, 14, 14)
    w = torch.cat([w0 * 1.1, 0.5 * w0[:, :1]], 1)               # RGB scaled by 1.1, NIR = half of R
    os.makedirs(os.path.join(tmp, "rgbi_s0"))
    torch.save({"trainable": {f"model.{PATCH_EMBED}.weight": w}}, os.path.join(tmp, "rgbi_s0", "final.pt"))
    orig = nd.pretrained_patch_embed
    nd.pretrained_patch_embed = lambda: w0
    try:
        r = nd.weight_report({"rgbi_s0": os.path.join(tmp, "rgbi_s0")})["rgbi_s0"]
    finally:
        nd.pretrained_patch_embed = orig
    assert abs(r["rgb_rel_change"] - 0.1) < 1e-5
    assert abs(r["norm_NIR"] - 0.5 * float(w0[:, 0].norm())) < 1e-4
    assert 0 < r["nir_over_mean_rgb"] < 1


def test_prompt_sampling():
    y = np.zeros((100, 100), np.int64) + 6
    y[:, :30] = 1
    rng = np.random.default_rng(0)
    for _ in range(20):
        cls = sample_prompts(y, 3, rng)
        assert len(cls) == 3 and len(set(cls.tolist())) == 3
        present = {c for c in cls.tolist() if CLASSES[c] in (1, 6)}
        assert len(present) == 2, "with 2 classes present and 3 prompts: 2 present + 1 absent"
    all_present = np.repeat(np.arange(1, 13), 834)[:10000].reshape(100, 100)
    assert len(sample_prompts(all_present, 3, rng)) == 3


def test_metrics():
    gt = np.array([[1, 1, 2, 2], [1, 1, 2, 13]])
    pred = np.array([[1, 1, 1, 2], [1, 1, 2, 5]])
    c = confusion(pred, gt)
    assert c.sum() == 7, "class 13 pixels are ignored"
    iou = iou_per_class(c)
    assert abs(iou[0] - 4 / 5) < 1e-9 and abs(iou[1] - 2 / 3) < 1e-9
    assert abs(miou(c) - (4 / 5 + 2 / 3) / 2) < 1e-9
    perfect = np.stack([confusion(gt, gt)] * 30)
    worse = np.stack([c] * 30)
    bs = paired_bootstrap_delta(worse, perfect, n_boot=50)
    assert bs["delta"] > 0 and bs["ci95"][0] > 0
    same = paired_bootstrap_delta(worse, worse, n_boot=20)
    assert same["delta"] == 0 and not same["significant"]
    pc = bs["per_class"]
    assert pc[1]["delta"] > 0 and pc[1]["significant"] and pc[3]["delta"] is None, "per-class deltas + CIs"


def test_figure_rule_skips_single_class_patches():
    from scripts.train_report import eligible_for_figures, select_figures
    single = confusion(np.full((10, 10), 4), np.full((10, 10), 4))          # one class: mIoU 1.0
    scene = np.zeros((10, 10), int)
    scene[:, :3], scene[:, 3:6], scene[:, 6:] = 1, 6, 10
    pred = scene.copy()
    pred[:, 3:6] = 7                                                        # one class wrong
    multi = confusion(pred, scene)
    conf = np.stack([single, multi, multi])
    ok = eligible_for_figures(conf)
    assert ok.tolist() == [False, True, True]
    picks = select_figures(np.array([0.0, 0.3, 0.5]), np.array([1.0, 0.6, 0.5]), n=2, eligible=ok)
    assert [i for i, _ in picks] == [1, 2], "the single-class patch must never be picked"


def test_trainable_state_roundtrip():
    a, b = TinySeg(4), TinySeg(4)
    st = trainable_state(a)
    load_trainable(b, st)
    x = torch.randn(1, 4, 32, 32)
    idx = torch.tensor([[0, 3]])
    assert torch.equal(a(x, idx), b(x, idx))
    try:
        load_trainable(TinySeg(3), st)
        raise AssertionError("loading a 4-channel checkpoint into a 3-channel model must fail")
    except RuntimeError:
        pass


def test_chunked_prediction_matches_one_pass():
    from train.sam3_rgbi import predict_classes
    torch.manual_seed(0)
    net = TinySeg(4).eval()
    x = torch.randn(2, 4, 64, 64)
    full = net(x, torch.arange(12).unsqueeze(0).expand(2, -1))
    expect = torch.nn.functional.interpolate(full, size=(64, 64), mode="bilinear", align_corners=False).argmax(1) + 1
    for chunk in (1, 4, 5, 12):
        assert torch.equal(predict_classes(net, x, (64, 64), chunk=chunk), expect), chunk


def test_schedule_is_paired_across_channels():
    try:
        import tifffile
    except ImportError:
        print("  (tifffile missing: skipped schedule pairing test)")
        return
    from train.data import TrainSchedule
    tmp = tempfile.mkdtemp()
    rels = []
    rng = np.random.default_rng(0)
    for k in range(4):
        d = os.path.join(tmp, "D001_2020", "aerial", "Z1")
        m = os.path.join(tmp, "D001_2020", "labels", "Z1")
        os.makedirs(d, exist_ok=True)
        os.makedirs(m, exist_ok=True)
        tifffile.imwrite(os.path.join(d, f"IMG_{k}.tif"), rng.integers(0, 255, (5, 64, 64), dtype=np.uint8),
                         planarconfig="separate", photometric="minisblack")  # one page, 5 bands (rasterio reads it as such)
        tifffile.imwrite(os.path.join(m, f"MSK_{k}.tif"), rng.integers(1, 13, (64, 64), dtype=np.uint8))
        rels.append(f"D001_2020/aerial/Z1/IMG_{k}.tif")
    a = TrainSchedule(tmp, rels, "rgb", "rgb", seed=3, n_items=10)
    b = TrainSchedule(tmp, rels, "rgb", "rgbi", seed=3, n_items=10)
    for i in range(10):
        xa, ca, ta, va = a[i]
        xb, cb, tb, vb = b[i]
        assert xa.shape[0] == 3 and xb.shape[0] == 4
        assert torch.equal(xa, xb[:3]) and torch.equal(ca, cb) and torch.equal(ta, tb), \
            "RGB and RGBI runs must see the same image, augmentation and prompts"
    resumed = TrainSchedule(tmp, rels, "rgb", "rgbi", seed=3, n_items=10, start_item=6)
    assert torch.equal(resumed[0][0], b[6][0]), "resuming must continue the same schedule"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"all {len(tests)} training tests passed")
