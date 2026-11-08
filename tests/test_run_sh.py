"""End-to-end test of run.sh on synthetic TIFFs with the explicit mock model.

Needs tifffile (or rasterio) and bash, no GPU, no weights. It runs the real script: unit tests,
data check, automatic band-order check, episode generation, smoke + full run, report, bundle.
Skips itself (exit 0) when tifffile/rasterio are not installed.

    python tests/test_run_sh.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

try:
    import tifffile
except ImportError:
    print("tifffile not installed: skipping run.sh end-to-end test")
    sys.exit(0)

# Multi-band TIFFs must be ONE page with N bands; tifffile's default writes N pages, which
# rasterio (used when installed, e.g. on the cluster) reads as a single band.
BANDS = {"planarconfig": "separate", "photometric": "minisblack"}

# class id -> (R, G, B) of the synthetic scene. Soil is red-dominant, water blue-dominant.
COLOURS = {4: (180, 140, 100), 5: (40, 80, 130), 7: (40, 100, 40)}


def make_dataset(root, order):
    """Two domains x 6 patches, 512x512, three vertical bands: soil | water | deciduous."""
    rng = np.random.default_rng(0)
    for dom in ("D001_2020", "D002_2021"):
        for k in range(6):                                  # 6 per domain -> 5 cross-domain supports
            name = f"{dom[1:4]}{k:03d}"
            img_dir = os.path.join(root, dom, "aerial", "Z1_AA")
            msk_dir = os.path.join(root, dom, "labels", "Z1_AA")
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(msk_dir, exist_ok=True)
            img = np.zeros((5, 512, 512), np.uint8)
            mask = np.zeros((512, 512), np.uint8)
            for j, (cls, rgb) in enumerate(COLOURS.items()):
                sl = slice(j * 171, 512 if j == 2 else (j + 1) * 171)
                chans = rgb if order == "rgb" else rgb[::-1]
                for c in range(3):
                    img[c, :, sl] = chans[c]
                mask[:, sl] = cls
            img[:3] = np.clip(img[:3].astype(int) + rng.integers(-4, 5, (3, 512, 512)), 0, 255)
            img[3] = 200                                    # NIR
            tifffile.imwrite(os.path.join(img_dir, f"IMG_{name}.tif"), img, **BANDS)
            tifffile.imwrite(os.path.join(msk_dir, f"MSK_{name}.tif"), mask)


def run(env_extra, tmp):
    env = {**os.environ, "PYTHON": sys.executable, "FLAIR_DATA_DIR": os.path.join(tmp, "data"),
           "OUT": os.path.join(tmp, "out"), "RUNS": "mock::", "N_PER_CLASS": "6",
           "SKIP_TESTS": "1", "CUDA_VISIBLE_DEVICES": "", **env_extra}
    env.pop("BAND_ORDER", None) if "BAND_ORDER" not in env_extra else None
    return subprocess.run(["bash", "run.sh"], cwd=ROOT, env=env, capture_output=True, text=True)


def test_full_run_rgb_and_bgr():
    for order in ("rgb", "bgr"):
        print(f"[run.sh] synthetic data stored as {order.upper()}: band order must be detected, all stages run...")
        tmp = tempfile.mkdtemp()
        try:
            make_dataset(os.path.join(tmp, "data"), order)
            r = run({}, tmp)
            out = os.path.join(tmp, "out")
            assert r.returncode == 0, f"run.sh exited {r.returncode}\n{r.stdout[-3000:]}\n{r.stderr[-1500:]}"
            assert f"BAND_ORDER={order}" in r.stdout, r.stdout[-2000:]
            assert open(os.path.join(out, "band_order.txt")).read().strip() == order
            meta = json.load(open(os.path.join(out, "episodes.json")))["meta"]
            assert meta["band_order"] == order and meta["episodes_sha"]
            assert meta["domain_mode"] == "cross", meta
            for f in ("summary.md", "per_class.csv", "per_class_iou.png"):
                assert os.path.exists(os.path.join(out, "report", f)), f
            figs = [f for f in os.listdir(os.path.join(out, "report", "figures")) if f.endswith(".png")]
            assert len(figs) == 10 or len(figs) == len(set(figs)) >= 3, figs
            summary = open(os.path.join(out, "report", "summary.md")).read()
            assert "NOT a model measurement" in summary, "mock runs must be flagged in the report"
            assert "baseline:all_fg" in summary and "empty predictions" in summary
            for arm in ("1shot", "5shot", "oracle", "text"):
                runs = os.path.join(out, "runs", f"mock_colour_{arm}", "results.jsonl")
                assert os.path.exists(runs) and sum(1 for _ in open(runs)) > 0, f"arm {arm} produced nothing"
            assert os.path.exists(os.path.join(out, "report", "all_arms", "summary.md"))
            assert os.path.exists(os.path.join(out, "report.tar.gz"))
            print(f"      --> Passed! ({len(figs)} figures)")
        finally:
            shutil.rmtree(tmp)


def test_guards():
    print("[run.sh] guards: stale episodes, missing data, unusable model, undecidable band order...")
    tmp = tempfile.mkdtemp()
    try:
        make_dataset(os.path.join(tmp, "data"), "rgb")
        assert run({}, tmp).returncode == 0

        # 1. Stale episodes: same file, different requested configuration -> refuse, do not reuse.
        r = run({"N_PER_CLASS": "7", "BAND_ORDER": "rgb"}, tmp)
        assert r.returncode != 0 and "DIFFERENT configuration" in r.stdout, r.stdout[-1500:]

        # 2. Identical configuration reuses the file and resumes cleanly.
        r = run({"BAND_ORDER": "rgb"}, tmp)
        assert r.returncode == 0 and "Reusing existing" in r.stdout, r.stdout[-1500:]

        # 3. Wrong band order declared against existing episodes -> refuse.
        r = run({"BAND_ORDER": "bgr"}, tmp)
        assert r.returncode != 0 and "DIFFERENT configuration" in r.stdout

        # 4. Results from a different episode list must never be mixed.
        shutil.rmtree(os.path.join(tmp, "out", "episodes.json"), ignore_errors=True)
        os.remove(os.path.join(tmp, "out", "episodes.json"))
        r = run({"BAND_ORDER": "rgb", "SEED": "1"}, tmp)
        assert r.returncode != 0 and "different episode file" in r.stdout, r.stdout[-1500:]

        # 5. Missing data directory.
        r = run({"FLAIR_DATA_DIR": os.path.join(tmp, "nope")}, tmp)
        assert r.returncode != 0 and "data check failed" in r.stdout

        # 6. One unusable model is skipped, the working one still runs, status is non-zero.
        shutil.rmtree(os.path.join(tmp, "out"))
        r = run({"RUNS": "mock:: sam1:vit_b:/does/not/exist.pth", "ARMS": "1shot"}, tmp)
        assert r.returncode == 2, (r.returncode, r.stdout[-1500:])
        assert "checkpoint not found" in r.stdout and "SKIPPED (preflight): sam1:vit_b" in r.stdout
        assert os.path.exists(os.path.join(tmp, "out", "report", "summary.md"))

        # 7. A 5-shot arm on an episodes file that stores only 1 support must fail loudly, not run 1-shot.
        import subprocess as sp
        e1 = os.path.join(tmp, "one_support.json")
        meta = json.load(open(os.path.join(tmp, "out", "episodes.json")))
        meta["meta"]["k_shot_max"] = 1
        json.dump(meta, open(e1, "w"))
        r = sp.run([sys.executable, "scripts/run_model.py", "--model", "mock", "--root", os.path.join(tmp, "data"),
                    "--episodes", e1, "--out", os.path.join(tmp, "k5"), "--k_shot", "5"],
                   cwd=ROOT, capture_output=True, text=True)
        assert r.returncode != 0 and "stores only 1 support" in (r.stdout + r.stderr), r.stdout[-800:]
        print("      --> Passed!")
    finally:
        shutil.rmtree(tmp)

    print("[run.sh] undecidable band order stops with instructions instead of guessing...")
    tmp = tempfile.mkdtemp()
    try:
        make_dataset(os.path.join(tmp, "data"), "rgb")
        # Make soil and water both look 'blue-ish' vs 'red-ish' inconsistently: grey scene.
        for p in os.popen(f"find {tmp}/data -name 'IMG_*.tif'").read().split():
            img = tifffile.imread(p)
            img[:3] = 120
            tifffile.imwrite(p, img, **BANDS)
        r = run({}, tmp)
        assert r.returncode != 0 and "could not be decided automatically" in r.stdout, r.stdout[-1500:]
        assert "BAND_ORDER=rgb bash run.sh" in r.stdout
        assert os.path.exists(os.path.join(tmp, "out", "band_check.png"))
        print("      --> Passed!")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    print("\n--- run.sh end-to-end tests (synthetic TIFFs, mock model) ---")
    test_full_run_rgb_and_bgr()
    test_guards()
    print("\nAll run.sh tests passed.\n")
