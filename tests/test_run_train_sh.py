"""End-to-end test of run_train.sh on synthetic TIFFs with the `tiny` stand-in model, on CPU.

Runs the real script: environment, data, band order, split, probe, parallel train + eval, report,
bundle; then checks resume and the refusal to mix settings. Skips when tifffile is missing.

    python tests/test_run_train_sh.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "tests"))

try:
    import tifffile  # noqa: F401
except ImportError:
    print("tifffile not installed: skipping run_train.sh end-to-end test")
    sys.exit(0)

from test_run_sh import make_dataset  # noqa: E402


def run(tmp, mode=None, **extra):
    env = {k: v for k, v in os.environ.items() if k != "BAND_ORDER"}
    env = {**env, "PYTHON": sys.executable, "FLAIR_DATA_DIR": os.path.join(tmp, "data"),
           "OUT": os.path.join(tmp, "out"), "MODEL": "tiny", "DEVICE": "cpu", "GPUS": "cpu",
           "SEEDS": "0 1", "STEPS": "4", "BATCH": "2", "ACCUM": "1", "WORKERS": "0",
           "SKIP_TESTS": "1", "CUDA_VISIBLE_DEVICES": "", **extra}
    return subprocess.run(["bash", "run_train.sh"] + ([mode] if mode else []), cwd=ROOT, env=env,
                          capture_output=True, text=True)


def main():
    tmp = tempfile.mkdtemp()
    try:
        make_dataset(os.path.join(tmp, "data"), "rgb")
        print("[run_train.sh] full run with the tiny stand-in...")
        r = run(tmp)
        out = os.path.join(tmp, "out", "train")
        assert r.returncode == 0, f"exit {r.returncode}\n{r.stdout[-4000:]}\n{r.stderr[-1500:]}"
        assert "BAND_ORDER=rgb" in r.stdout
        assert "pipeline stand-in" in r.stdout
        for tag in ("rgb_s0", "rgbi_s0", "rgb_s1", "rgbi_s1", "zeroshot_rgb"):
            assert os.path.exists(os.path.join(out, "runs", tag, "DONE")), tag
        cfg0 = json.load(open(os.path.join(out, "runs", "rgb_s0", "config.json")))
        cfg1 = json.load(open(os.path.join(out, "runs", "rgbi_s0", "config.json")))
        assert {k for k in cfg0 if cfg0[k] != cfg1[k]} == {"channels"}, "RGB and RGBI must differ only in channels"
        summ = open(os.path.join(out, "report", "summary.md")).read()
        assert "RGB+NIR" in summ and "95% CI" in summ and "NOT results" in summ, summ
        res = json.load(open(os.path.join(out, "report", "summary.json")))
        assert res["seeds_paired"] == [0, 1] and "delta_rgbi_minus_rgb" in res
        figs = [f for f in os.listdir(os.path.join(out, "report", "figures")) if f.endswith(".png")]
        n_test = res["n_test"]
        assert len(figs) == min(10, n_test), (figs, n_test)
        assert os.path.exists(os.path.join(tmp, "out", "train_report.tar.gz"))

        print("[train_report] without a zero-shot run: no zero-shot column...")
        rep = subprocess.run([sys.executable, "scripts/train_report.py", "--runs",
                              *[os.path.join(out, "runs", t) for t in ("rgb_s0", "rgbi_s0", "rgb_s1", "rgbi_s1")],
                              "--root", os.path.join(tmp, "data"), "--split", os.path.join(out, "split.json"),
                              "--band_order", "rgb", "--out", os.path.join(tmp, "rep_nozs"), "--n_figures", "0",
                              "--n_boot", "20"], cwd=ROOT, capture_output=True, text=True)
        assert rep.returncode == 0, rep.stdout[-2000:] + rep.stderr[-2000:]
        md = open(os.path.join(tmp, "rep_nozs", "summary.md")).read()
        assert "zero-shot" not in md and "| Class | RGB | RGB+NIR | Δ |" in md, md

        print("[nir_diagnostic] loss curves + NIR ablation on the finished runs...")
        d = run(tmp, "diagnose", DIAG_N="3")
        assert d.returncode == 0, d.stdout[-2000:] + d.stderr[-2000:]
        assert "NIR ablation" in d.stdout
        diag = json.load(open(os.path.join(out, "diagnostic", "nir_diagnostic.json")))
        assert set(diag["ablation"]) == {"rgbi_s0", "rgbi_s1"} and "nir_swap" in diag["ablation"]["rgbi_s0"]
        assert diag["loss"]["rgb_s0"]["steps"] == 4

        print("[run_train.sh] re-run resumes (nothing retrained)...")
        final = os.path.join(out, "runs", "rgb_s0", "final.pt")
        mtime = os.path.getmtime(final)
        r = run(tmp)
        assert r.returncode == 0, r.stdout[-3000:]
        assert os.path.getmtime(final) == mtime, "a finished run must not be retrained"

        print("[run_train.sh] changed settings on existing runs are refused before anything runs...")
        r = run(tmp, STEPS="5")
        assert r.returncode == 1 and "different settings" in r.stdout, (r.returncode, r.stdout[-3000:])
        assert os.path.getmtime(final) == mtime

        print("[run_train.sh] a 3-band dataset is refused before training...")
        tmp2 = tempfile.mkdtemp()
        try:
            make_dataset(os.path.join(tmp2, "data"), "rgb")
            for dp, _, fs in os.walk(os.path.join(tmp2, "data")):
                for f in fs:
                    if f.startswith("IMG_"):
                        p = os.path.join(dp, f)
                        tifffile.imwrite(p, tifffile.imread(p)[:3], planarconfig="separate", photometric="minisblack")
            r = run(tmp2, BAND_ORDER="rgb")
            assert r.returncode == 1 and "needs R, G, B, NIR" in r.stdout, r.stdout[-2000:]
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("run_train.sh end-to-end tests passed")


if __name__ == "__main__":
    main()
