"""Does a trained model survive a change of NIR camera? Evaluate one finished run under simulated cameras.

    python scripts/nir_shift.py --run outputs/train/runs/rgbi_s0 --train_dir outputs/train \
        --root $FLAIR_DATA_DIR --band_order rgb --n_images 500

Conditions (train/nir.py): the real NIR ("normal"), simulated cameras (gain_0.6 ... gain_1.4, gamma_1.25,
camera_b = gain 0.8 + offset 15 + gamma 1.15), no NIR at all ("missing"), and "calib:<condition>" = the
same camera followed by label-free NIR calibration (moment matching of each test DOMAIN's NIR to the
FLAIR training set). Calibration statistics of a domain come from up to --calib_n UNLABELLED test
patches of that domain that are NOT among the evaluated patches.

An RGB-only run is evaluated once ("normal"): no NIR transform can change its input, so its number is
the reference line for every condition.

Every run is evaluated on the SAME fixed test subset (train.data.test_subset), so all runs and conditions
are paired by image. Output: <train_dir>/shift/<run>.npz (per-image 12x12 confusions per condition) and
<run>.json (mIoU per condition). Resumable; adding conditions later only evaluates the new ones.
Exit 3 = out of GPU memory (progress is saved).
"""
import argparse
import json
import os
import sys
import time
import zlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402

from datasets.flair import domain_of  # noqa: E402
from train.data import CHANNELS, load_rgbi, load_split, test_subset  # noqa: E402
from train.metrics import K, confusion, miou  # noqa: E402
from train.nir import calibrate, nir_moments, parse_condition, shift_nir  # noqa: E402

DEFAULT_CONDITIONS = ["normal", "gain_0.6", "gain_0.8", "gain_1.2", "gain_1.4", "gamma_1.25", "camera_b",
                      "missing", "calib:normal", "calib:gain_0.6", "calib:gain_1.4", "calib:camera_b"]
REF_N = 300                       # training patches for the FLAIR reference NIR statistics


# --------------------------------------------------------------------------- calibration statistics
def _atomic_json(path, obj):
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


def reference_stats(train_dir, root, band_order, split, n=REF_N):
    """(mean, std) of raw NIR over a fixed (seed 0) sample of TRAINING patches. Cached in nir_ref_stats.json."""
    path = os.path.join(train_dir, "nir_ref_stats.json")
    if os.path.exists(path):
        st = json.load(open(path))
        if st.get("split_sha") == split["sha"] and st.get("n") == min(n, len(split["train"])):
            return st["mean"], st["std"]
    rels = list(split["train"])
    idx = np.sort(np.random.default_rng(0).permutation(len(rels))[:n])
    mean, std = nir_moments(load_rgbi(root, rels[i], band_order)[0][3] for i in idx)
    _atomic_json(path, {"mean": mean, "std": std, "n": int(len(idx)), "split_sha": split["sha"]})
    return mean, std


def calibration_pool(split, exclude, per_domain):
    """{domain: [rel, ...]}: up to `per_domain` test patches of each domain that are not in `exclude`
    (fixed order; falls back to the excluded ones only if a domain has nothing else)."""
    excl = set(exclude)
    by_dom = {}
    for r in sorted(split["test"]):
        by_dom.setdefault(domain_of(r), []).append(r)
    pool = {}
    for d, rs in by_dom.items():
        rng = np.random.default_rng([1, zlib.crc32(d.encode())])
        rs = [rs[i] for i in rng.permutation(len(rs))]
        free = [r for r in rs if r not in excl]
        pool[d] = (free or rs)[:per_domain]
    return pool


def domain_stats(root, band_order, pool, conditions):
    """{base_condition: {domain: (mean, std)}} of the TRANSFORMED NIR, for the calib: conditions."""
    bases = sorted({b for b, cal in map(parse_condition, conditions) if cal})
    out = {b: {} for b in bases}
    if not bases:
        return out
    for d, rels in pool.items():
        nirs = [load_rgbi(root, r, band_order)[0][3] for r in rels]
        for b in bases:
            out[b][d] = nir_moments(shift_nir(n, b) for n in nirs)
    return out


def model_input(x_uint8, condition, domain, ref, dstats, n_ch):
    """(C, H, W) float32 on 0..255 for one condition; RGB untouched."""
    x = x_uint8[:n_ch].astype(np.float32)
    if n_ch < 4:
        return x
    base, cal = parse_condition(condition)
    x[3] = shift_nir(x[3], base)
    if cal:
        x[3] = calibrate(x[3], dstats[base][domain], ref)
    return x


# ---------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="finished training run (final.pt + config.json)")
    ap.add_argument("--train_dir", default="outputs/train", help="holds split.json; results go to <train_dir>/shift")
    ap.add_argument("--root", required=True)
    ap.add_argument("--band_order", required=True, choices=["rgb", "bgr"])
    ap.add_argument("--n_images", type=int, default=500, help="fixed test subset, identical for every run")
    ap.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    ap.add_argument("--calib_n", type=int, default=50, help="unlabelled patches per domain for calibration")
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    for c in a.conditions:
        parse_condition(c)                                   # fail early on a typo

    name = os.path.basename(os.path.normpath(a.run))
    cfg = json.load(open(os.path.join(a.run, "config.json")))
    split = load_split(os.path.join(a.train_dir, "split.json"))
    if cfg["split_sha"] != split["sha"]:
        sys.exit(f"ERROR: {a.run} was trained on split {cfg['split_sha']}, not {split['sha']}")
    n_ch = CHANNELS[cfg["channels"]]
    conds = list(a.conditions) if n_ch == 4 else ["normal"]
    rels = test_subset(split, a.n_images)
    out_dir = os.path.join(a.train_dir, "shift")
    os.makedirs(out_dir, exist_ok=True)
    npz, partial = os.path.join(out_dir, f"{name}.npz"), os.path.join(out_dir, f"{name}.partial.npz")

    # previous results (finished or partial) for the same images are kept; only missing conditions run
    for p in (npz, partial):
        if os.path.exists(p):
            conds += [str(c) for c in np.load(p, allow_pickle=False)["conditions"] if str(c) not in conds]
    conf = np.zeros((len(conds), len(rels), K, K), np.int64)
    done = np.zeros((len(conds), len(rels)), bool)
    for p in (npz, partial):
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=False)
        if str(z["split_sha"]) != split["sha"] or [str(r) for r in z["rels"]] != rels:
            sys.exit(f"ERROR: {p} is from another split or test subset (n_images changed?); move it away")
        old = [str(c) for c in z["conditions"]]
        for j, c in enumerate(conds):
            if c in old:
                conf[j], done[j] = z["conf"][old.index(c)], z["done"][old.index(c)]
    if done.all():
        print(f"  {name}: all {len(conds)} conditions already evaluated")
        _write(npz, partial, conf, done, conds, rels, split, cfg, name, out_dir)
        return

    import torch
    from scripts.eval_sam3 import load_net
    from scripts.train_sam3 import pick_device
    from train.sam3_rgbi import normalise, predict_classes

    ref = reference_stats(a.train_dir, a.root, a.band_order, split) if n_ch == 4 else None
    pool = calibration_pool(split, rels, a.calib_n)
    dstats = domain_stats(a.root, a.band_order, pool, conds) if n_ch == 4 else {}
    if dstats:
        _atomic_json(os.path.join(out_dir, "calibration_stats.json"),
                     {"reference": ref, "per_domain": {b: {d: list(v) for d, v in s.items()} for b, s in dstats.items()},
                      "calib_n": a.calib_n})

    dev = pick_device(a.device)
    net = load_net(a.run, cfg["model"], cfg["channels"], cfg["lora_rank"], False, dev)
    amp = dev == "cuda"
    todo_imgs = np.flatnonzero(~done.all(0))
    print(f"  {name}: {len(todo_imgs)} images x up to {len(conds)} conditions on {dev}", flush=True)
    t0, n = time.time(), 0
    try:
        with torch.inference_mode():
            for i in todo_imgs:
                x, y = load_rgbi(a.root, rels[i], a.band_order)
                dom = domain_of(rels[i])
                for j, c in enumerate(conds):
                    if done[j, i]:
                        continue
                    xt = normalise(torch.from_numpy(model_input(x, c, dom, ref, dstats, n_ch)).unsqueeze(0).to(dev))
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                        pred = predict_classes(net, xt, out_size=y.shape)[0].cpu().numpy()
                    conf[j, i] = confusion(pred, y)
                    done[j, i] = True
                n += 1
                if n % a.save_every == 0:
                    _save(partial, conf, done, conds, rels, split)
                    print(f"  {n}/{len(todo_imgs)} images  ({(time.time() - t0) / n:.1f} s/image)", flush=True)
    except torch.cuda.OutOfMemoryError:
        _save(partial, conf, done, conds, rels, split)
        print("OUT OF GPU MEMORY; progress saved; exit 3 = re-queue")
        sys.exit(3)
    _write(npz, partial, conf, done, conds, rels, split, cfg, name, out_dir)


def _save(path, conf, done, conds, rels, split):
    tmp = path + ".tmp.npz"
    np.savez(tmp, conf=conf, done=done, conditions=np.array(conds), rels=np.array(rels), split_sha=split["sha"])
    os.replace(tmp, path)


def _write(npz, partial, conf, done, conds, rels, split, cfg, name, out_dir):
    _save(npz, conf, done, conds, rels, split)
    if os.path.exists(partial):
        os.remove(partial)
    res = {"run": name, "model": cfg["model"], "channels": cfg["channels"], "seed": cfg["seed"],
           "nir_aug": bool(cfg.get("nir_aug", False)), "n_images": len(rels), "is_measurement": cfg["model"] != "tiny",
           "miou": {c: miou(conf[j].sum(0)) for j, c in enumerate(conds)}}
    _atomic_json(os.path.join(out_dir, f"{name}.json"), res)
    print(f"  {name}: " + "  ".join(f"{c} {v:.4f}" for c, v in res["miou"].items()))


if __name__ == "__main__":
    main()
