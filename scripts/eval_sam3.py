"""Evaluate a trained run (or untrained SAM 3 with --zero_shot) on the held-out test domains.

    python scripts/eval_sam3.py --run outputs/train/runs/rgbi_s0 --root ... --split ... --band_order rgb
    python scripts/eval_sam3.py --zero_shot --channels rgb --out outputs/train/runs/zeroshot_rgb ...

All 12 class prompts are run per image; the prediction is the per-pixel argmax. Writes eval.npz
(per-image 12x12 confusions, paired by image across runs) and eval.json (mIoU, per-class IoU).
Resumable: partial results are saved every --save_every images.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from scripts.train_sam3 import pick_device  # noqa: E402
from train.data import CHANNELS, TestSet, load_split, test_subset  # noqa: E402
from train.metrics import K, confusion, summary  # noqa: E402
from train.sam3_rgbi import build, load_trainable, normalise, predict_classes  # noqa: E402


def load_net(run_dir, model, channels, lora_rank, zero_shot, device):
    """The model of a finished run (or untrained SAM 3) on `device`, in eval mode."""
    if zero_shot:
        net = build(model, CHANNELS[channels], lora_rank)
    else:
        with open(os.path.join(run_dir, "config.json")) as fh:
            cfg = json.load(fh)
        final = os.path.join(run_dir, "final.pt")
        if not os.path.exists(final):
            raise FileNotFoundError(f"{final} missing: training of {run_dir} did not finish")
        net = build(cfg["model"], CHANNELS[cfg["channels"]], cfg["lora_rank"], seed=cfg["seed"])
        load_trainable(net, torch.load(final, map_location="cpu")["trainable"])
    return net.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="a finished training run directory (reads its config.json)")
    ap.add_argument("--zero_shot", action="store_true", help="untrained SAM 3, text prompts only")
    ap.add_argument("--model", default="sam3", choices=["sam3", "tiny"])
    ap.add_argument("--channels", default="rgb", choices=sorted(CHANNELS), help="zero-shot only")
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--out", help="defaults to --run")
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--band_order", required=True, choices=["rgb", "bgr"])
    ap.add_argument("--limit", type=int, default=0, help="evaluate a fixed random subset of the test set")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    if bool(a.run) == bool(a.zero_shot):
        sys.exit("ERROR: give exactly one of --run or --zero_shot")
    out = a.out or a.run
    os.makedirs(out, exist_ok=True)

    split = load_split(a.split)
    rels = test_subset(split, a.limit)
    meta = {"split_sha": split["sha"], "n_test": len(rels), "limit": a.limit or None, "zero_shot": a.zero_shot}
    if a.run:
        with open(os.path.join(a.run, "config.json")) as fh:
            cfg = json.load(fh)
        if cfg["split_sha"] != split["sha"]:
            sys.exit(f"ERROR: {a.run} was trained on split {cfg['split_sha']}, not {split['sha']}")
        meta.update({k: cfg[k] for k in ("model", "channels", "seed")})
        meta["nir_aug"] = bool(cfg.get("nir_aug", False))
    else:
        meta.update({"model": a.model, "channels": a.channels, "seed": None})
    meta["is_measurement"] = meta["model"] != "tiny"

    final_npz, partial = os.path.join(out, "eval.npz"), os.path.join(out, "eval_partial.npz")
    if os.path.exists(final_npz):
        old = np.load(final_npz, allow_pickle=False)
        if str(old["split_sha"]) == split["sha"] and int(old["conf"].shape[0]) == len(rels):
            print(f"  {out}: evaluation already finished")
            return
        sys.exit(f"ERROR: {final_npz} is from a different split or test subset; delete it to re-evaluate")

    conf = np.zeros((len(rels), K, K), dtype=np.int64)
    done = np.zeros(len(rels), dtype=bool)
    if os.path.exists(partial):
        p = np.load(partial, allow_pickle=False)
        if str(p["split_sha"]) == split["sha"] and p["conf"].shape == conf.shape:
            conf, done = p["conf"], p["done"]
            print(f"  resuming evaluation: {int(done.sum())}/{len(rels)} images already done")

    dev = pick_device(a.device)
    net = load_net(a.run, a.model, a.channels, a.lora_rank, a.zero_shot, dev)
    todo = [i for i in range(len(rels)) if not done[i]]
    ds = TestSet(a.root, rels, a.band_order, meta["channels"])
    dl = DataLoader(torch.utils.data.Subset(ds, todo), batch_size=1, num_workers=a.workers)
    amp = dev == "cuda"
    t0, n = time.time(), 0

    def save(path):
        tmp = path + ".tmp.npz"
        np.savez(tmp, conf=conf, done=done, split_sha=split["sha"], rels=np.array(rels))
        os.replace(tmp, path)

    try:
        with torch.inference_mode():
            for x, y, i in dl:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                    pred = predict_classes(net, normalise(x.to(dev)), out_size=y.shape[-2:])
                conf[int(i)] = confusion(pred[0].cpu().numpy(), y[0].numpy())
                done[int(i)] = True
                n += 1
                if n % a.save_every == 0:
                    save(partial)
                    print(f"  {int(done.sum())}/{len(rels)} images  ({(time.time() - t0) / n:.2f} s/image)", flush=True)
    except torch.cuda.OutOfMemoryError:
        save(partial)
        print("OUT OF GPU MEMORY during evaluation; progress saved; exit 3 = re-queue")
        sys.exit(3)
    if dev == "cuda":
        peak = torch.cuda.max_memory_reserved() / 1e9
        print(f"  eval peak GPU memory (reserved) {peak:.2f} GB")
        with open(os.path.join(out, "eval_mem.json"), "w") as fh:
            json.dump({"peak_gb": peak}, fh)

    save(final_npz)
    if os.path.exists(partial):
        os.remove(partial)
    res = {**meta, **summary(conf.sum(0))}
    with open(os.path.join(out, "eval.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"  {out}: mIoU {res['miou']:.4f}  pixel acc {res['pixel_acc']:.4f}  ({len(rels)} test images)")


if __name__ == "__main__":
    main()
