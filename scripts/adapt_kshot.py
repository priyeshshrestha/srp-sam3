"""Few-shot adaptation to a NEW REGION (and a new NIR camera): fine-tune a trained run on K labelled
patches of an unseen test domain, then evaluate on other patches of that domain.

    python scripts/adapt_kshot.py --run outputs/train/runs/rgbi_s0 --train_dir outputs/train \
        --root $FLAIR_DATA_DIR --band_order rgb --batch 2

This is the drone scenario: a model trained on FLAIR meets a region it has never seen, possibly through
a different camera, and we can afford to label only K patches there.

Protocol (fixed before any result):
  * domains   --domains test domains, chosen by a fixed rule (seed 0) among domains with enough patches;
              identical for every run
  * per domain: a fixed (seed) permutation of its patches; the first --eval_n are the EVALUATION set,
              the next K are the labelled support set. Supports are nested (K=1 in K=5 in K=20).
  * K = 0     the trained model without adaptation (reference for every K)
  * K > 0     fine-tune the same parts that were trained (LoRA, DETR encoder, semantic head, patch embedding
              at 0.1x lr) for --steps steps of --eff_batch images drawn from the K supports (random 90-degree
              rotation/flip, 3 class prompts, as in training), AdamW, lr --lr, warm-up + cosine. No NIR
              augmentation during adaptation, for every run alike.
  * conditions  normal          new region, same camera
                camera_b        new region + simulated new NIR camera (train/nir.SHIFTS), applied to supports
                                and evaluation patches alike
                calib:camera_b  as camera_b, then label-free NIR calibration (domain statistics from the
                                domain's unlabelled patches, matched to the FLAIR training set)
              An RGB-only run is adapted under "normal" only (no NIR transform changes its input).

Output: <out>/cells/<domain>__<condition>__k<K>.npz (per-image confusions on the evaluation set) and
<out>/results.jsonl. Resumable per cell; exit 3 = out of GPU memory (the unfinished cell is redone).
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
from train.data import CHANNELS, augment, load_rgbi, load_split, sample_prompts  # noqa: E402
from train.metrics import K as NCLS, confusion, miou  # noqa: E402
from train.nir import parse_condition  # noqa: E402

DEFAULT_CONDITIONS = ["normal", "camera_b", "calib:camera_b"]
CFG_KEYS = ("domains", "ks", "conditions", "steps", "lr", "eff_batch", "prompts", "eval_n", "seed", "split_sha",
            "band_order", "run", "calib_n")


def choose_domains(split, n_domains, need):
    """Fixed rule, independent of the model: test domains with >= `need` patches, seed-0 order."""
    by_dom = {}
    for r in sorted(split["test"]):
        by_dom.setdefault(domain_of(r), []).append(r)
    doms = sorted(by_dom)
    order = [doms[i] for i in np.random.default_rng(0).permutation(len(doms))]
    big = [d for d in order if len(by_dom[d]) >= need]
    chosen = (big or sorted(order, key=lambda d: -len(by_dom[d])))[:n_domains]
    return chosen, by_dom


def domain_sets(by_dom, domain, eval_n, k_max):
    rs = by_dom[domain]
    rng = np.random.default_rng([2, zlib.crc32(domain.encode())])
    rs = [rs[i] for i in rng.permutation(len(rs))]
    n_eval = min(eval_n, max(1, len(rs) - k_max))
    return rs[:n_eval], rs[n_eval:n_eval + k_max]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--train_dir", default="outputs/train")
    ap.add_argument("--root", required=True)
    ap.add_argument("--band_order", required=True, choices=["rgb", "bgr"])
    ap.add_argument("--out", help="default <train_dir>/adapt/<run name>")
    ap.add_argument("--domains", type=int, default=4)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 20])
    ap.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--eff_batch", type=int, default=4, help="images per optimizer step")
    ap.add_argument("--batch", type=int, default=2, help="images per forward pass (memory)")
    ap.add_argument("--prompts", type=int, default=3)
    ap.add_argument("--eval_n", type=int, default=200)
    ap.add_argument("--calib_n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    for c in a.conditions:
        base, _ = parse_condition(c)
        if base == "missing":
            sys.exit("ERROR: adapting to a missing NIR band is not a condition here")

    name = os.path.basename(os.path.normpath(a.run))
    out = a.out or os.path.join(a.train_dir, "adapt", name)
    os.makedirs(os.path.join(out, "cells"), exist_ok=True)
    rcfg = json.load(open(os.path.join(a.run, "config.json")))
    split = load_split(os.path.join(a.train_dir, "split.json"))
    if rcfg["split_sha"] != split["sha"]:
        sys.exit(f"ERROR: {a.run} was trained on split {rcfg['split_sha']}, not {split['sha']}")
    n_ch = CHANNELS[rcfg["channels"]]
    conds = list(a.conditions) if n_ch == 4 else ["normal"]
    ks = sorted({0, *a.ks})
    domains, by_dom = choose_domains(split, a.domains, a.eval_n + max(ks))

    cfg = {"domains": domains, "ks": ks, "conditions": conds, "steps": a.steps, "lr": a.lr, "eff_batch": a.eff_batch,
           "prompts": a.prompts, "eval_n": a.eval_n, "seed": a.seed, "split_sha": split["sha"],
           "band_order": a.band_order, "run": name, "calib_n": a.calib_n}
    cfg_path = os.path.join(out, "config.json")
    if os.path.exists(cfg_path):
        old = json.load(open(cfg_path))
        diff = {k: (old.get(k), cfg[k]) for k in ("steps", "lr", "eff_batch", "prompts", "eval_n", "seed",
                                                   "split_sha", "band_order", "run", "calib_n") if old.get(k) != cfg[k]}
        if diff:
            sys.exit(f"ERROR: {out} was made with different settings {diff}; use another --out")
    with open(cfg_path, "w") as fh:
        json.dump({**cfg, "source": {k: rcfg.get(k) for k in ("model", "channels", "seed", "steps", "nir_aug")}}, fh, indent=1)

    cells = [(d, c, k) for d in domains for c in conds for k in ks]
    cell_path = {cl: os.path.join(out, "cells", f"{cl[0]}__{cl[1].replace(':', '-')}__k{cl[2]}.npz") for cl in cells}
    todo = [cl for cl in cells if not os.path.exists(cell_path[cl])]
    if not todo:
        print(f"  {name}: all {len(cells)} adaptation cells done")
        return
    print(f"  {name}: {len(todo)}/{len(cells)} cells to do  (domains {domains}, K {ks}, conditions {conds})", flush=True)

    import torch
    import torch.nn.functional as F
    from scripts.eval_sam3 import load_net
    from scripts.nir_shift import model_input, reference_stats
    from scripts.train_sam3 import lr_at, pick_device, seg_loss
    from train.nir import calibrate, nir_moments, shift_nir  # noqa: F401
    from train.sam3_rgbi import CLASSES, PATCH_EMBED, load_trainable, normalise, predict_classes, trainable_state

    dev = pick_device(a.device)
    amp = dev == "cuda"
    net = load_net(a.run, rcfg["model"], rcfg["channels"], rcfg["lora_rank"], False, dev)
    source = {k: v.clone() for k, v in trainable_state(net).items()}
    net.enable_checkpointing()
    ref = reference_stats(a.train_dir, a.root, a.band_order, split) if n_ch == 4 else None
    b = max(1, min(a.batch, a.eff_batch))
    while a.eff_batch % b:
        b -= 1
    accum = a.eff_batch // b

    cache = {}

    def domain_data(d):
        if d not in cache:
            ev, sup = domain_sets(by_dom, d, a.eval_n, max(ks))
            ev_data = [load_rgbi(a.root, r, a.band_order) for r in ev]
            sup_data = [load_rgbi(a.root, r, a.band_order) for r in sup]
            # calibration statistics: the domain's unlabelled patches (evaluation + support pool + more)
            extra = [r for r in by_dom[d] if r not in set(ev) | set(sup)][:max(0, a.calib_n - len(ev) - len(sup))]
            nirs = [x[3] for x, _ in ev_data + sup_data] + [load_rgbi(a.root, r, a.band_order)[0][3] for r in extra]
            cache.clear()
            cache[d] = (ev, ev_data, sup_data, nirs[:a.calib_n])
        return cache[d]

    def dstats_for(cond, d, nirs):
        base, cal = parse_condition(cond)
        return {base: {d: nir_moments(shift_nir(n, base) for n in nirs)}} if cal else {}

    def evaluate(ev_data, cond, d, dst):
        net.eval()
        confs = np.zeros((len(ev_data), NCLS, NCLS), np.int64)
        with torch.inference_mode():
            for i, (x, y) in enumerate(ev_data):
                xt = normalise(torch.from_numpy(model_input(x, cond, d, ref, dst, n_ch)).unsqueeze(0).to(dev))
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                    confs[i] = confusion(predict_classes(net, xt, out_size=y.shape)[0].cpu().numpy(), y)
        return confs

    def adapt(sup_data, k, cond, d, dst):
        load_trainable(net, {kk: v.to(dev) for kk, v in source.items()})
        net.train()
        trainable = [(n, p) for n, p in net.named_parameters() if p.requires_grad]
        groups = [{"params": [p for n, p in trainable if PATCH_EMBED in n], "lr_scale": 0.1},
                  {"params": [p for n, p in trainable if PATCH_EMBED not in n], "lr_scale": 1.0}]
        opt = torch.optim.AdamW([g for g in groups if g["params"]], lr=a.lr, weight_decay=1e-4)
        warmup = max(1, a.steps // 10)
        dom_key = zlib.crc32(d.encode())
        losses = []
        for step in range(a.steps):
            for g in opt.param_groups:
                g["lr"] = lr_at(step, a.steps, a.lr, warmup) * g["lr_scale"]
            for m in range(accum):
                xs, cl, tg, vl = [], [], [], []
                for j in range(b):
                    rng = np.random.default_rng([a.seed, dom_key, k, step, m * b + j])
                    x, y = sup_data[int(rng.integers(k))]
                    x, y = augment(x, y, rng)
                    cls = sample_prompts(y, a.prompts, rng)
                    xs.append(model_input(x, cond, d, ref, dst, n_ch))
                    cl.append(cls)
                    tg.append(np.stack([(y == CLASSES[c]) for c in cls]).astype(np.float32))
                    vl.append((y >= 1) & (y <= 19))
                xt = normalise(torch.from_numpy(np.stack(xs)).to(dev))
                target, valid = torch.from_numpy(np.stack(tg)).to(dev), torch.from_numpy(np.stack(vl)).to(dev)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                    logits = net(xt, torch.from_numpy(np.stack(cl)).to(dev))
                logits = F.interpolate(logits.float(), size=target.shape[-2:], mode="bilinear", align_corners=False)
                loss = seg_loss(logits, target, valid)
                (loss / accum).backward()
                losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        return float(np.mean(losses[-max(1, len(losses) // 5):]))

    t0 = time.time()
    try:
        for (d, cond, k) in todo:
            ev, ev_data, sup_data, nirs = domain_data(d)
            dst = dstats_for(cond, d, nirs)
            tc = time.time()
            last_loss = None
            if k == 0:
                load_trainable(net, {kk: v.to(dev) for kk, v in source.items()})
            else:
                last_loss = adapt(sup_data[:k], k, cond, d, dst)
            confs = evaluate(ev_data, cond, d, dst)
            m = miou(confs.sum(0))
            tmp = cell_path[(d, cond, k)] + ".tmp.npz"
            np.savez(tmp, conf=confs, rels=np.array(ev), domain=d, condition=cond, k=k, split_sha=split["sha"])
            os.replace(tmp, cell_path[(d, cond, k)])
            rec = {"domain": d, "condition": cond, "k": k, "miou": m, "n_eval": len(ev), "final_loss": last_loss,
                   "seconds": round(time.time() - tc, 1)}
            with open(os.path.join(out, "results.jsonl"), "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"  {name} {d} {cond} K={k}: mIoU {m:.4f}  ({rec['seconds']:.0f} s)", flush=True)
    except torch.cuda.OutOfMemoryError:
        print("OUT OF GPU MEMORY; finished cells are saved; exit 3 = re-queue")
        sys.exit(3)
    print(f"  {name}: adaptation finished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
