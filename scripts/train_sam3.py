"""Fine-tune SAM 3 (LoRA + semantic head) on FLAIR with RGB or RGB+NIR input. One run = one
(channels, seed). Resumable: re-running the same command continues from the last checkpoint.

    python scripts/train_sam3.py --channels rgbi --seed 0 --root $FLAIR_DATA_DIR \
        --split outputs/train/split.json --band_order rgb --out outputs/train/runs/rgbi_s0

    --probe N   run N optimizer steps, print peak GPU memory and seconds/step, save nothing.
                Exit 3 = out of GPU memory (run_train.sh then halves the batch).

Outputs in --out: config.json, train_log.jsonl, last.pt (resume state), final.pt (trainable weights).
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from train.data import CHANNELS, TrainSchedule, load_split  # noqa: E402
from train.sam3_rgbi import PATCH_EMBED, build, load_trainable, normalise, trainable_state  # noqa: E402

# Settings that define a run. A resumed run must match them exactly.
# (batch x accum may be re-split, e.g. after an out-of-memory probe: the data schedule depends only on
# their product, so a run can resume with a different split of the same effective batch.)
RUN_KEYS = ("model", "channels", "seed", "steps", "effective_batch", "lr", "prompts", "lora_rank",
            "split_sha", "band_order", "precision")


def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seg_loss(logits, target, valid):
    """BCE + Dice per prompted map, over valid pixels. logits/target (B, P, H, W), valid (B, H, W)."""
    v = valid.unsqueeze(1).float().expand_as(target)
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * v).sum() / v.sum().clamp(min=1)
    p = torch.sigmoid(logits) * v
    inter = (p * target).sum((2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((2, 3)) + (target * v).sum((2, 3)) + 1)
    return bce + dice.mean()


def lr_at(step, total, base, warmup):
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sam3", choices=["sam3", "tiny"])
    ap.add_argument("--channels", required=True, choices=sorted(CHANNELS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--band_order", required=True, choices=["rgb", "bgr"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=4000, help="optimizer steps")
    ap.add_argument("--batch", type=int, default=2, help="images per forward pass")
    ap.add_argument("--accum", type=int, default=4, help="forward passes per optimizer step")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--prompts", type=int, default=3, help="class prompts per image per step")
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--save_every", type=int, default=250, help="checkpoint every N steps ...")
    ap.add_argument("--save_minutes", type=float, default=10.0, help="... or every M minutes (shared GPUs: a run "
                    "killed by someone else's memory use loses at most this much work)")
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--probe", type=int, default=0)
    a = ap.parse_args()

    split = load_split(a.split)
    cfg = {k: getattr(a, k) for k in RUN_KEYS if hasattr(a, k)}
    cfg["split_sha"] = split["sha"]
    cfg["effective_batch"] = a.batch * a.accum
    os.makedirs(a.out, exist_ok=True)
    cfg_path, last_path, final_path = (os.path.join(a.out, f) for f in ("config.json", "last.pt", "final.pt"))

    if not a.probe:
        if os.path.exists(cfg_path):
            with open(cfg_path) as fh:
                old = json.load(fh)
            diff = {k: (old.get(k), cfg[k]) for k in RUN_KEYS if old.get(k) != cfg[k]}
            if diff:
                sys.exit(f"ERROR: {a.out} was started with different settings {diff}. Use a new --out "
                         f"or delete that directory; runs are never mixed.")
        else:
            with open(cfg_path, "w") as fh:
                json.dump(cfg, fh, indent=1)
        if os.path.exists(final_path):
            print(f"  {a.out}: already finished ({a.steps} steps)")
            return

    dev = pick_device(a.device)
    amp = dev == "cuda" and a.precision == "bf16"
    print(f"  device {dev}  |  bf16 autocast {amp}  |  {a.channels} ({CHANNELS[a.channels]} ch)  |  seed {a.seed}")

    torch.manual_seed(a.seed)
    net = build(a.model, CHANNELS[a.channels], a.lora_rank, seed=a.seed).to(dev)
    net.enable_checkpointing()
    net.train()
    trainable = [(n, p) for n, p in net.named_parameters() if p.requires_grad]
    print(f"  trainable {sum(p.numel() for _, p in trainable) / 1e6:.2f} M parameters"
          f"  |  LoRA layers {getattr(net, 'n_lora', 0)}")
    groups = [{"params": [p for n, p in trainable if PATCH_EMBED in n], "lr_scale": 0.1},
              {"params": [p for n, p in trainable if PATCH_EMBED not in n], "lr_scale": 1.0}]
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=1e-4)

    step = 0
    if os.path.exists(last_path) and not a.probe:
        st = torch.load(last_path, map_location="cpu")
        load_trainable(net, st["trainable"])
        opt.load_state_dict(st["opt"])
        step = st["step"]
        print(f"  resumed from step {step}")

    total = a.probe or a.steps
    per_step = a.batch * a.accum
    ds = TrainSchedule(a.root, split["train"], a.band_order, a.channels, a.seed, n_items=total * per_step,
                       n_prompts=a.prompts, start_item=step * per_step)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=a.workers,
                    pin_memory=dev == "cuda", persistent_workers=False)
    warmup = max(1, min(200, a.steps // 10))
    log = open(os.path.join(a.out, "train_log.jsonl"), "a") if not a.probe else None
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_last, losses, step_times = time.time(), [], []
    t_saved = time.time()
    micro = 0
    try:
        for x, cls, target, valid in dl:
            x, cls = normalise(x.to(dev, non_blocking=True)), cls.to(dev)
            target, valid = target.to(dev), valid.to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                logits = net(x, cls)
            logits = F.interpolate(logits.float(), size=target.shape[-2:], mode="bilinear", align_corners=False)
            loss = seg_loss(logits, target, valid)
            if not torch.isfinite(loss):
                print(f"ERROR: loss is {loss.item()} at step {step}; stopping (lower --lr?)")
                sys.exit(4)
            (loss / a.accum).backward()
            losses.append(loss.item())
            micro += 1
            if micro % a.accum:
                continue
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, a.lr, warmup) * g["lr_scale"]
            torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            now = time.time()
            step_times.append(now - t_last)
            t_last = now
            if log and (step % a.log_every == 0 or step == total):
                rec = {"step": step, "loss": sum(losses) / len(losses), "lr": lr_at(step - 1, total, a.lr, warmup),
                       "s_per_step": sum(step_times) / len(step_times)}
                if dev == "cuda":
                    rec["peak_gb"] = torch.cuda.max_memory_reserved() / 1e9
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(f"  step {step}/{total}  loss {rec['loss']:.4f}  {rec['s_per_step']:.2f} s/step", flush=True)
                losses, step_times = [], []
            if not a.probe and (step % a.save_every == 0 or step == total
                                or time.time() - t_saved > 60 * a.save_minutes):
                t_saved = time.time()
                tmp = last_path + ".tmp"
                torch.save({"trainable": trainable_state(net), "opt": opt.state_dict(), "step": step}, tmp)
                os.replace(tmp, last_path)
    except torch.cuda.OutOfMemoryError:
        print(f"OUT OF GPU MEMORY at batch {a.batch} (step {step}); exit 3 = re-queue, resume from last checkpoint")
        sys.exit(3)

    if a.probe:
        # reserved (not allocated): what nvidia-smi sees, so what other users' jobs compete with
        peak = torch.cuda.max_memory_reserved() / 1e9 if dev == "cuda" else float("nan")
        sps = sum(step_times[1:]) / max(1, len(step_times) - 1) if len(step_times) > 1 else float("nan")
        print(f"PROBE ok: batch {a.batch}  peak {peak:.2f} GB  {sps:.2f} s/optimizer-step "
              f"(batch {a.batch} x accum {a.accum})")
        with open(os.path.join(a.out, "probe.json"), "w") as fh:
            json.dump({"batch": a.batch, "accum": a.accum, "peak_gb": peak, "s_per_step": sps}, fh)
        return
    if step < total:
        sys.exit(f"ERROR: data ran out at step {step} < {total}")
    torch.save({"trainable": trainable_state(net), "config": cfg}, final_path)
    print(f"  finished: {final_path}")


if __name__ == "__main__":
    main()
