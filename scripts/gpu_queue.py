"""Opportunistic GPU queue for a SHARED machine: start a run wherever and whenever a GPU has room.

    python scripts/gpu_queue.py free                                   table of GPU memory
    python scripts/gpu_queue.py wait --min_gb 8                        block until a GPU has 8 GB free; print its index
    python scripts/gpu_queue.py run --tags rgb_s0 rgbi_s0 ... --cmd "bash run_train.sh _job" \
        --mem outputs/train/memory.json --runs_dir outputs/train/runs --status outputs/train/queue_status.json

`run` keeps going until every tag is DONE (or failed --max_fail times for reasons other than memory):
  * every --poll seconds it reads free memory with nvidia-smi
  * a pending run starts on a GPU not already running one of OUR runs, if the GPU has enough free
    memory (from the measured memory model in --mem, plus --margin_gb) for some per-pass batch b;
    b is the largest power of two <= max_batch that fits, accum = effective_batch / b
  * the command run is: <cmd> <tag> <gpu> <batch> <accum>
  * exit 3 = out of GPU memory (someone else grabbed it): the run goes back in the queue and resumes
    from its last checkpoint later; this never counts as a failure
  * a lock file stops two queues from running the same runs at once
Never touches other users' processes.
"""
import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time


def gpu_free_gb(allowed=None):
    """{index: free GB} from nvidia-smi ({} if unavailable)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.total,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return {}
    free = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            idx = parts[0]
            if allowed is None or idx in allowed:
                free[idx] = float(parts[1]) / 1024.0
    return free


def need_gb(mem, b):
    """Peak memory (GB) of a training run at per-pass batch b, from the probe measurements."""
    b1, b2 = mem["train_b1_gb"], mem.get("train_b2_gb")
    per_img = (b2 - b1) if b2 else b1 * 0.5
    return max(b1 + per_img * (b - 1), mem.get("eval_gb", 0.0))


def pick_batch(mem, free, margin):
    eff, bmax = mem["effective_batch"], mem["max_batch"]
    b = bmax
    while b >= 1:
        if eff % b == 0 and need_gb(mem, b) + margin <= free:
            return b, eff // b
        b //= 2
    return None


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def run_queue(a):
    os.makedirs(a.runs_dir, exist_ok=True)
    os.makedirs(a.logs_dir, exist_ok=True)
    lock = os.path.join(os.path.dirname(os.path.abspath(a.runs_dir)), "queue.lock")
    if os.path.exists(lock):
        try:
            pid = int(open(lock).read().strip())
        except ValueError:
            pid = -1
        if pid > 0 and pid != os.getpid() and _alive(pid):
            sys.exit(f"ERROR: another queue (pid {pid}) is already running these runs. Stop it first "
                     f"(kill {pid}) or wait; two queues would train the same run twice.")
    with open(lock, "w") as fh:
        fh.write(str(os.getpid()))

    mem = json.load(open(a.mem))
    allowed = set(a.gpus) if a.gpus and a.gpus != ["all"] else None
    tags = list(a.tags)
    state = {t: {"status": "pending", "fails": 0, "ooms": 0, "not_before": 0.0, "gpu": None, "batch": None}
             for t in tags}
    procs = {}
    cmd = shlex.split(a.cmd)

    def done(t):
        return os.path.exists(os.path.join(a.runs_dir, t, "DONE"))

    def stop_all(*_):
        for t, p in procs.items():
            p.send_signal(signal.SIGTERM)
        if os.path.exists(lock):
            os.remove(lock)
        sys.exit(130)

    signal.signal(signal.SIGINT, stop_all)
    signal.signal(signal.SIGTERM, stop_all)

    def max_parallel():
        """--max_parallel, overridden live by an integer in --max_parallel_file (edit it to change the cap)."""
        n = a.max_parallel
        if a.max_parallel_file and os.path.exists(a.max_parallel_file):
            try:
                n = int(open(a.max_parallel_file).read().strip())
            except ValueError:
                pass
        return max(0, n) if n is not None else 10 ** 6

    def write_status():
        with open(a.status, "w") as fh:
            json.dump({"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "hours_running": (time.time() - t0) / 3600,
                       "max_gpus": None if max_parallel() >= 10 ** 6 else max_parallel(), "runs": state}, fh, indent=1)

    for t in tags:
        if done(t):
            state[t]["status"] = "done"
    t0 = time.time()
    last_wait_msg = 0.0
    while True:
        # 1. collect finished processes
        for t, p in list(procs.items()):
            rc = p.poll()
            if rc is None:
                continue
            del procs[t]
            s = state[t]
            if rc == 0 and done(t):
                s["status"] = "done"
                print(f"[{time.strftime('%m-%d %H:%M')}] done    {t}  (gpu {s['gpu']})", flush=True)
            elif rc == 3:
                s["ooms"] += 1
                s["status"], s["not_before"] = "pending", time.time() + a.oom_backoff
                print(f"[{time.strftime('%m-%d %H:%M')}] OOM     {t} on gpu {s['gpu']} (another job took the memory?)"
                      f"; re-queued, resumes from its last checkpoint", flush=True)
            else:
                s["fails"] += 1
                if s["fails"] >= a.max_fail:
                    s["status"] = "failed"
                    print(f"[{time.strftime('%m-%d %H:%M')}] FAILED  {t} (exit {rc}, {s['fails']} times; "
                          f"see its log). Giving up on it.", flush=True)
                else:
                    s["status"], s["not_before"] = "pending", time.time() + a.fail_backoff
                    print(f"[{time.strftime('%m-%d %H:%M')}] error   {t} (exit {rc}); retry {s['fails']}/{a.max_fail - 1} "
                          f"in {a.fail_backoff // 60} min", flush=True)

        pending = [t for t in tags if state[t]["status"] == "pending" and time.time() >= state[t]["not_before"]]
        if not procs and not any(state[t]["status"] == "pending" for t in tags):
            break

        # 2. start what fits, on the GPUs with the most free memory, never more than the cap at once
        limit = max_parallel()
        if pending and len(procs) < limit:
            free = gpu_free_gb(allowed)
            busy = {state[t]["gpu"] for t in procs}
            for g in sorted(free, key=lambda k: -free[k]):
                if not pending or len(procs) >= limit:
                    break
                if g in busy:
                    continue
                choice = pick_batch(mem, free[g], a.margin_gb)
                if choice is None:
                    continue
                t = pending.pop(0)
                b, acc = choice
                log = open(os.path.join(a.logs_dir, f"{t}.log"), "a")
                log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')}  start on gpu {g} "
                          f"({free[g]:.1f} GB free), batch {b} x accum {acc}\n")
                log.flush()
                procs[t] = subprocess.Popen(cmd + [t, g, str(b), str(acc)], stdout=log, stderr=subprocess.STDOUT)
                state[t].update(status="running", gpu=g, batch=b)
                busy.add(g)
                print(f"[{time.strftime('%m-%d %H:%M')}] start   {t} on gpu {g} ({free[g]:.1f} GB free, batch {b})",
                      flush=True)
            if pending and not procs and time.time() - last_wait_msg > 1800:
                best = max(free.values()) if free else 0
                print(f"[{time.strftime('%m-%d %H:%M')}] waiting: {len(pending)} run(s) need "
                      f"~{need_gb(mem, 1) + a.margin_gb:.1f} GB free on one GPU; best now {best:.1f} GB", flush=True)
                last_wait_msg = time.time()

        write_status()
        time.sleep(a.poll)

    write_status()
    if os.path.exists(lock):
        os.remove(lock)
    failed = [t for t in tags if state[t]["status"] == "failed"]
    print(f"queue finished: {sum(state[t]['status'] == 'done' for t in tags)} done, {len(failed)} failed "
          f"{failed if failed else ''}")
    return 2 if failed else 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("free")
    w = sub.add_parser("wait")
    w.add_argument("--min_gb", type=float, required=True)
    w.add_argument("--gpus", nargs="*")
    w.add_argument("--poll", type=int, default=60)
    r = sub.add_parser("run")
    r.add_argument("--tags", nargs="+", required=True)
    r.add_argument("--cmd", required=True)
    r.add_argument("--mem", required=True)
    r.add_argument("--runs_dir", required=True)
    r.add_argument("--logs_dir", required=True)
    r.add_argument("--status", required=True)
    r.add_argument("--gpus", nargs="*")
    r.add_argument("--margin_gb", type=float, default=1.5)
    r.add_argument("--poll", type=int, default=60)
    r.add_argument("--max_fail", type=int, default=3)
    r.add_argument("--max_parallel", type=int, default=None, help="at most this many of our runs at once [no cap]")
    r.add_argument("--max_parallel_file", default=None, help="file holding an integer that overrides --max_parallel "
                   "while the queue runs")
    r.add_argument("--oom_backoff", type=int, default=300)
    r.add_argument("--fail_backoff", type=int, default=600)
    a = ap.parse_args()

    if a.mode == "free":
        for g, f in sorted(gpu_free_gb().items(), key=lambda kv: int(kv[0])):
            print(f"  gpu {g}: {f:5.1f} GB free")
    elif a.mode == "wait":
        allowed = set(a.gpus) if a.gpus and a.gpus != ["all"] else None
        said = False
        while True:
            free = gpu_free_gb(allowed)
            if not free:
                sys.exit("ERROR: nvidia-smi reports no GPUs")
            g = max(free, key=free.get)
            if free[g] >= a.min_gb:
                print(g)
                return
            if not said:
                print(f"  waiting for a GPU with {a.min_gb:.1f} GB free (best now: gpu {g}, {free[g]:.1f} GB)...",
                      file=sys.stderr, flush=True)
                said = True
            time.sleep(a.poll)
    else:
        sys.exit(run_queue(a))


if __name__ == "__main__":
    main()
