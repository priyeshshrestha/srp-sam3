"""Tests for scripts/gpu_queue.py with a FAKE nvidia-smi (no GPU needed).

Checks: a run only starts on a GPU with enough free memory; one of our runs per GPU; the per-pass
batch is chosen to fit; an out-of-memory exit (3) re-queues without counting as a failure; real
errors are retried then given up; a second queue on the same runs is refused.

    python tests/test_gpu_queue.py
"""
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from scripts.gpu_queue import need_gb, pick_batch  # noqa: E402

# A job: records its arguments; behaviour per tag comes from <runs>/<tag>/plan (list of exit codes).
JOB = r'''#!/usr/bin/env python3
import json, os, sys
tag, gpu, b, acc = sys.argv[1:5]
runs = os.environ["Q_RUNS"]
d = os.path.join(runs, tag); os.makedirs(d, exist_ok=True)
with open(os.path.join(d, "calls"), "a") as fh:
    fh.write(json.dumps([gpu, int(b), int(acc)]) + "\n")
plan_p = os.path.join(d, "plan")
plan = json.load(open(plan_p)) if os.path.exists(plan_p) else [0]
rc = plan.pop(0) if plan else 0
json.dump(plan, open(plan_p, "w"))
if rc == 0:
    open(os.path.join(d, "DONE"), "w").close()
sys.exit(rc)
'''

MEM = {"key": "x", "train_b1_gb": 6.0, "train_b2_gb": 9.0, "eval_gb": 4.0, "max_batch": 2, "effective_batch": 8}


def test_batch_choice():
    assert need_gb(MEM, 1) == 6.0 and need_gb(MEM, 2) == 9.0
    assert pick_batch(MEM, 12.0, 1.5) == (2, 4)
    assert pick_batch(MEM, 8.0, 1.5) == (1, 8)
    assert pick_batch(MEM, 7.0, 1.5) is None


def run_queue(tmp, free_mb, plans, extra=()):
    bin_dir = os.path.join(tmp, "bin")
    os.makedirs(bin_dir, exist_ok=True)
    smi = os.path.join(bin_dir, "nvidia-smi")
    with open(smi, "w") as fh:
        fh.write("#!/bin/sh\n" + "".join(f"echo '{i}, {f}, 16376, 50'\n" for i, f in enumerate(free_mb)))
    job = os.path.join(tmp, "job.py")
    with open(job, "w") as fh:
        fh.write(JOB)
    for p in (smi, job):
        os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC)
    runs, logs = os.path.join(tmp, "runs"), os.path.join(tmp, "logs")   # created by the queue itself
    for tag, plan in plans.items():
        if plan == [0]:
            continue                                                 # default behaviour: no plan file needed
        os.makedirs(os.path.join(runs, tag), exist_ok=True)
        json.dump(plan, open(os.path.join(runs, tag, "plan"), "w"))
    mem = os.path.join(tmp, "memory.json")
    json.dump(MEM, open(mem, "w"))
    env = {**os.environ, "PATH": bin_dir + os.pathsep + os.environ["PATH"], "Q_RUNS": runs}
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "gpu_queue.py"), "run", "--tags", *plans,
           "--cmd", f"{sys.executable} {job}", "--mem", mem, "--runs_dir", runs, "--logs_dir", logs,
           "--status", os.path.join(tmp, "status.json"), "--poll", "1", "--oom_backoff", "0",
           "--fail_backoff", "0", *extra]
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120), runs


def calls(runs, tag):
    return [json.loads(l) for l in open(os.path.join(runs, tag, "calls"))]


def test_queue_places_runs_by_free_memory():
    tmp = tempfile.mkdtemp()
    try:
        # gpu 0: 3 GB free (too little), gpu 1: 12 GB (batch 2), gpu 2: 8 GB (batch 1 only)
        r, runs = run_queue(tmp, [3000, 12288, 8192], {"a": [0], "b": [0], "c": [0]})
        assert r.returncode == 0, r.stdout + r.stderr
        used = [c for t in "abc" for c in calls(runs, t)]
        assert all(g != "0" for g, _, _ in used), "nothing may start on a GPU without room"
        assert all((g, b, acc) in (("1", 2, 4), ("2", 1, 8)) for g, b, acc in used), used
        st = json.load(open(os.path.join(tmp, "status.json")))
        assert all(v["status"] == "done" for v in st["runs"].values())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_oom_requeues_and_errors_give_up():
    tmp = tempfile.mkdtemp()
    try:
        r, runs = run_queue(tmp, [12288], {"oom": [3, 3, 0], "bad": [1, 1, 1]}, extra=("--max_fail", "2"))
        assert r.returncode == 2, r.stdout + r.stderr
        assert len(calls(runs, "oom")) == 3 and os.path.exists(os.path.join(runs, "oom", "DONE")), \
            "out-of-memory exits must be retried and never count as failures"
        assert len(calls(runs, "bad")) == 2 and "FAILED  bad" in r.stdout
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_gpu_cap_uses_only_the_most_free():
    tmp = tempfile.mkdtemp()
    try:
        # 4 GPUs with room; cap 2 -> only the two with the most free memory (gpu 2, gpu 0) are ever used
        r, runs = run_queue(tmp, [12000, 9000, 15000, 8500], {t: [0] for t in "abcde"}, extra=("--max_parallel", "2"))
        assert r.returncode == 0, r.stdout + r.stderr
        used = {c[0] for t in "abcde" for c in calls(runs, t)}
        assert used <= {"2", "0"}, used
        st = json.load(open(os.path.join(tmp, "status.json")))
        assert st["max_gpus"] == 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_second_queue_is_refused():
    tmp = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmp, "runs"))
        sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            with open(os.path.join(tmp, "queue.lock"), "w") as fh:
                fh.write(str(sleeper.pid))
            r, _ = run_queue(tmp, [12288], {"a": [0]})
            assert r.returncode != 0 and "already running" in (r.stdout + r.stderr)
        finally:
            sleeper.kill()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"all {len(tests)} gpu-queue tests passed")
