# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run a benchmark gate on saved candidates in parallel GPU groups and tabulate vs stock.

The spec is a JSON list of ``{"label", "task", "iter_dir"[, "stock", "profile"]}``
(``profile``: ``bench_loop`` settings, e.g. ``converge``); ``iter_dir`` is a
run's iteration directory (``patches/`` plus ``confirm/`` or plain g2 output),
or its ``iter_000_base`` for stock. Each candidate gets its own worktree under
``--ws-root``; results go to ``--out/<label>/result.json``.

    python -m rsi.tools.bench_candidates --spec S.json --out D --gpus 0,1,2,3 [--gate bench_loop --gpus-per-job 4]
"""

import argparse
import json
import os
import queue
import subprocess
import sys
import threading

import yaml

from rsi.evaluate.__main__ import HARNESS
from rsi.orchestrator.workspace import Workspace, load_patches


def g2_dir(iter_dir):
    confirm = os.path.join(iter_dir, "confirm", "g2_loop_zero_diff")
    return confirm if os.path.isdir(confirm) else os.path.join(iter_dir, "g2_loop_zero_diff")


def run_one(item, out_root, ws_root, gpus, gate="bench", parts=None):
    """One evaluator process per candidate: ``evaluate`` sets the candidate's env on ``os.environ``."""
    with open(item["task"]) as f:
        task = yaml.safe_load(f)
    iter_dir = os.path.realpath(item["iter_dir"])
    ws = Workspace(task, os.path.join(ws_root, item["label"]))
    ws.reset(load_patches(os.path.join(iter_dir, "patches")))
    out_dir = os.path.join(out_root, item["label"])
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "rsi.evaluate",
        "--task",
        item["task"],
        "--worktree",
        ws.root,
        "--gate",
        gate,
        "--out-dir",
        out_dir,
        "--gpus",
        gpus,
        "--from-dir",
        g2_dir(iter_dir),
    ]
    if parts:
        cmd += ["--bench-parts", parts]
    if item.get("profile"):
        cmd += ["--bench-profile", item["profile"]]
    with open(os.path.join(out_dir, "evaluate.log"), "w") as log:
        subprocess.run(cmd, cwd="/", env={**os.environ, "PYTHONPATH": HARNESS}, stdout=log, stderr=subprocess.STDOUT)
    path = os.path.join(out_dir, "result.json")
    if not os.path.exists(path):
        return {"status": "error"}
    with open(path) as f:
        return json.load(f)


def _fmt(v):
    if v is None:
        return "n/a"
    return f"{v:.0f}" if abs(v) >= 100 else f"{v:.4g}"


def table(spec, out_root):
    rows = []
    for item in spec:
        path = os.path.join(out_root, item["label"], "result.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            r = json.load(f)
        rows.append((item, r.get("status"), r.get("metrics") or {}))
    stock = {item["task"]: m for item, _, m in rows if item.get("stock")}
    keys = sorted({k for _, _, m in rows for k in m})
    lines = ["| candidate | status | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
    for item, status, m in rows:
        base = stock.get(item["task"], {})
        cells = []
        for k in keys:
            v = m.get(k)
            if v is not None and base.get(k) and not item.get("stock"):
                cells.append(f"{_fmt(v)} ({v / base[k]:.0%})")
            else:
                cells.append(_fmt(v))
        lines.append(f"| {item['label']} | {status} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ws-root", default="/home/public/yx/rsi_work/bench_ws", help="one per concurrent driver")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--gpus-per-job", type=int, default=1)
    parser.add_argument("--gate", default="bench", choices=["bench", "bench_loop"])
    parser.add_argument(
        "--parts", default=None, help="bench: comma-separated subset of rollout_eager,rollout_cudagraph,actor"
    )
    parser.add_argument("--table-only", action="store_true")
    args = parser.parse_args()
    with open(args.spec) as f:
        spec = json.load(f)
    os.makedirs(args.out, exist_ok=True)

    if not args.table_only:
        todo = queue.Queue()
        for item in spec:
            todo.put(item)

        def worker(gpus):
            while True:
                try:
                    item = todo.get_nowait()
                except queue.Empty:
                    return
                r = run_one(item, args.out, args.ws_root, gpus, args.gate, args.parts)
                print(f"{item['label']} on GPUs {gpus}: {r.get('status')} {json.dumps(r.get('metrics'))}", flush=True)

        gpu_list = args.gpus.split(",")
        k = args.gpus_per_job
        groups = [",".join(gpu_list[i : i + k]) for i in range(0, len(gpu_list) - k + 1, k)]
        threads = [threading.Thread(target=worker, args=(g,)) for g in groups]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    text = table(spec, args.out)
    with open(os.path.join(args.out, "table.md"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
