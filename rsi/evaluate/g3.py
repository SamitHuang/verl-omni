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
"""Gate g3: the candidate's actor gradients are correct.

A 0-diff forward says nothing about the backward: custom autograd wrappers,
detached ops or a wrong scale still pass g1 and g2 (g2 only requires a
nonzero grad norm). On fixed fixture rows and a fixed surrogate loss, g3
compares the candidate's gradients with an fp32 reference and uses stock bf16
training as the yardstick:

- every parameter with a nonzero reference gradient has a nonzero candidate gradient;
- per parameter, ``cos(candidate, ref) >= min(cos(stock, ref) - margin, floor)``;
- the total gradient norm ratio candidate / ref is within ``norm_ratio``.

The reference is candidate-independent and cached per (checkpoint, fixture).

LoRA scopes: at initialization B == 0, so the merged weight equals the base
weight and the full-weight reference gradient G applies. Each pair's expected B
gradient is ``scaling * G @ A^T`` with the candidate's own A; the same
comparison then runs on the B gradients.
"""

import hashlib
import json
import os

import torch

from rsi.evaluate import proc
from rsi.evaluate.common import gate_env

DEFAULTS = {"margin": 0.05, "floor": 0.99, "norm_ratio": [0.8, 1.25]}


def _cache_path(task, fixture):
    h = hashlib.sha256()
    h.update(task["scope"]["checkpoint"].encode())
    with open(fixture, "rb") as f:
        h.update(hashlib.sha256(f.read()).digest())
    with open(os.path.join(os.path.dirname(__file__), "grad_common.py"), "rb") as f:
        h.update(f.read())
    return os.path.join(os.path.dirname(fixture), f"grad_ref_{h.hexdigest()[:12]}.pt")


def compare(stats, cfg):
    """Gate decision from ``grad_common.grad_stats`` output. Returns (ok, checks, metrics)."""
    missing, worse, rows = [], [], []
    for name, s in stats.items():
        if s["cos_c"] is None:
            missing.append(name)
            continue
        bar = min(s["cos_s"] - cfg["margin"], cfg["floor"])
        rows.append({"param": name, "cos_candidate": round(s["cos_c"], 4), "cos_stock": round(s["cos_s"], 4)})
        if s["cos_c"] < bar:
            worse.append(name)
    ref_norm = sum(s["r_norm2"] for s in stats.values()) ** 0.5
    ratio_c = sum(s["c_norm2"] for s in stats.values()) ** 0.5 / ref_norm
    ratio_s = sum(s["s_norm2"] for s in stats.values()) ** 0.5 / ref_norm
    lo, hi = cfg["norm_ratio"]
    rows.sort(key=lambda x: x["cos_candidate"] - x["cos_stock"])
    checks = [
        {"name": "nonzero_grads", "ok": not missing, "missing": missing[:20], "count": len(missing)},
        {"name": "per_param_cosine", "ok": not worse, "worse": worse[:20], "count": len(worse), "worst": rows[:10]},
        {"name": "norm_ratio", "ok": lo <= ratio_c <= hi, "candidate": round(ratio_c, 4), "stock": round(ratio_s, 4)},
    ]
    metrics = {
        "params_checked": len(rows) + len(missing),
        "min_cos_candidate": min((x["cos_candidate"] for x in rows), default=None),
        "min_cos_stock": min((x["cos_stock"] for x in rows), default=None),
        "norm_ratio_candidate": round(ratio_c, 4),
    }
    return all(c["ok"] for c in checks), checks, metrics


def run_g3(task, worktree, out_dir, gpus, harness, fixture=None, timeout_s=1800):
    fixture = fixture or task["grad_gate"]["fixture"]
    cfg = {**DEFAULTS, **{k: v for k, v in (task.get("grad_gate") or {}).items() if k in DEFAULTS}}
    python = task["start"]["python"]
    gpu = gpus.split(",")[0]
    os.makedirs(out_dir, exist_ok=True)
    result = {"gate": "g3_grad", "artifacts": [], "wall_clock_s": 0.0}

    ref_path = _cache_path(task, fixture)
    if not os.path.exists(ref_path):
        env = gate_env(task, worktree, harness, gpu)
        env["PYTHONPATH"] = harness  # stock transformers only; no candidate code
        log = os.path.join(out_dir, "g3_ref.log")
        status, rc, wall = proc.run(
            [
                python,
                "-m",
                "rsi.evaluate.g3_grad_ref",
                "--ckpt",
                task["scope"]["checkpoint"],
                "--fixture",
                fixture,
                "--out",
                ref_path + ".tmp",
                "--family",
                task["scope"].get("family", "qwen3_omni"),
            ],
            env,
            log,
            timeout_s,
        )
        result["wall_clock_s"] += wall
        result["artifacts"].append(log)
        if status != "ok":
            result.update(
                status="error", metrics={}, checks=[{"name": "reference", "status": status, "returncode": rc}]
            )
            return result
        os.replace(ref_path + ".tmp", ref_path)

    cand_path = os.path.join(out_dir, "g3_candidate.pt")
    log = os.path.join(out_dir, "g3_candidate.log")
    status, rc, wall = proc.run(
        [
            python,
            "-m",
            "rsi.evaluate.g3_grad_actor",
            "--worktree",
            worktree,
            "--harness",
            harness,
            "--ckpt",
            task["scope"]["checkpoint"],
            "--fixture",
            fixture,
            "--ref",
            ref_path,
            "--out",
            cand_path,
            "--lora",
            json.dumps(task["scope"].get("lora") or False),
            "--family",
            task["scope"].get("family", "qwen3_omni"),
        ],
        gate_env(task, worktree, harness, gpu),
        log,
        timeout_s,
    )
    result["wall_clock_s"] += wall
    result["artifacts"].append(log)
    if status != "ok":
        result.update(
            status="timeout" if status == "timeout" else "error",
            metrics={},
            checks=[{"name": "candidate", "status": status, "returncode": rc}],
        )
        return result

    cand = torch.load(cand_path, weights_only=False)
    if task["scope"].get("lora"):
        nonzero_b = [n for n, m in cand["lora"].items() if m["B_norm"] != 0]
        if not cand["lora"] or nonzero_b:
            result.update(
                status="error",
                metrics={},
                checks=[
                    {
                        "name": "lora_init",
                        "pairs": len(cand["lora"]),
                        "nonzero_B": nonzero_b[:5],
                        "note": "the LoRA reference assumes B == 0 at initialization",
                    }
                ],
            )
            return result
    ok, checks, metrics = compare(cand["stats"], cfg)
    ref_loss = torch.load(ref_path, weights_only=False, mmap=True)["loss"]
    metrics["loss"] = {"ref": ref_loss["ref"], "stock": ref_loss["stock"], "candidate": cand["loss"]}
    result.update(status="pass" if ok else "fail", metrics=metrics, checks=checks, reference=ref_path)
    return result
