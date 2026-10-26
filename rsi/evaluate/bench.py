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
"""Benchmark gates: candidate throughput at full depth (measurement only).

``bench`` uses the engine args and step-1 batch that g2 captured for the same
candidate (``from_dir``) and the task's ``bench`` section:

- ``rollout_eager`` / ``rollout_cudagraph``: decode tokens/s per batch size
  (``bench_rollout``, one GPU); CUDA graphs are reported as unavailable if the
  engine cannot start that way;
- ``actor``: log-prob recompute and forward+backward tokens/s (``bench_actor``),
  under torchrun on ``bench.actor.gpus`` GPUs so FSDP communication counts.

``bench_loop`` runs the real verl loop (``g2_loop.sh`` without the capture
hooks) at full depth with the task's ``bench.loop`` settings and reports step
and phase times from verl's metrics, skipping warmup steps, plus verl's own
rollout-vs-actor probability difference.

Jobs run one at a time so they do not compete for the host.
"""

import json
import os
import statistics

from rsi.evaluate import proc
from rsi.evaluate.common import gate_env
from rsi.evaluate.g1_actor import _free_port
from rsi.evaluate.g2 import _parse_step_metrics
from rsi.evaluate.loop_config import FAMILY_PIPELINE

_HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULTS = {
    "rollout": {"batch_sizes": "32,128", "max_tokens": 512, "repeats": 3, "gpu_memory_utilization": 0.85},
    "actor": {"tokens": 32768, "chunk_tokens": 8192, "repeats": 3, "gpus": 2},
    "loop": {"gpus": 2, "steps": 5, "warmup_steps": 1, "train_batch_size": 32, "n": 8, "overrides": []},
}
PARTS = ("rollout_eager", "rollout_cudagraph", "actor")
LOOP_TIMINGS = (
    "timing_s/step",
    "timing_s/gen",
    "timing_s/old_log_prob",
    "timing_s/update_actor",
    "timing_s/update_weights",
)
LOOP_STATS = ("response_length/mean", "response_length/max", "perf/total_num_tokens", "perf/throughput")


def profile_config(task, part):
    """``bench.<part>`` over the defaults. A loop profile (e.g. ``converge``) is layered on ``bench.loop``."""
    bench = task.get("bench") or {}
    if part in DEFAULTS:
        return {**DEFAULTS[part], **(bench.get(part) or {})}
    if part not in bench:
        raise KeyError(f"task has no bench.{part} profile")
    return {**profile_config(task, "loop"), **bench[part]}


def run_bench(task, worktree, out_dir, gpus, harness, from_dir=None, parts=None, timeout_s=3600):
    cfg = task.get("bench") or {}
    rollout_cfg, actor_cfg = profile_config(task, "rollout"), profile_config(task, "actor")
    scope = task["scope"]
    family = scope.get("family", "qwen3_omni")
    from_dir = from_dir or out_dir
    fixture = os.path.join(from_dir, "batch_step1.pt")
    engine_args = (
        sorted(f for f in os.listdir(from_dir) if f.startswith("engine_args_")) if os.path.isdir(from_dir) else []
    )
    result = {"gate": "bench", "artifacts": []}
    gpu_list = [g for g in gpus.split(",") if g]
    n_actor = int(actor_cfg["gpus"])
    n_needed = n_actor if "actor" in (parts or PARTS) else 1
    if not os.path.exists(fixture) or not engine_args or len(gpu_list) < n_needed:
        result.update(
            status="error", metrics={}, checks=[{"name": "inputs", "ok": False, "from_dir": from_dir, "gpus": gpus}]
        )
        return result
    os.makedirs(out_dir, exist_ok=True)
    python = task["start"]["python"]
    rollout_cmd = [
        python,
        "-m",
        "rsi.evaluate.bench_rollout",
        "--engine-args",
        os.path.join(from_dir, engine_args[0]),
        "--fixture",
        fixture,
        "--ckpt",
        cfg.get("rollout_checkpoint", scope["checkpoint"]),
        "--batch-sizes",
        str(rollout_cfg["batch_sizes"]),
        "--max-tokens",
        str(rollout_cfg["max_tokens"]),
        "--repeats",
        str(rollout_cfg["repeats"]),
        "--gpu-memory-utilization",
        str(rollout_cfg["gpu_memory_utilization"]),
        "--pipeline",
        FAMILY_PIPELINE[family],
    ]
    launcher = [
        python,
        "-m",
        "torch.distributed.run",
        "--nproc-per-node",
        str(n_actor),
        "--master-port",
        str(_free_port()),
    ]
    jobs = {
        "rollout_eager": ([*rollout_cmd, "--out", os.path.join(out_dir, "rollout_eager.json")], gpu_list[0]),
        "rollout_cudagraph": (
            [*rollout_cmd, "--cudagraph", "--out", os.path.join(out_dir, "rollout_cudagraph.json")],
            gpu_list[0],
        ),
        "actor": (
            [
                *launcher,
                "-m",
                "rsi.evaluate.bench_actor",
                "--worktree",
                worktree,
                "--harness",
                harness,
                "--ckpt",
                cfg.get("actor_checkpoint", scope["checkpoint"]),
                "--fixture",
                fixture,
                "--tokens",
                str(actor_cfg["tokens"]),
                "--chunk-tokens",
                str(actor_cfg["chunk_tokens"]),
                "--repeats",
                str(actor_cfg["repeats"]),
                "--lora",
                json.dumps(scope.get("lora") or False),
                "--family",
                family,
                "--out",
                os.path.join(out_dir, "actor.json"),
            ],
            ",".join(gpu_list[:n_actor]),
        ),
    }
    optional = {"rollout_cudagraph"}
    checks, metrics, spread, wall = [], {}, {}, 0.0
    for name in parts or PARTS:
        cmd, job_gpus = jobs[name]
        log_path = os.path.join(out_dir, f"bench_{name}.log")
        result["artifacts"].append(log_path)
        # cwd goes first on sys.path under `python -m`; a cwd holding another verl_omni would shadow the candidate's.
        status, rc, seconds = proc.run(
            cmd, gate_env(task, worktree, harness, job_gpus), log_path, timeout_s, cwd=worktree
        )
        wall += seconds
        out_path = os.path.join(out_dir, f"{name}.json")
        if status != "ok" or not os.path.exists(out_path):
            if name in optional:
                checks.append({"name": name, "status": "unavailable", "log": log_path})
                continue
            result.update(
                status="timeout" if status == "timeout" else "error",
                wall_clock_s=wall,
                metrics=metrics,
                checks=[*checks, {"name": name, "status": status, "returncode": rc, "log": log_path}],
            )
            return result
        with open(out_path) as f:
            data = json.load(f)
        checks.append({"name": name, "status": "ok", **data})
        if name.startswith("rollout"):
            rows = {f"{name}_tok_s_bs{bs}": row for bs, row in data["batches"].items()}
        else:
            rows = {"actor_logprob_tok_s": data["logprob"], "actor_fwd_bwd_tok_s": data["fwd_bwd"]}
        for key, row in rows.items():
            metrics[key] = row["tok_s"]
            spread[key] = row["spread"]
    # spread: (max - min) / median wall time over the repeats, the noise model's input.
    result.update(status="pass", wall_clock_s=wall, metrics=metrics, spread=spread, checks=checks)
    return result


def loop_overrides(task, n_gpus, profile="loop"):
    loop = profile_config(task, profile)
    bench = task.get("bench") or {}
    batch = int(loop["train_batch_size"])
    return [
        f"actor_rollout_ref.model.path={bench.get('actor_checkpoint', task['scope']['checkpoint'])}",
        f"data.train_batch_size={batch}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={batch}",
        f"actor_rollout_ref.rollout.n={int(loop['n'])}",
        f"trainer.n_gpus_per_node={n_gpus}",
        f"trainer.total_training_steps={int(loop['steps'])}",
        f"trainer.experiment_name=bench_{profile}",
        # A YAML alias to another profile's list arrives as a nested list.
        *(o for item in loop["overrides"] for o in (item if isinstance(item, list) else [item])),
    ]


def summarize_loop(step_metrics, warmup_steps, window=10):
    steps = sorted(s for s in step_metrics if s > warmup_steps)
    out = {"steps_measured": steps}
    for key in LOOP_TIMINGS + LOOP_STATS:
        values = [step_metrics[s][key] for s in steps if key in step_metrics[s]]
        if values:
            out[key] = statistics.median(values)
    diffs = [step_metrics[s].get("training/rollout_probs_diff_max") for s in sorted(step_metrics)]
    out["rollout_probs_diff_max"] = max((d for d in diffs if d is not None), default=None)
    scores = [
        step_metrics[s]["critic/score/mean"] for s in sorted(step_metrics) if "critic/score/mean" in step_metrics[s]
    ]
    if scores:
        out["score_first"] = statistics.mean(scores[:window])
        out["score_last"] = statistics.mean(scores[-window:])
    val = {s: v for s, m in step_metrics.items() for k, v in m.items() if k.startswith("val-core/") and "/mean@" in k}
    if val:
        out["val_first"], out["val_last"] = val[min(val)], val[max(val)]
    return out


def run_bench_loop(task, worktree, out_dir, gpus, harness, profile="loop"):
    loop = profile_config(task, profile)
    timeout_s = int(loop.get("timeout_s", 7200))
    gpu_list = [g for g in gpus.split(",") if g]
    n_gpus = int(loop["gpus"])
    result = {"gate": "bench_loop", "profile": profile, "artifacts": []}
    if len(gpu_list) < n_gpus:
        result.update(status="error", metrics={}, checks=[{"name": "gpus", "ok": False, "need": n_gpus, "got": gpus}])
        return result
    os.makedirs(out_dir, exist_ok=True)
    env = dict(os.environ)
    env.update(
        RSI_WORKTREE=worktree,
        RSI_HARNESS=harness,
        RSI_DUMP_DIR=out_dir,
        RSI_HOOKS="0",
        RSI_CKPT=task["scope"]["checkpoint"],
        RSI_PYTHON=task["start"]["python"],
        RSI_LORA=json.dumps(task["scope"].get("lora") or False),
        RSI_FAMILY=task["scope"].get("family", "qwen3_omni"),
        CUDA_VISIBLE_DEVICES=",".join(gpu_list[:n_gpus]),
    )
    log_path = os.path.join(out_dir, f"bench_{profile}.log")
    result["artifacts"].append(log_path)
    status, rc, wall = proc.run(
        ["bash", os.path.join(_HERE, "g2_loop.sh"), *loop_overrides(task, n_gpus, profile)], env, log_path, timeout_s
    )
    step_metrics = _parse_step_metrics(log_path)
    summary = summarize_loop(step_metrics, int(loop["warmup_steps"]))
    complete = status == "ok" and len(summary["steps_measured"]) == int(loop["steps"]) - int(loop["warmup_steps"])
    metrics = {k.replace("timing_s/", "").replace("/", "_"): v for k, v in summary.items() if k != "steps_measured"}
    result.update(
        status="pass" if complete else ("timeout" if status == "timeout" else "error"),
        returncode=rc,
        wall_clock_s=wall,
        metrics=metrics,
        checks=[{"name": "steps", "measured": summary["steps_measured"], "per_step": step_metrics}],
    )
    return result
