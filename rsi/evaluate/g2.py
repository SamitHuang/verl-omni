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
"""Gate g2: zero-diff inside the real verl loop.

Runs ``g2_loop.sh`` (two training steps with a real optimizer update and weight
sync between them) and checks, from evaluator-owned dumps only:

- raw rollout vs actor log-prob mismatch on every step (target 0);
- every weight sync delivers bitwise the tensors the actor sent, to every
  rollout replica;
- weights change between syncs and ``actor/grad_norm`` is nonzero, so step 2
  is measured after a real update.
"""

import glob
import json
import os
import re

import torch

from rsi.evaluate import proc
from rsi.evaluate.compare import compare_logprobs

_HERE = os.path.dirname(os.path.abspath(__file__))
_METRIC_RE = re.compile(r"step:(\d+) - (.*)")
# Validation metrics are logged as numpy scalar reprs, e.g. ``np.float64(0.3)``.
_NP_SCALAR_RE = re.compile(r"^np\.\w+\((.*)\)$")


def _parse_step_metrics(log_path):
    metrics = {}
    with open(log_path, errors="replace") as f:
        for line in f:
            m = _METRIC_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            for kv in m.group(2).split(" - "):
                key, _, val = kv.partition(":")
                val = _NP_SCALAR_RE.sub(r"\1", val.strip())
                try:
                    metrics.setdefault(step, {})[key.strip()] = float(val)
                except ValueError:
                    continue
    return metrics


def _load_digests(dump_dir, kind):
    out = []
    for path in glob.glob(os.path.join(dump_dir, f"weights_{kind}_*.json")):
        with open(path) as f:
            d = json.load(f)
        out.append((d["time_ns"], d["pid"], d["digests"]))
    return sorted(out)


def _weight_checks(dump_dir, steps):
    sent = _load_digests(dump_dir, "sent")
    recv = _load_digests(dump_dir, "recv")
    replicas = sorted({pid for _, pid, _ in recv})
    by_replica = {pid: [d for _, p, d in recv if p == pid] for pid in replicas}
    syncs = []
    for i, (_, _, s) in enumerate(sent):
        per_replica = {}
        for pid in replicas:
            r = by_replica[pid][i] if i < len(by_replica[pid]) else {}
            missing = sorted(set(s) - set(r))
            unequal = sorted(k for k in set(s) & set(r) if s[k] != r[k])
            per_replica[str(pid)] = {
                "missing": len(missing),
                "unequal": len(unequal),
                "examples": (missing + unequal)[:5],
            }
        changed = None
        if i > 0:
            prev = sent[i - 1][2]
            changed = sum(1 for k in s if k in prev and s[k] != prev[k])
        syncs.append({"sync": i, "tensors": len(s), "changed_since_prev": changed, "replicas": per_replica})
    ok = (
        len(sent) >= steps + 1
        and bool(replicas)
        and all(r["missing"] == 0 and r["unequal"] == 0 for s in syncs for r in s["replicas"].values())
        and all(s["changed_since_prev"] for s in syncs[1:])
    )
    return ok, {"name": "weight_sync", "ok": ok, "syncs": syncs, "replicas": len(replicas)}


def analyze(dump_dir, log_path, steps):
    checks, step_results = [], []
    total_mismatch = total_tokens = 0
    diff_max, diff_sum = 0.0, 0.0
    for step in range(1, steps + 1):
        path = os.path.join(dump_dir, f"batch_step{step}.pt")
        if not os.path.exists(path):
            checks.append({"name": f"step{step}_dump", "ok": False})
            return "fail", None, checks
        fields = torch.load(path, weights_only=False)["fields"]
        r = compare_logprobs(fields["rollout_log_probs"], fields["old_log_probs"], fields["response_mask"])
        step_results.append(
            {
                "step": step,
                **{
                    k: r[k] for k in ("mismatch_total", "tokens_compared", "abs_diff_max", "abs_diff_mean", "nonfinite")
                },
            }
        )
        total_mismatch += r["mismatch_total"]
        total_tokens += r["tokens_compared"]
        diff_max = max(diff_max, r["abs_diff_max"])
        diff_sum += r["abs_diff_mean"] * r["tokens_compared"]
    checks.append({"name": "loop_logprobs", "steps": step_results})

    weights_ok, weight_check = _weight_checks(dump_dir, steps)
    checks.append(weight_check)

    step_metrics = _parse_step_metrics(log_path)
    grad_norms = {s: m.get("actor/grad_norm") for s, m in step_metrics.items()}
    update_ok = all(grad_norms.get(s) for s in range(1, steps + 1))
    checks.append({"name": "real_update", "ok": update_ok, "grad_norm": grad_norms})

    metrics = {
        "mismatch_total": total_mismatch,
        "tokens_compared": total_tokens,
        "abs_diff_max": diff_max,
        "abs_diff_mean": diff_sum / total_tokens if total_tokens else 0.0,
        "step_time_s": {s: m.get("timing_s/step") for s, m in step_metrics.items()},
    }
    valid = weights_ok and update_ok and total_tokens > 0
    if not valid:
        status = "invalid"
    else:
        status = "pass" if total_mismatch == 0 else "fail"
    return status, metrics, checks


def run_g2(
    task: dict, worktree: str, out_dir: str, gpus: str, harness: str, steps: int = 2, timeout_s: int = 3600
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    env = dict(os.environ)
    env.update(
        RSI_WORKTREE=worktree,
        RSI_HARNESS=harness,
        RSI_DUMP_DIR=out_dir,
        RSI_CKPT=task["scope"]["checkpoint"],
        RSI_PYTHON=task["start"]["python"],
        RSI_STEPS=str(steps),
        RSI_LORA=json.dumps(task["scope"].get("lora") or False),
        RSI_FAMILY=task["scope"].get("family", "qwen3_omni"),
        CUDA_VISIBLE_DEVICES=gpus,
    )
    log_path = os.path.join(out_dir, "g2.log")
    run_status, rc, wall = proc.run(["bash", os.path.join(_HERE, "g2_loop.sh")], env, log_path, timeout_s)
    result = {"gate": "g2_loop_zero_diff", "wall_clock_s": wall, "returncode": rc, "artifacts": [log_path]}
    if run_status != "ok":
        result.update(status="timeout" if run_status == "timeout" else "error", metrics={}, checks=[])
        return result
    status, metrics, checks = analyze(out_dir, log_path, steps)
    result.update(status=status, metrics=metrics or {}, checks=checks)
    return result
