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
"""Gate g1: offline zero-diff suite and first-divergence localisation.

Replays the step-1 batch and the exact rollout engine call that g2 captured
for the same candidate, through the candidate's actor (``g1_actor``) and a
rebuilt vLLM-Omni engine (``g1_rollout``). Checks:

- ``actor_replay_vs_loop``: the replay reproduces the loop's actor log-probs
  (evaluator integrity; nonzero makes the result invalid);
- ``prefill_vs_actor``, ``loop_decode_vs_actor``: rollout vs actor;
- ``prefill_vs_decode``, ``alone_vs_batch``: vLLM self-consistency;
- ``actor_alone_vs_packed``: actor batch invariance;
- ``cudagraph_prefill_vs_actor`` (acceptance only): the same under CUDA graphs;
- ``first_divergence``: first op whose output differs, for one row.
"""

import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor

import torch

from rsi.evaluate import proc
from rsi.evaluate.common import gate_env
from rsi.evaluate.compare import compare_logprobs
from rsi.evaluate.loop_config import FAMILY_PIPELINE
from rsi.evaluate.probe import first_divergence


def _summary(name, x, y, mask):
    c = compare_logprobs(x, y, mask)
    return {"name": name, **{k: c[k] for k in ("mismatch_total", "tokens_compared", "abs_diff_max", "abs_diff_mean")}}


def replay_step(from_dir):
    """First step g2 found mismatched (where the diagnostics are needed), else 1."""
    path = os.path.join(from_dir, "result.json")
    if not os.path.exists(path):
        return 1
    with open(path) as f:
        g2 = json.load(f)
    loop = next((c for c in g2.get("checks", []) if c.get("name") == "loop_logprobs"), {})
    return next((s["step"] for s in loop.get("steps", []) if s.get("mismatch_total")), 1)


def run_g1(
    task, worktree, out_dir, gpus, harness, from_dir=None, acceptance=False, probe_row=0, timeout_s=2400, step=None
):
    from_dir = from_dir or out_dir
    step = step or replay_step(from_dir)
    fixture = os.path.join(from_dir, f"batch_step{step}.pt")
    engine_args = sorted(glob.glob(os.path.join(from_dir, "engine_args_*.pkl")))
    # Step k runs with the weights of sync k-1; g2 dumps them at sync RSI_STATE_SYNC (1).
    state = os.path.join(from_dir, f"actor_state_sync{step - 1}.pt")
    weights = os.path.join(from_dir, f"weights_sent_sync{step - 1}.pt")
    result = {"gate": "g1_offline_zero_diff", "artifacts": []}
    inputs = [fixture] + ([state, weights] if step > 1 else [])
    if not all(os.path.exists(p) for p in inputs) or not engine_args:
        result.update(
            status="error",
            metrics={},
            checks=[
                {
                    "name": "inputs",
                    "ok": False,
                    "from_dir": from_dir,
                    "step": step,
                    "missing": [p for p in inputs if not os.path.exists(p)],
                }
            ],
        )
        return result
    actor_extra = ["--state", state] if step > 1 else []
    rollout_extra = ["--weights", weights] if step > 1 else []
    rollout_extra += ["--pipeline", FAMILY_PIPELINE[task["scope"].get("family", "qwen3_omni")]]
    os.makedirs(out_dir, exist_ok=True)
    gpu_list = [g for g in gpus.split(",") if g]
    actor_gpu, rollout_gpu = gpu_list[0], gpu_list[-1]
    python = task["start"]["python"]
    actor_out, rollout_out = os.path.join(out_dir, "g1_actor.pt"), os.path.join(out_dir, "g1_rollout.pt")

    actor_job = (
        "actor",
        [
            python,
            "-m",
            "rsi.evaluate.g1_actor",
            "--worktree",
            worktree,
            "--harness",
            harness,
            "--fixture",
            fixture,
            "--ckpt",
            task["scope"]["checkpoint"],
            "--out",
            actor_out,
            "--probe-row",
            str(probe_row),
            "--lora",
            json.dumps(task["scope"].get("lora") or False),
            "--family",
            task["scope"].get("family", "qwen3_omni"),
            *actor_extra,
        ],
        actor_gpu,
    )
    rollout_job = (
        "rollout",
        [
            python,
            "-m",
            "rsi.evaluate.g1_rollout",
            "--engine-args",
            engine_args[0],
            "--fixture",
            fixture,
            "--out",
            rollout_out,
            "--probe-row",
            str(probe_row),
            *rollout_extra,
        ],
        rollout_gpu,
    )
    cudagraph_job = (
        "rollout_cudagraph",
        [
            python,
            "-m",
            "rsi.evaluate.g1_rollout",
            "--engine-args",
            engine_args[0],
            "--fixture",
            fixture,
            "--out",
            os.path.join(out_dir, "g1_rollout_cudagraph.pt"),
            "--cudagraph",
            "--batch-copies",
            "1",
            *rollout_extra,
        ],
        rollout_gpu,
    )
    phases = [[actor_job, rollout_job]] if actor_gpu != rollout_gpu else [[actor_job], [rollout_job]]
    if acceptance:
        phases.append([cudagraph_job])
    # The pinned vLLM-Omni thinker cannot start under torch.compile + CUDA
    # graphs, so that check is reported as unavailable instead of failing.
    optional = {"rollout_cudagraph"}
    unavailable = set()

    wall = 0.0
    for phase in phases:
        with ThreadPoolExecutor(len(phase)) as ex:
            futures = {
                name: ex.submit(
                    proc.run,
                    cmd,
                    gate_env(task, worktree, harness, gpu),
                    os.path.join(out_dir, f"g1_{name}.log"),
                    timeout_s,
                )
                for name, cmd, gpu in phase
            }
            outcomes = {name: f.result() for name, f in futures.items()}
        wall += max(o[2] for o in outcomes.values())
        for name, (status, rc, _) in outcomes.items():
            result["artifacts"].append(os.path.join(out_dir, f"g1_{name}.log"))
            if status != "ok" and name in optional:
                unavailable.add(name)
            elif status != "ok":
                result.update(
                    status="timeout" if status == "timeout" else "error",
                    wall_clock_s=wall,
                    metrics={},
                    checks=[{"name": name, "status": status, "returncode": rc}],
                )
                return result

    fields = torch.load(fixture, weights_only=False)["fields"]
    actor = torch.load(actor_out, weights_only=False)
    rollout = torch.load(rollout_out, weights_only=False)
    mask = fields["response_mask"]
    checks = [
        _summary("actor_replay_vs_loop", actor["loop_grouping"], fields["old_log_probs"], mask),
        _summary("prefill_vs_actor", rollout["prefill_alone"], actor["loop_grouping"], mask),
        _summary("loop_decode_vs_actor", fields["rollout_log_probs"], actor["loop_grouping"], mask),
        _summary("prefill_vs_decode", rollout["prefill_alone"], fields["rollout_log_probs"], mask),
        _summary("actor_alone_vs_packed", actor["alone"], actor["all_packed"], mask),
    ]
    for c, copy in enumerate(rollout["prefill_in_batch"]):
        checks.append(_summary(f"alone_vs_batch_copy{c}", rollout["prefill_alone"], copy, mask))
    if acceptance and "rollout_cudagraph" in unavailable:
        checks.append(
            {
                "name": "cudagraph_prefill_vs_actor",
                "status": "unavailable",
                "log": os.path.join(out_dir, "g1_rollout_cudagraph.log"),
            }
        )
    elif acceptance:
        cg = torch.load(os.path.join(out_dir, "g1_rollout_cudagraph.pt"), weights_only=False)
        checks.append(_summary("cudagraph_prefill_vs_actor", cg["prefill_alone"], actor["loop_grouping"], mask))

    num_layers = task["scope"]["truncate_layers"]
    first, rows = first_divergence(actor.get("probe", {}), rollout.get("probe", {}), num_layers)
    first_row = next((r for r in rows if r["op"] == first), None)
    checks.append({"name": "first_divergence", "row": probe_row, "op": first, "detail": first_row, "ops": rows})

    gated = [c for c in checks if "mismatch_total" in c and c["name"] != "actor_replay_vs_loop"]
    mismatch_total = sum(c["mismatch_total"] for c in gated)
    integrity_ok = checks[0]["mismatch_total"] == 0
    if not integrity_ok:
        status = "invalid"
    else:
        status = "pass" if mismatch_total == 0 else "fail"
    result.update(
        status=status,
        wall_clock_s=wall,
        metrics={
            "mismatch_total": mismatch_total,
            "tokens_compared": sum(c["tokens_compared"] for c in gated),
            "first_divergence": first,
            "replay_step": step,
        },
        checks=checks,
    )
    for path in glob.glob(os.path.join(from_dir, "actor_state_sync*.pt")) + glob.glob(
        os.path.join(from_dir, "weights_sent_sync*.pt")
    ):
        os.remove(path)  # up to a full model each; the digests stay
    return result
