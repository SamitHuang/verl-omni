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
"""Evaluator entry point.

    python -m rsi.evaluate --task T.yaml --worktree W --gate g2_loop_zero_diff --out-dir D --gpus 2,3

Writes ``D/result.json`` following the evaluator contract in ``rsi/DESIGN.md``.
"""

import argparse
import contextlib
import json
import os
import subprocess

import yaml

HARNESS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _candidate(worktree, task):
    from rsi.orchestrator.workspace import Workspace, patch_id

    ws = Workspace(task, worktree)
    out = {
        name: subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", ws.path(name), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        for name in task["start"]["repos"]
    }
    out["patch_id"] = patch_id(ws.diff())
    return out


def _env_info(task, gpus):
    info = {"python": task["start"]["python"], "gpus": gpus}
    first = (gpus or "0").split(",")[0]
    query = subprocess.run(
        ["nvidia-smi", "-i", first, "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    if query.returncode == 0:
        info["gpu_name"], info["compute_cap"], info["driver"] = (s.strip() for s in query.stdout.split(","))
    return info


@contextlib.contextmanager
def env_overlay(env: dict):
    """Set ``env`` in ``os.environ`` for the block, restoring the previous values after."""
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update({k: str(v) for k, v in env.items()})
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _run_gate(task, gate, worktree, out_dir, gpus, harness, from_dir, acceptance, bench_parts=None, bench_profile=None):
    if gate == "g0_static":
        from rsi.evaluate.g0 import run_g0

        return run_g0(task, worktree)
    if gate == "g2_loop_zero_diff":
        from rsi.evaluate.g2 import run_g2

        return run_g2(task, worktree, out_dir, gpus, harness)
    if gate == "g1_offline_zero_diff":
        from rsi.evaluate.g1 import run_g1

        return run_g1(task, worktree, out_dir, gpus, harness, from_dir=from_dir, acceptance=acceptance)
    if gate == "g3_grad":
        from rsi.evaluate.g3 import run_g3

        return run_g3(task, worktree, out_dir, gpus, harness)
    if gate == "bench":
        from rsi.evaluate.bench import run_bench

        return run_bench(task, worktree, out_dir, gpus, harness, from_dir=from_dir, parts=bench_parts)
    if gate == "bench_loop":
        from rsi.evaluate.bench import run_bench_loop

        return run_bench_loop(task, worktree, out_dir, gpus, harness, profile=bench_profile or "loop")
    raise ValueError(f"unknown gate {gate}")


def evaluate(
    task_path,
    worktree,
    gate,
    out_dir,
    gpus,
    harness=HARNESS,
    from_dir=None,
    acceptance=False,
    extra_env=None,
    bench_parts=None,
    bench_profile=None,
):
    with open(task_path) as f:
        task = yaml.safe_load(f)
    os.makedirs(out_dir, exist_ok=True)
    # Opt-in switches of the candidate (e.g. VLLM_BATCH_INVARIANT=1); inherited by every gate subprocess.
    # g0 rejects CANDIDATE_ENV keys outside the task's candidate_env_rules.
    from rsi.orchestrator.workspace import read_env

    candidate_env = {**(task.get("candidate_env") or {}), **read_env(worktree), **(extra_env or {})}
    with env_overlay(candidate_env):
        result = _run_gate(
            task, gate, worktree, out_dir, gpus, harness, from_dir, acceptance, bench_parts, bench_profile
        )
    result.update(
        task_id=task["id"], candidate=_candidate(worktree, task), env=_env_info(task, gpus), candidate_env=candidate_env
    )
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--gpus", default="")
    parser.add_argument(
        "--from-dir", default=None, help="g1, bench: g2 output dir with the captured batch and engine args"
    )
    parser.add_argument("--acceptance", action="store_true", help="g1: also run the CUDA-graph check")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="candidate opt-in env, added to the task's candidate_env",
    )
    parser.add_argument(
        "--bench-parts", default=None, help="bench: comma-separated subset of rollout_eager,rollout_cudagraph,actor"
    )
    parser.add_argument(
        "--bench-profile",
        default=None,
        help="bench_loop: bench.<profile> settings layered on bench.loop (default loop)",
    )
    args = parser.parse_args()
    extra_env = dict(item.split("=", 1) for item in args.env)
    # Gate subprocesses run from "/", so relative paths would resolve elsewhere.
    absolute = {
        k: os.path.abspath(v) if v else v
        for k, v in vars(args).items()
        if k in ("task", "worktree", "out_dir", "from_dir")
    }
    result = evaluate(
        absolute["task"],
        absolute["worktree"],
        args.gate,
        absolute["out_dir"],
        args.gpus,
        from_dir=absolute["from_dir"],
        acceptance=args.acceptance,
        extra_env=extra_env,
        bench_parts=args.bench_parts.split(",") if args.bench_parts else None,
        bench_profile=args.bench_profile,
    )
    print(json.dumps({k: result.get(k) for k in ("gate", "status", "metrics")}, default=str))


if __name__ == "__main__":
    main()
