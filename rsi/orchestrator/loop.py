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
"""RSI orchestrator loop.

    python -m rsi.orchestrator.loop --task T.yaml --run-dir R --agent noop --iterations 1 --gpus 0,1,2,3

Per iteration: reset the workspace to the best candidate so far, render the
brief, run the agent, save its patches, evaluate g0 -> g2 -> g1, append to
``ledger.jsonl`` and update the best candidate. A solved candidate is re-run
from a clean workspace before it is reported. Writes ``report.md`` at the end.
"""

import argparse
import hashlib
import json
import os
import shutil
import time

import yaml

from rsi.evaluate.__main__ import HARNESS, evaluate
from rsi.evaluate.bench import profile_config
from rsi.evaluate.probe import op_order
from rsi.orchestrator import brief as brief_mod
from rsi.orchestrator import objective
from rsi.orchestrator.agents import BRIEF_NAME, HYPOTHESIS_NAME, make_agent
from rsi.orchestrator.audit import audit_log
from rsi.orchestrator.gpus import lease
from rsi.orchestrator.workspace import ENV_FILE, Workspace, load_patches, patch_id, save_patches

GATE_GPUS = {"g0_static": 0, "g3_grad": 1, "g2_loop_zero_diff": 2, "g1_offline_zero_diff": 2}
RESULT_GATES = (*GATE_GPUS, "bench", "bench_stock", "bench_loop")


def _gate_gpus(task, gate):
    if gate == "bench":
        return max(1, int(profile_config(task, "actor")["gpus"]))
    if gate == "bench_loop":
        return int(profile_config(task, "loop")["gpus"])
    return GATE_GPUS[gate]


def _run_gate(task_path, task, workspace_root, gate, gate_dir, gpu_pool, lock_dir, **kw):
    """Run one gate on leased GPUs; return ``(result, gpu_seconds)``."""
    n = _gate_gpus(task, gate)
    if n == 0:
        return evaluate(task_path, workspace_root, gate, gate_dir, "", HARNESS, **kw), 0.0
    with lease(gpu_pool, n, lock_dir) as gpus:
        r = evaluate(task_path, workspace_root, gate, gate_dir, gpus, HARNESS, **kw)
    return r, n * float(r.get("wall_clock_s") or 0)


def progress(results, num_layers):
    """Ordering key, larger is better: g2 pass, later first mismatched step, later first divergence,
    lower loop mismatch fraction.

    The step comes before the depth: g1 probes the first mismatched step, so a
    shallow divergence at step 2 means step 1 is already bitwise equal.

    The fraction is rounded to 1% because rollout sampling is not reproducible
    across runs (async scheduling), which moves the count by a few tokens.
    """
    g2 = results.get("g2_loop_zero_diff") or {}
    g1 = results.get("g1_offline_zero_diff") or {}
    if g2.get("status") not in ("pass", "fail"):
        return None
    order = op_order(num_layers)
    first = (g1.get("metrics") or {}).get("first_divergence")
    depth = (
        len(order)
        if first is None and g1.get("status") in ("pass", "fail")
        else (order.index(first) if first in order else -1)
    )
    m = g2["metrics"]
    fraction = round(m["mismatch_total"] / max(m["tokens_compared"], 1), 2)
    step = (g1.get("metrics") or {}).get("replay_step", 1)
    return (g2["status"] == "pass", step, depth, -fraction)


def _paired_bench(task_path, task, workspace_root, out_dir, gpu_pool, lock_dir, stock):
    """Bench the candidate, then stock, back to back on the same GPUs; return ``(candidate, stock, gpu_seconds)``.

    Machine speed drifts (stock CUDA-graph decode moved 16% between two sessions), so a
    ratio is only meaningful against a stock measurement taken right next to it.
    """
    n = _gate_gpus(task, "bench")
    with lease(gpu_pool, n, lock_dir) as gpus:
        cand = evaluate(
            task_path,
            workspace_root,
            "bench",
            os.path.join(out_dir, "bench"),
            gpus,
            HARNESS,
            from_dir=os.path.join(out_dir, "g2_loop_zero_diff"),
        )
        ref = evaluate(
            task_path,
            stock["root"],
            "bench",
            os.path.join(out_dir, "bench_stock"),
            gpus,
            HARNESS,
            from_dir=stock["from_dir"],
        )
    return cand, ref, n * (float(cand.get("wall_clock_s") or 0) + float(ref.get("wall_clock_s") or 0))


def evaluate_candidate(
    task_path, task, workspace_root, out_dir, gpu_pool, lock_dir, acceptance_on_pass=True, all_gates=False, stock=None
):
    """``all_gates`` keeps going after a g3 failure (used for the baseline, where every metric is wanted).

    ``stock`` (maximize objective: the stock worktree root and its g2 capture) also measures
    throughput, paired with stock, once every correctness gate passes.
    """
    results, gpu_seconds = {}, 0.0

    def run_gate(gate, **kw):
        nonlocal gpu_seconds
        r, gs = _run_gate(task_path, task, workspace_root, gate, os.path.join(out_dir, gate), gpu_pool, lock_dir, **kw)
        gpu_seconds += gs
        results[gate] = r
        return r

    if run_gate("g0_static")["status"] != "pass":
        return results, gpu_seconds, "invalid"
    # Wrong gradients invalidate a candidate whatever its forward does; cheap, so it runs before g2.
    if "g3_grad" in task["gates"]:
        g3 = run_gate("g3_grad")
        if g3["status"] != "pass" and not all_gates:
            return results, gpu_seconds, "invalid" if g3["status"] == "fail" else "error"
    g2 = run_gate("g2_loop_zero_diff")
    if g2["status"] not in ("pass", "fail"):
        return results, gpu_seconds, "invalid" if g2["status"] == "invalid" else "error"
    g1 = run_gate(
        "g1_offline_zero_diff",
        from_dir=os.path.join(out_dir, "g2_loop_zero_diff"),
        acceptance=acceptance_on_pass and g2["status"] == "pass",
    )
    if g1["status"] == "invalid":
        return results, gpu_seconds, "invalid"
    solved = (
        g2["status"] == "pass"
        and g1["status"] == "pass"
        and results.get("g3_grad", {"status": "pass"})["status"] == "pass"
    )
    if solved and stock:
        results["bench"], results["bench_stock"], gs = _paired_bench(
            task_path, task, workspace_root, out_dir, gpu_pool, lock_dir, stock
        )
        gpu_seconds += gs
        if results["bench"]["status"] != "pass" or results["bench_stock"]["status"] != "pass":
            return results, gpu_seconds, "error"
    return results, gpu_seconds, "solved" if solved else "evaluated"


def _stock_reference(task_path, task, stock_ws, out_dir, gpu_pool, lock_dir):
    """Session setup for maximize tasks: a stock worktree, its g2 capture (bench inputs) and its full-depth loop.

    The loop's phase times go into the brief; throughput ratios use the paired stock bench.
    """
    stock_ws.reset({})
    results, gpu_seconds = {}, 0.0
    for gate in ("g2_loop_zero_diff", "bench_loop"):
        results[gate], gs = _run_gate(
            task_path, task, stock_ws.root, gate, os.path.join(out_dir, gate), gpu_pool, lock_dir
        )
        gpu_seconds += gs
    if results["bench_loop"]["status"] != "pass" or results["g2_loop_zero_diff"]["status"] not in ("pass", "fail"):
        raise SystemExit(f"stock reference failed; see {out_dir}")
    results["root"], results["from_dir"] = stock_ws.root, os.path.join(out_dir, "g2_loop_zero_diff")
    return results, gpu_seconds


def _reuse_stock_reference(src, stock_ws):
    """An earlier session's stock reference (same task and pins); the paired benches still run fresh."""
    results = {}
    for gate in ("g2_loop_zero_diff", "bench_loop"):
        path = os.path.join(src, gate, "result.json")
        if not os.path.exists(path):
            raise SystemExit(f"--stock-from {src} has no {gate} result")
        results[gate] = json.load(open(path))
    if results["bench_loop"]["status"] != "pass" or results["g2_loop_zero_diff"]["status"] not in ("pass", "fail"):
        raise SystemExit(f"--stock-from {src} is not a usable stock reference")
    stock_ws.reset({})
    results["root"], results["from_dir"] = stock_ws.root, os.path.join(src, "g2_loop_zero_diff")
    return results


def _resume(src, run_dir, ledger_path, start_patches, rank):
    """Continue ``src``: copy its ledger (backfilling rejection reasons), link its iteration dirs, and
    restore the best candidate.

    Returns ``(best, gpu_seconds_total, next_iteration)``. Wall clock restarts; GPU-hours carry over.
    """
    entries = _read_ledger(os.path.join(src, "ledger.jsonl"))
    if not entries:
        raise SystemExit(f"nothing to resume in {src}")
    best = None
    with open(ledger_path, "w") as f:
        for e in entries:
            name = "iter_000_base" if e["iteration"] == 0 else f"iter_{e['iteration']:03d}"
            src_dir = os.path.abspath(os.path.join(src, name))
            results = {
                g: json.load(open(os.path.join(src_dir, g, "result.json")))
                for g in RESULT_GATES
                if os.path.exists(os.path.join(src_dir, g, "result.json"))
            }
            if e["verdict"] in ("invalid", "error", "unconfirmed") and "rejection" not in e:
                e["rejection"] = _rejection(results)
            e["resumed_from"] = src
            if e["verdict"] in ("baseline", "improved", "solved"):
                patches = start_patches if e["iteration"] == 0 else load_patches(os.path.join(src_dir, "patches"))
                best = {"id": e["candidate"], "patches": patches, "results": results, "key": rank(results)}
            dst = os.path.join(run_dir, name)
            if os.path.isdir(src_dir) and not os.path.lexists(dst):
                os.symlink(src_dir, dst)
            f.write(json.dumps(e, default=str) + "\n")
    return best, entries[-1]["gpu_hours_total"] * 3600, entries[-1]["iteration"] + 1


def _confirm(task_path, task, workspace_root, out_dir, gpu_pool, lock_dir, key, primary):
    """Full-depth loop on the candidate in ``workspace_root``; returns ``(result, verdict, gpu_seconds)``.

    The gates run a truncated model; the gain must hold, still bitwise, in the full-depth loop.
    """
    out = os.path.join(out_dir, "bench_loop")
    r, gs = _run_gate(task_path, task, workspace_root, "bench_loop", out, gpu_pool, lock_dir)
    if r["status"] != "pass" or (r.get("metrics") or {}).get("rollout_probs_diff_max") != 0.0:
        verdict = "unconfirmed"
    else:
        verdict = "solved" if objective.solved(key, primary) else "improved"
    return r, verdict, gs


def _unconfirmed_at_target(entries, primary):
    """Earlier iterations that reached the target but were never confirmed, best first."""
    hits = [
        e
        for e in entries
        if e["verdict"] in ("neutral", "regressed") and objective.solved(tuple(e["progress"]), primary)
    ]
    return sorted(hits, key=lambda e: e["progress"][1], reverse=True)


def _rejection(results):
    """Why the first non-passing gate rejected a candidate, in a form the next brief can show."""
    for gate in ("g0_static", "g3_grad", "g2_loop_zero_diff", "g1_offline_zero_diff"):
        r = results.get(gate)
        if not r or r.get("status") == "pass":
            continue
        if gate == "g0_static":
            details = {k: r.get(k) for k in ("scope_violations", "integrity_flags", "env_violations") if r.get(k)}
        else:
            details = [c for c in r.get("checks", []) if c.get("ok") is False or c.get("status") not in (None, "ok")]
        return {"gate": gate, "status": r.get("status"), "details": details or r.get("checks", [])[:3]}
    return None


def _tree_hash(root):
    """Hash of every .py/.sh/.yaml file under ``root``: identifies the gate code a run was judged by."""
    h = hashlib.sha256()
    for dirpath, dirnames, filenames in sorted(os.walk(root)):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(filenames):
            if name.endswith((".py", ".sh", ".yaml")):
                path = os.path.join(dirpath, name)
                h.update(os.path.relpath(path, root).encode())
                h.update(open(path, "rb").read())
    return h.hexdigest()[:16]


def _summary(results):
    return {
        gate: {"status": r.get("status"), "metrics": r.get("metrics"), "wall_clock_s": r.get("wall_clock_s")}
        for gate, r in results.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--agent", default="noop")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--lock-dir", default="/home/public/yx/rsi_work/locks")
    parser.add_argument("--kb", default=None, help="knowledge-base markdown file shown to the agent")
    parser.add_argument(
        "--toolset", default=None, help="directory copied into the workspace as rsi_tools/ each iteration"
    )
    parser.add_argument("--baseline-from", default=None, help="iteration dir to reuse as the evaluated baseline")
    parser.add_argument(
        "--resume-from",
        default=None,
        help="earlier run dir to continue (ledger, best candidate, GPU-hours, numbering), e.g. after a harness fix",
    )
    parser.add_argument(
        "--stock-from",
        default=None,
        help="maximize tasks: earlier stock_<time>/ dir to reuse instead of re-measuring stock",
    )
    args = parser.parse_args()
    # Gate subprocesses run from "/", so relative paths would resolve elsewhere.
    for name in ("task", "run_dir", "kb", "toolset", "baseline_from", "resume_from", "stock_from"):
        if getattr(args, name):
            setattr(args, name, os.path.abspath(getattr(args, name)))

    with open(args.task) as f:
        task = yaml.safe_load(f)
    budget = task["budget"]
    agent_cfg = task.get("agent") or {}
    iterations = budget["iterations"] if args.iterations is None else args.iterations
    gpu_pool = [int(g) for g in args.gpus.split(",")]
    num_layers = task["scope"]["truncate_layers"]
    os.makedirs(args.run_dir, exist_ok=True)
    shutil.copy(args.task, os.path.join(args.run_dir, "task.yaml"))
    ledger_path = os.path.join(args.run_dir, "ledger.jsonl")
    # The workspace lives outside the run dir: the agent user may read its workspace but not the ledger or gate outputs.
    ws_parent = agent_cfg.get("workspace_root")
    ws_root = (
        os.path.join(ws_parent, os.path.basename(os.path.abspath(args.run_dir)))
        if ws_parent
        else os.path.join(args.run_dir, "workspace")
    )
    ws = Workspace(task, ws_root)
    agent = make_agent(args.agent)
    # The agent needs the model and data for its own experiments; a root-only file sends it looking for copies.
    unreadable = agent.unreadable(
        [task["scope"]["checkpoint"], task["start"]["python"], *agent_cfg.get("readable", [])]
    )
    if unreadable:
        raise SystemExit(f"agent user cannot read {len(unreadable)} input(s), e.g. {unreadable[:5]}")
    # Other runs' workspaces and gate outputs hold solutions (other conditions, the reference fix).
    foreign = [os.path.dirname(os.path.abspath(args.run_dir))]
    if ws_parent:
        foreign += [ws_parent] + [
            os.path.join(ws_parent, d) for d in os.listdir(ws_parent) if os.path.join(ws_parent, d) != ws_root
        ]
    exposed = agent.readable(foreign)
    if exposed:
        raise SystemExit(f"agent user can read {len(exposed)} other run path(s), e.g. {exposed[:5]}; lock them first")
    agent.reset_home()
    kb_text = open(args.kb).read() if args.kb else ""
    with open(os.path.join(args.run_dir, "run_config.json"), "w") as f:
        json.dump(
            {
                "agent": args.agent,
                "kb": args.kb,
                "toolset": args.toolset,
                "workspace": ws_root,
                "gpus": args.gpus,
                "stock_from": args.stock_from,
                "baseline_from": args.baseline_from,
                "harness": os.path.realpath(os.path.join(HARNESS, "rsi")),
                "harness_hash": _tree_hash(os.path.join(HARNESS, "rsi")),
            },
            f,
            indent=2,
        )

    start = time.time()
    gpu_seconds_total = 0.0
    # A task may start from an earlier solution (e.g. LoRA on top of the full-parameter 0-diff patches).
    start_patches = load_patches(task["start"].get("patches"))
    if task["start"].get("patches") and not start_patches:
        raise SystemExit(f"start.patches {task['start']['patches']!r} has no *.diff or {ENV_FILE}")
    best = {"id": patch_id(start_patches), "patches": start_patches, "results": None, "key": None}

    primary = task["objective"]["primary"]
    maximize = primary.get("type") == "maximize"
    constraint_gates = [g for g in task["gates"] if g in objective.CONSTRAINT_GATES]
    stock, stock_gs = None, 0.0
    if maximize:
        # Under the run dir, which the agent user cannot read (only its own workspace is exposed).
        stock_ws = Workspace(task, os.path.join(args.run_dir, "stock_workspace"))
        if args.stock_from:
            stock = _reuse_stock_reference(args.stock_from, stock_ws)
        else:
            stock, stock_gs = _stock_reference(
                args.task,
                task,
                stock_ws,
                os.path.join(args.run_dir, time.strftime("stock_%Y%m%dT%H%M%S")),
                gpu_pool,
                args.lock_dir,
            )

    def rank(results):
        if maximize:
            return objective.key(results, results.get("bench"), results.get("bench_stock"), primary, constraint_gates)
        return progress(results, num_layers)

    def gain_needed(*results):
        return objective.threshold(
            primary, *((r.get(b) or {}).get("spread") for r in results if r for b in ("bench", "bench_stock"))
        )

    base_dir = os.path.join(args.run_dir, "iter_000_base")
    first_iteration = 1
    if args.resume_from:
        best, gpu_seconds_total, first_iteration = _resume(
            args.resume_from, args.run_dir, ledger_path, start_patches, rank
        )
    elif args.baseline_from:
        shutil.copytree(args.baseline_from, base_dir, dirs_exist_ok=True)
        results = {
            g: json.load(open(os.path.join(base_dir, g, "result.json")))
            for g in RESULT_GATES
            if os.path.exists(os.path.join(base_dir, g, "result.json"))
        }
        verdict = "baseline"
    else:
        ws.reset(start_patches)
        results, gs, _ = evaluate_candidate(
            args.task, task, ws.root, base_dir, gpu_pool, args.lock_dir, all_gates=True, stock=stock
        )
        gpu_seconds_total += gs
        if maximize and objective.constraints_ok(results, constraint_gates):
            # The brief's phase table compares the parent's full-depth loop with stock's.
            results["bench_loop"], gs = _run_gate(
                args.task, task, ws.root, "bench_loop", os.path.join(base_dir, "bench_loop"), gpu_pool, args.lock_dir
            )
            gpu_seconds_total += gs
        verdict = "baseline"
    gpu_seconds_total += stock_gs
    if not args.resume_from:
        best.update(results=results, key=rank(results))
        _append(ledger_path, 0, "baseline", best["id"], "", results, verdict, gpu_seconds_total, best["key"])

    done = False
    if args.resume_from and maximize:
        # A harness fix may have changed which earlier candidates count; confirm those before new agent turns.
        for e in _unconfirmed_at_target(_read_ledger(ledger_path), primary):
            src_dir = os.path.join(args.run_dir, f"iter_{e['iteration']:03d}")
            out_dir = os.path.join(args.run_dir, f"iter_{first_iteration:03d}")
            results = {}
            for g in RESULT_GATES:
                if g != "bench_loop" and os.path.exists(os.path.join(src_dir, g, "result.json")):
                    os.makedirs(os.path.join(out_dir, g), exist_ok=True)
                    shutil.copy(os.path.join(src_dir, g, "result.json"), os.path.join(out_dir, g, "result.json"))
                    results[g] = json.load(open(os.path.join(src_dir, g, "result.json")))
            patches = load_patches(os.path.join(src_dir, "patches"))
            save_patches(patches, os.path.join(out_dir, "patches"))
            _clear_root(ws)
            ws.reset(patches)
            key = tuple(e["progress"])
            results["bench_loop"], verdict, gs = _confirm(
                args.task, task, ws.root, out_dir, gpu_pool, args.lock_dir, key, primary
            )
            gpu_seconds_total += gs
            _append(
                ledger_path,
                first_iteration,
                "confirm",
                e["candidate"],
                f"confirm iteration {e['iteration']}",
                results,
                verdict,
                gpu_seconds_total,
                key,
                extra={"confirms": e["iteration"]},
            )
            first_iteration += 1
            if verdict == "solved":
                best = {"id": e["candidate"], "patches": patches, "results": results, "key": key}
                done = True
                break

    for i in range(first_iteration, iterations + 1):
        over_budget = (
            gpu_seconds_total / 3600 >= budget["gpu_hours"] or (time.time() - start) / 3600 >= budget["wall_clock_h"]
        )
        if done or over_budget:
            break
        it_dir = os.path.join(args.run_dir, f"iter_{i:03d}")
        os.makedirs(it_dir, exist_ok=True)
        _clear_root(ws)
        ws.reset(best["patches"])
        if args.toolset:
            shutil.copytree(args.toolset, os.path.join(ws.root, "rsi_tools"))
        recent = _read_ledger(ledger_path)[-8:]
        if recent and recent[-1].get("rejection"):
            # The workspace restarts from the parent; without these the agent cannot see what was rejected.
            prev = os.path.join(args.run_dir, f"iter_{recent[-1]['iteration']:03d}", "patches")
            if os.path.isdir(prev):
                shutil.copytree(prev, os.path.join(ws.root, "rsi_previous_candidate"))
        n_agent_gpus = int(agent_cfg.get("gpus", 1))
        with open(os.path.join(ws.root, BRIEF_NAME), "w") as f:
            f.write(
                brief_mod.render(
                    task,
                    i,
                    best["id"],
                    best["results"],
                    recent,
                    kb_text,
                    n_agent_gpus,
                    bool(args.toolset),
                    stock=stock,
                    gain_needed=gain_needed(best["results"]) if maximize else None,
                )
            )
        shutil.copy(os.path.join(ws.root, BRIEF_NAME), os.path.join(it_dir, BRIEF_NAME))
        hyp_path = os.path.join(ws.root, HYPOTHESIS_NAME)

        agent_log = os.path.join(it_dir, "agent.log")
        agent_start = time.time()
        with lease(gpu_pool, n_agent_gpus, args.lock_dir) as agent_gpus:
            agent_status = agent.run(
                ws.root, agent_log, budget["iteration_timeout_min"] * 60, env={"CUDA_VISIBLE_DEVICES": agent_gpus}
            )
        agent_seconds = time.time() - agent_start
        gpu_seconds_total += n_agent_gpus * agent_seconds
        audit = audit_log(agent_log, [tuple(r) for r in (task.get("audit") or {}).get("flag", [])])
        hypothesis = open(hyp_path).read().strip() if os.path.exists(hyp_path) else ""
        patches = ws.diff()
        save_patches(patches, os.path.join(it_dir, "patches"))
        pid = patch_id(patches)

        results, gs, outcome = evaluate_candidate(
            args.task, task, ws.root, it_dir, gpu_pool, args.lock_dir, stock=stock
        )
        gpu_seconds_total += gs
        key = rank(results)
        if outcome in ("invalid", "error") or key is None:
            verdict = outcome if outcome in ("invalid", "error") else "error"
        elif maximize:
            verdict = objective.verdict(key, best["key"], gain_needed(results, best["results"]))
        elif best["key"] is None or key > best["key"]:
            verdict = "improved"
        elif key == best["key"]:
            verdict = "neutral"
        else:
            verdict = "regressed"

        if maximize and objective.needs_confirmation(verdict, key, primary):
            results["bench_loop"], verdict, gs2 = _confirm(
                args.task, task, ws.root, it_dir, gpu_pool, args.lock_dir, key, primary
            )
            gpu_seconds_total += gs2
        elif outcome == "solved" and not maximize:
            ws.reset(load_patches(os.path.join(it_dir, "patches")))
            confirm, gs2, outcome2 = evaluate_candidate(
                args.task, task, ws.root, os.path.join(it_dir, "confirm"), gpu_pool, args.lock_dir
            )
            gpu_seconds_total += gs2
            verdict = "solved" if outcome2 == "solved" else "unconfirmed"

        if verdict in ("improved", "solved"):
            best = {"id": pid, "patches": patches, "results": results, "key": key}
        _append(
            ledger_path,
            i,
            f"{agent.name}:{agent_status}",
            pid,
            hypothesis,
            results,
            verdict,
            gpu_seconds_total,
            key,
            extra={
                "agent_seconds": round(agent_seconds),
                "audit": audit,
                **({"rejection": _rejection(results)} if verdict in ("invalid", "error", "unconfirmed") else {}),
            },
        )
        if verdict == "solved":
            break

    # Leave nothing a later run's agent could read: the candidates are saved under the run dir.
    _clear_root(ws)
    ws.reset()
    _report(args.run_dir, task, _read_ledger(ledger_path), best, gpu_seconds_total, time.time() - start)
    agent.lock(ws.root)


def _clear_root(ws):
    """Remove everything at the workspace root except the repo worktrees (briefs, tools, agent notes)."""
    if not os.path.isdir(ws.root):
        return
    for name in os.listdir(ws.root):
        if name in ws.repos:
            continue
        path = os.path.join(ws.root, name)
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def _append(path, iteration, agent, pid, hypothesis, results, verdict, gpu_seconds_total, key, extra=None):
    g2 = results.get("g2_loop_zero_diff") or {}
    g1 = results.get("g1_offline_zero_diff") or {}
    entry = {
        "iteration": iteration,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "agent": agent,
        "candidate": pid,
        "hypothesis": hypothesis,
        "verdict": verdict,
        "g2_mismatch": (g2.get("metrics") or {}).get("mismatch_total"),
        "first_divergence": (g1.get("metrics") or {}).get("first_divergence"),
        "progress": key,
        "gates": _summary(results),
        "gpu_hours_total": round(gpu_seconds_total / 3600, 3),
        **(extra or {}),
    }
    with open(path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _read_ledger(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _report(run_dir, task, ledger, best, gpu_seconds, wall):
    lines = [
        f"# RSI run report: {task['id']}",
        "",
        f"- iterations: {len(ledger) - 1}",
        f"- GPU-hours: {gpu_seconds / 3600:.2f}, wall clock: {wall / 3600:.2f} h",
        f"- best candidate: `{best['id']}`, progress {best['key']}",
        "",
        "| iter | verdict | progress | g2 mismatch | first divergence | candidate | hypothesis |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in ledger:
        hyp = (e.get("hypothesis") or "").replace("\n", " ").replace("|", "/")[:100]
        lines.append(
            f"| {e['iteration']} | {e['verdict']} | {e.get('progress')} | {e['g2_mismatch']} "
            f"| {e['first_divergence']} | `{e['candidate']}` | {hyp} |"
        )
    with open(os.path.join(run_dir, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
