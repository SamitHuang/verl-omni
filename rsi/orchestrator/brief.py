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
"""Render the per-iteration brief the agent reads."""

import json

from rsi.evaluate.g0 import INTEGRITY_PATTERNS
from rsi.orchestrator.agents import HYPOTHESIS_NAME


def _divergence_table(g1):
    if not g1:
        return "No g1 result yet."
    fd = next((c for c in g1.get("checks", []) if c.get("name") == "first_divergence"), None)
    lines = [
        f"First diverging op: `{g1.get('metrics', {}).get('first_divergence')}`",
        "",
        "| op | status | tokens differ | max abs diff |",
        "|---|---|---|---|",
    ]
    for row in (fd or {}).get("ops", [])[:20]:
        lines.append(
            f"| {row['op']} | {row['status']} | {row.get('tokens_differ', '')} | {row.get('abs_diff_max', '')} |"
        )
    lines += ["", "| check | mismatched / compared |", "|---|---|"]
    for c in g1.get("checks", []):
        if "mismatch_total" in c:
            lines.append(f"| {c['name']} | {c['mismatch_total']} / {c['tokens_compared']} |")
        elif c.get("status") == "unavailable":
            lines.append(f"| {c['name']} | unavailable |")
    return "\n".join(lines)


def _environment(task, n_gpus, has_tools):
    repos = list(task["start"]["repos"])
    lines = [
        f"- Python: `{task['start']['python']}` (shared venv; do not install or upgrade packages).",
        f"- To run code against your edits, put the workspace repos first: `PYTHONPATH={':'.join(repos)}` "
        "from the workspace root.",
        f"- GPUs: {n_gpus} leased for this turn via `CUDA_VISIBLE_DEVICES`; use only those. "
        "Every process you start is killed when your turn ends.",
        "- Opt-in switches: write `CANDIDATE_ENV` at the workspace root, one `KEY=VALUE` per line. The evaluator sets"
        " them for every gate (rollout and actor processes). Allowed keys: "
        + ", ".join(f"`{p}`" for p in (task.get("candidate_env_rules") or {}).get("allow", []))
        + ".",
    ]
    if has_tools:
        lines.append(
            "- Diagnostics: `rsi_tools/` at the workspace root (see its README). It is not part of the candidate."
        )
    notes = (task.get("environment") or "").strip()
    return "\n".join(lines) + (f"\n\n{notes}" if notes else "")


LOOP_PHASES = ("step", "gen", "old_log_prob", "update_actor", "update_weights")


def _objective(primary):
    if primary.get("type") != "maximize":
        return f"Objective: `{primary['metric']} {primary['goal']}` on the\nevaluator's gates."
    target = f" (solved at >= {primary['target']:.0%} of stock)" if primary.get("target") is not None else ""
    return (
        f"Objective: maximize the worst of {', '.join(f'`{m}`' for m in primary['metrics'])} relative to the\n"
        f"unpatched code{target}, while every correctness gate below keeps passing (any failure invalidates\n"
        "the candidate). A gain is confirmed in a full-depth training loop that must stay bitwise (diff 0)."
    )


def _num(v, fmt=".0f", missing="n/a"):
    return missing if v is None else format(v, fmt)


def _throughput(primary, stock, last, gain_needed):
    if primary.get("type") != "maximize" or not stock:
        return ""
    s, p = ((last.get(b) or {}).get("metrics") or {} for b in ("bench_stock", "bench"))
    lines = [
        "## Throughput (tokens/s; parent and stock measured back to back on the same GPUs)",
        "",
        "| metric | stock | parent | parent / stock |",
        "|---|---|---|---|",
    ]
    for k in sorted(set(s) | set(p)):
        sv, pv = s.get(k), p.get(k)
        ratio = f"{pv / sv:.1%}" if sv and pv is not None else "n/a"
        mark = " (objective)" if k in primary["metrics"] else ""
        lines.append(f"| `{k}`{mark} | {_num(sv)} | {_num(pv)} | {ratio} |")
    sl, pl = (stock.get("bench_loop") or {}).get("metrics") or {}, (last.get("bench_loop") or {}).get("metrics") or {}
    lines += [
        "",
        "Full-depth training loop, median seconds per step:",
        "",
        "| phase | stock | parent |",
        "|---|---|---|",
    ]
    for k in LOOP_PHASES:
        lines.append(f"| {k} | {_num(sl.get(k), '.1f')} | {_num(pl.get(k), '.1f', 'not measured')} |")
    lines += ["", f"A change counts as an improvement only above {gain_needed:.0%} (run-to-run noise)."]
    return "\n".join(lines) + "\n"


def render(
    task,
    iteration,
    parent,
    last_results,
    recent_ledger,
    kb_text="",
    n_gpus=1,
    has_tools=False,
    stock=None,
    gain_needed=0.05,
):
    last = last_results or {}
    primary = task["objective"]["primary"]
    maximize = primary.get("type") == "maximize"
    g0, g1, g2, g3 = (last.get(g) for g in ("g0_static", "g1_offline_zero_diff", "g2_loop_zero_diff", "g3_grad"))
    ledger_lines = []
    for e in recent_ledger:
        result = (
            f"progress={e.get('progress')}"
            if maximize
            else f"g2_mismatch={e.get('g2_mismatch')} first_div={e.get('first_divergence')}"
        )
        ledger_lines.append(
            f"- iter {e['iteration']}: verdict={e['verdict']} {result} — {e.get('hypothesis', '')[:160]}"
        )
        if e.get("rejection"):
            ledger_lines.append(
                f"  rejected by {e['rejection']['gate']}: {json.dumps(e['rejection']['details'])[:1500]}"
            )
    if recent_ledger and recent_ledger[-1].get("rejection"):
        ledger_lines.append(
            "  Its patches are in `rsi_previous_candidate/` (reference only; the workspace starts from the parent)."
        )
    ledger_lines = ledger_lines or ["- (none)"]
    forbidden = ", ".join(f"`{name}`" for name, _ in INTEGRITY_PATTERNS)
    if maximize:
        step1 = (
            "1. Make one focused change that you expect to raise the objective throughput without breaking any gate.\n"
            "   Measure it on your GPUs before you stop, and re-check that rollout and actor still agree bitwise."
        )
    else:
        step1 = (
            "1. Make one focused change that you expect to move the first divergence later or reduce mismatches.\n"
            "   Verify it on your GPUs before you stop (for example a small script that compares the two sides' op)."
        )
    g0_flags = json.dumps((g0 or {}).get("scope_violations", []) + (g0 or {}).get("integrity_flags", []))[:800]
    g3_failed = json.dumps([c for c in (g3 or {}).get("checks", []) if not c.get("ok", True)])[:800]
    return f"""# RSI task brief — iteration {iteration}

## Goal
{task["description"].strip()}

{_objective(primary)} The evaluator is outside this workspace; you cannot run or edit it.

## Scope (fixed; changing it invalidates the candidate)
```json
{json.dumps(task["scope"], indent=2)}
```

## Workspace
Repos at the workspace root: {", ".join(f"`{r}/`" for r in task["start"]["repos"])}. They already
contain the parent candidate (`{parent}`). Edit them in place; do not commit.

Editable: {", ".join(f"`{g}`" for g in task["editable"])}
Never edit: {", ".join(f"`{g}`" for g in task.get("protected", []))}
Added or removed lines matching these patterns fail the candidate: {forbidden}.
Do not use the network except for your own model API: no web search, no fetching URLs, no git history
beyond the start commits. Keep changes minimal and opt-in where possible.

## Environment
{_environment(task, n_gpus, has_tools)}

## Last evaluation
g0: {(g0 or {}).get("status")}  {g0_flags}
g3 (actor gradients vs an fp32 reference; stock bf16 is the bar; failing it invalidates the candidate):
{(g3 or {}).get("status")}  {json.dumps((g3 or {}).get("metrics", {}))}  {g3_failed}
g2 (real loop, 2 steps): {(g2 or {}).get("status")}  {json.dumps((g2 or {}).get("metrics", {}))}

g1 (offline replay of the g2 step-1 batch, per-op comparison for one row; every check below must be 0):
{(g1 or {}).get("status")}  mismatch_total={(g1 or {}).get("metrics", {}).get("mismatch_total")}
{_divergence_table(g1)}

{_throughput(primary, stock, last, gain_needed)}
## Recent iterations
{chr(10).join(ledger_lines)}

## Knowledge base
{kb_text or "(empty)"}

## Deliverable
{step1}
2. Write `{HYPOTHESIS_NAME}` at the workspace root: what you changed, why, what you verified, and what
   result you expect from the gates.
"""
