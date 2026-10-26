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
"""The ``maximize`` objective: throughput under hard constraints.

A candidate first has to pass every correctness gate (the constraints); among
those, it is ranked by the worst ratio of its bench metrics to stock measured
back to back on the same GPUs, so a candidate cannot buy decode speed with actor
speed. Ratios still move between runs, so a gain counts only above a noise threshold.

Task config (``objective.primary``)::

    type: maximize
    metrics: [rollout_eager_tok_s_bs128, rollout_cudagraph_tok_s_bs128]
    min_gain: 0.05      # smallest gain that counts as an improvement
    target: 0.9         # optional: worst ratio at which the task is solved
"""

CONSTRAINT_GATES = ("g0_static", "g3_grad", "g2_loop_zero_diff", "g1_offline_zero_diff")


def constraints_ok(results, gates=CONSTRAINT_GATES):
    """Every listed gate ran and passed; pass the task's own gate list (g3 is optional per task)."""
    return all((results.get(g) or {}).get("status") == "pass" for g in gates)


def ratios(metrics, stock, keys):
    """Candidate / stock per metric; a metric stock lacks (e.g. CUDA graphs unavailable on this model) is skipped."""
    return {k: (metrics.get(k) or 0.0) / stock[k] for k in keys if stock.get(k)}


def score(metrics, stock, keys):
    r = ratios(metrics, stock, keys)
    return min(r.values()) if r else None


def threshold(primary, *spreads):
    """Relative gain needed to count: ``min_gain``, or twice the worst repeat spread of either side.

    Only the scored metrics count: an unscored, noisy one (stock CUDA graph at bs32 varied by 38%) would
    otherwise make every gain look like noise.
    """
    metrics = set(primary["metrics"])
    worst = max((s for spread in spreads for m, s in (spread or {}).items() if m in metrics), default=0.0)
    return max(float(primary.get("min_gain", 0.05)), 2 * worst)


def key(results, bench, stock, primary, gates=CONSTRAINT_GATES):
    """Ordering key, larger is better: constraints pass, then the worst ratio to stock (rounded to 0.1%).

    ``stock`` is the bench measured right next to ``bench`` (paired), not a session-wide one.
    """
    ok = constraints_ok(results, gates)
    s = (
        score(bench.get("metrics") or {}, stock.get("metrics") or {}, primary["metrics"])
        if ok and bench and stock
        else None
    )
    return (ok, round(s, 3) if s is not None else 0.0)


def verdict(new_key, best_key, gain_needed):
    """``improved`` only when the constraints hold and the score beats the best by more than the noise."""
    if not new_key[0]:
        return "invalid"
    if best_key is None or not best_key[0]:
        return "improved"
    if new_key[1] > best_key[1] * (1 + gain_needed):
        return "improved"
    if new_key[1] < best_key[1] * (1 - gain_needed):
        return "regressed"
    return "neutral"


def solved(new_key, primary):
    return new_key[0] and primary.get("target") is not None and new_key[1] >= float(primary["target"])


def needs_confirmation(verdict, new_key, primary):
    """Confirm in the full-depth loop an improvement, or any candidate at the target.

    The target is absolute; a candidate that reaches it within the noise of the best is still a solution.
    """
    return verdict == "improved" or (verdict in ("neutral", "regressed") and solved(new_key, primary))
