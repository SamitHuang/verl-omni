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

import os

import yaml

from rsi.orchestrator import brief, objective

PRIMARY = {"metrics": ["eager", "graph"], "min_gain": 0.05, "target": 0.9}
GATES = ("g0_static", "g2_loop_zero_diff")
PASS = {g: {"status": "pass"} for g in GATES}
STOCK = {"metrics": {"eager": 100.0, "graph": 200.0}}


def _bench(eager, graph):
    return {"metrics": {"eager": eager, "graph": graph}}


def test_score_is_the_worst_ratio_and_skips_metrics_stock_lacks():
    assert objective.score({"eager": 80.0, "graph": 190.0}, STOCK["metrics"], ["eager", "graph"]) == 0.8
    assert objective.score({"eager": 80.0}, {"eager": 100.0, "graph": None}, ["eager", "graph"]) == 0.8
    assert objective.score({"eager": 80.0}, STOCK["metrics"], ["eager", "graph"]) == 0.0


def test_no_paired_stock_gives_no_score():
    assert objective.key(PASS, _bench(150.0, 300.0), None, PRIMARY, GATES) == (True, 0.0)


def test_failed_or_missing_constraint_makes_a_candidate_invalid():
    fast = _bench(150.0, 300.0)
    assert objective.key({**PASS, "g2_loop_zero_diff": {"status": "fail"}}, fast, STOCK, PRIMARY, GATES) == (False, 0.0)
    assert objective.key({"g0_static": {"status": "pass"}}, fast, STOCK, PRIMARY, GATES) == (False, 0.0)
    assert objective.verdict((False, 0.0), (True, 0.5), 0.05) == "invalid"


def test_gains_inside_the_noise_are_neutral():
    best = objective.key(PASS, _bench(60.0, 140.0), STOCK, PRIMARY, GATES)
    assert best == (True, 0.6)
    assert objective.verdict(objective.key(PASS, _bench(62.0, 140.0), STOCK, PRIMARY, GATES), best, 0.05) == "neutral"
    assert objective.verdict(objective.key(PASS, _bench(66.0, 140.0), STOCK, PRIMARY, GATES), best, 0.05) == "improved"
    assert objective.verdict(objective.key(PASS, _bench(50.0, 140.0), STOCK, PRIMARY, GATES), best, 0.05) == "regressed"


def test_threshold_widens_with_the_measured_spread():
    assert objective.threshold(PRIMARY, {"eager": 0.01}) == 0.05
    assert objective.threshold(PRIMARY, {"eager": 0.01}, {"graph": 0.04}) == 0.08


def test_threshold_ignores_unscored_metrics():
    assert objective.threshold(PRIMARY, {"eager": 0.01, "graph_bs32": 0.385}) == 0.05


def test_solved_needs_constraints_and_target():
    assert objective.solved((True, 0.91), PRIMARY)
    assert not objective.solved((True, 0.85), PRIMARY)
    assert not objective.solved((False, 0.0), PRIMARY)
    assert not objective.solved((True, 2.0), {"metrics": []})


def test_resume_confirms_earlier_candidates_at_the_target_best_first():
    from rsi.orchestrator.loop import _unconfirmed_at_target

    entries = [
        {"iteration": 3, "verdict": "improved", "progress": [True, 0.898]},
        {"iteration": 4, "verdict": "neutral", "progress": [True, 0.93]},
        {"iteration": 5, "verdict": "neutral", "progress": [True, 0.944]},
        {"iteration": 6, "verdict": "unconfirmed", "progress": [True, 0.99]},
        {"iteration": 7, "verdict": "neutral", "progress": [False, 0.0]},
    ]
    assert [e["iteration"] for e in _unconfirmed_at_target(entries, PRIMARY)] == [5, 4]


def test_stock_reference_is_reused_only_when_usable(tmp_path):
    import json

    import pytest

    from rsi.orchestrator.loop import _reuse_stock_reference

    class Ws:
        root = str(tmp_path / "stock_ws")

        def reset(self, patches):
            self.patches = patches

    def write(gate, status):
        os.makedirs(tmp_path / "stock" / gate, exist_ok=True)
        (tmp_path / "stock" / gate / "result.json").write_text(json.dumps({"status": status}))

    write("g2_loop_zero_diff", "fail")
    with pytest.raises(SystemExit):
        _reuse_stock_reference(str(tmp_path / "stock"), Ws())
    write("bench_loop", "error")
    with pytest.raises(SystemExit):
        _reuse_stock_reference(str(tmp_path / "stock"), Ws())
    write("bench_loop", "pass")
    ws = Ws()
    stock = _reuse_stock_reference(str(tmp_path / "stock"), ws)
    assert ws.patches == {}
    assert stock["from_dir"] == str(tmp_path / "stock" / "g2_loop_zero_diff")
    assert stock["bench_loop"]["status"] == "pass"


def test_a_candidate_at_the_target_is_confirmed_even_within_the_noise():
    assert objective.needs_confirmation("improved", (True, 0.7), PRIMARY)
    assert objective.needs_confirmation("neutral", (True, 0.955), PRIMARY)
    assert not objective.needs_confirmation("neutral", (True, 0.85), PRIMARY)
    assert not objective.needs_confirmation("invalid", (False, 0.0), PRIMARY)


def test_maximize_brief_shows_ratios_to_stock_and_loop_phases():
    task = yaml.safe_load(
        open(os.path.join(os.path.dirname(__file__), "..", "tasks", "t1_minicpmo_thinker_zero_diff_throughput.yaml"))
    )
    metric = task["objective"]["primary"]["metrics"][0]
    stock = {"bench_loop": {"metrics": {"step": 30.0, "gen": 15.0}}}
    parent = {
        "g2_loop_zero_diff": {"status": "pass"},
        "bench": {"metrics": {metric: 150.0}},
        "bench_stock": {"metrics": {metric: 200.0}},
    }
    entry = {"iteration": 1, "verdict": "neutral", "progress": [True, 0.75], "hypothesis": "h"}
    text = brief.render(task, 2, "p", parent, [entry], stock=stock, gain_needed=0.08)
    assert f"| `{metric}` (objective) | 200 | 150 | 75.0% |" in text
    assert "| step | 30.0 | not measured |" in text
    assert "only above 8%" in text and "progress=[True, 0.75]" in text
    assert "raise the objective throughput" in text and "first divergence later" not in text
