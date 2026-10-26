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

from rsi.orchestrator import brief
from rsi.orchestrator.loop import _rejection

TASK = yaml.safe_load(
    open(os.path.join(os.path.dirname(__file__), "..", "tasks", "t1_qwen3omni_thinker_lora_zero_diff.yaml"))
)


def test_rejection_names_first_failing_gate():
    flag = {"file": "f.py", "pattern": "sampling_params", "line": "+top_k = 5"}
    results = {
        "g0_static": {"status": "fail", "scope_violations": [], "integrity_flags": [flag]},
        "g2_loop_zero_diff": {"status": "fail", "checks": []},
    }
    assert _rejection(results) == {"gate": "g0_static", "status": "fail", "details": {"integrity_flags": [flag]}}

    results = {
        "g0_static": {"status": "pass"},
        "g3_grad": {
            "status": "fail",
            "checks": [{"name": "nonzero_grads", "ok": False}, {"name": "norm_ratio", "ok": True}],
        },
    }
    assert _rejection(results)["details"] == [{"name": "nonzero_grads", "ok": False}]
    assert _rejection({"g0_static": {"status": "pass"}}) is None


def test_brief_shows_rejection_and_previous_candidate():
    entry = {
        "iteration": 4,
        "verdict": "invalid",
        "hypothesis": "h",
        "rejection": {"gate": "g0_static", "status": "fail", "details": {"integrity_flags": [{"line": "+top_k = 5"}]}},
    }
    text = brief.render(TASK, 5, "p", {}, [entry])
    assert "rejected by g0_static" in text and "+top_k = 5" in text
    assert "rsi_previous_candidate/" in text
