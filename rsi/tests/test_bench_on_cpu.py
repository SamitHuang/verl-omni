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

import json
import os

import pytest
import torch
import yaml

from rsi.evaluate.bench import loop_overrides, profile_config, run_bench, summarize_loop
from rsi.evaluate.bench_actor import _chunks, _tile
from rsi.evaluate.g2 import _parse_step_metrics
from rsi.tools.bench_candidates import table


def _fields(lengths):
    return {
        "input_ids": [torch.arange(n) for n in lengths],
        "position_ids": [torch.arange(n) for n in lengths],
        "response_mask": [torch.ones(n // 2) for n in lengths],
    }


def test_tile_repeats_rows_in_order():
    tiled = _tile(_fields([3, 5]), 3)
    assert [int(t.shape[0]) for t in tiled["input_ids"]] == [3, 5, 3, 5, 3, 5]
    assert set(tiled) == {"input_ids", "position_ids", "response_mask"}


def test_chunks_respect_limit_and_keep_oversized_rows():
    fields = _fields([4, 4, 4, 10, 2])
    assert _chunks(fields, 8) == [[0, 1], [2], [3], [4]]
    assert sum(len(c) for c in _chunks(fields, 100)) == 5


def test_missing_capture_is_an_error(tmp_path):
    task = {"scope": {"checkpoint": "/x"}, "start": {"python": "python", "repos": {}}}
    r = run_bench(task, str(tmp_path), str(tmp_path / "out"), "0", "/h", from_dir=str(tmp_path / "nope"))
    assert r["status"] == "error"


def test_table_is_relative_to_stock_of_the_same_task(tmp_path):
    spec = [
        {"label": "stock", "stock": True, "task": "t"},
        {"label": "cand", "task": "t"},
        {"label": "other_stock", "stock": True, "task": "u"},
    ]
    metrics = {
        "stock": {"actor_fwd_bwd_tok_s": 100.0, "diff": 0.47},
        "cand": {"actor_fwd_bwd_tok_s": 80.0, "diff": None},
        "other_stock": {"actor_fwd_bwd_tok_s": 10.0},
    }
    for label, m in metrics.items():
        os.makedirs(tmp_path / label)
        (tmp_path / label / "result.json").write_text(json.dumps({"status": "pass", "metrics": m}))
    text = table(spec, str(tmp_path))
    assert "| cand | pass | 80 (80%) | n/a |" in text
    assert "| stock | pass | 100 | 0.47 |" in text


_PROFILES = """
scope: {checkpoint: /c}
bench:
  actor_checkpoint: /full
  loop: {gpus: 4, overrides: [a=1]}
  converge: &converge
    steps: 150
    overrides: &base [x=1, y=2]
  converge_bypass:
    <<: *converge
    overrides: [*base, z=3]
"""


def test_profiles_layer_on_loop_and_flatten_aliased_overrides():
    task = yaml.safe_load(_PROFILES)
    assert profile_config(task, "loop")["steps"] == 5
    conv = profile_config(task, "converge_bypass")
    assert (conv["gpus"], conv["steps"], conv["warmup_steps"]) == (4, 150, 1)
    args = loop_overrides(task, 4, "converge_bypass")
    assert args[-3:] == ["x=1", "y=2", "z=3"]
    assert "trainer.total_training_steps=150" in args and "trainer.experiment_name=bench_converge_bypass" in args
    with pytest.raises(KeyError):
        profile_config(task, "nope")


def test_summarize_loop_skips_warmup_and_tracks_scores_and_validation():
    steps = {
        0: {"val-core/openai/gsm8k/acc/mean@1": 0.5},
        1: {"timing_s/step": 100.0, "critic/score/mean": 0.1, "training/rollout_probs_diff_max": 0.3},
        2: {"timing_s/step": 10.0, "critic/score/mean": 0.2},
        3: {"timing_s/step": 12.0, "critic/score/mean": 0.4, "val-core/openai/gsm8k/acc/mean@1": 0.7},
    }
    out = summarize_loop(steps, warmup_steps=1, window=2)
    assert out["steps_measured"] == [2, 3]
    assert out["timing_s/step"] == 11.0
    assert out["rollout_probs_diff_max"] == 0.3
    assert out["score_first"] == pytest.approx(0.15) and out["score_last"] == pytest.approx(0.3)
    assert (out["val_first"], out["val_last"]) == (0.5, 0.7)


def test_step_metrics_parse_numpy_scalar_reprs(tmp_path):
    log = tmp_path / "log"
    log.write_text(
        "(TaskRunner) step:0 - val-core/openai/gsm8k/acc/mean@1:np.float64(0.30859375)"
        " - val-aux/num_turns/min:np.int64(2)\n"
    )
    assert _parse_step_metrics(str(log)) == {
        0: {"val-core/openai/gsm8k/acc/mean@1": 0.30859375, "val-aux/num_turns/min": 2.0}
    }
