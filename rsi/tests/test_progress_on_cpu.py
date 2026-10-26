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

from rsi.orchestrator.loop import progress


def _res(g2_status, mismatch, first, g1_status="fail", tokens=16384, step=1):
    return {
        "g2_loop_zero_diff": {"status": g2_status, "metrics": {"mismatch_total": mismatch, "tokens_compared": tokens}},
        "g1_offline_zero_diff": {"status": g1_status, "metrics": {"first_divergence": first, "replay_step": step}},
    }


def test_sampling_noise_is_neutral():
    assert progress(_res("fail", 16383, "L0.input_norm"), 4) == progress(_res("fail", 16384, "L0.input_norm"), 4)


def test_later_divergence_is_better_even_if_all_tokens_differ():
    assert progress(_res("fail", 16384, "L0.q_norm"), 4) > progress(_res("fail", 16384, "L0.input_norm"), 4)


def test_pass_beats_everything_and_invalid_has_no_key():
    assert progress(_res("pass", 0, None, "pass"), 4) > progress(_res("fail", 10, "logits"), 4)
    assert progress(_res("invalid", 0, None), 4) is None


def test_equal_first_step_beats_deeper_divergence_at_step_one():
    # T2 iteration 4: step 1 bitwise equal, step 2 diverges at L0.q_norm.
    assert progress(_res("fail", 8192, "L0.q_norm", step=2), 4) > progress(_res("fail", 16383, None), 4)
