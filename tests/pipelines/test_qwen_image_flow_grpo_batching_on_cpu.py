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

import torch

from verl_omni.pipelines.qwen_image_flow_grpo.common import (
    gather_padded_prompt_batch,
    pad_prompt_embeds_to_len,
    rope_txt_seq_lens,
)
from verl_omni.pipelines.qwen_image_flow_grpo.vllm_omni_rollout_adapter import QwenImagePipelineWithLogProb


def test_pad_prompt_embeds_to_len():
    short = torch.ones(1, 3, 4)
    long = torch.ones(1, 7, 4)
    padded_short = pad_prompt_embeds_to_len(short, 7)
    padded_long = pad_prompt_embeds_to_len(long, 7)
    assert padded_short.shape == (1, 7, 4)
    assert padded_long.shape == (1, 7, 4)
    assert torch.equal(padded_short[:, :3], short)
    assert torch.equal(padded_short[:, 3:], torch.zeros(1, 4, 4))


def test_gather_padded_prompt_batch():
    embeds_a = torch.ones(1, 5, 8)
    embeds_b = torch.full((1, 9, 8), 2.0)
    mask_a = torch.ones(1, 5, dtype=torch.long)
    mask_b = torch.ones(1, 9, dtype=torch.long)
    embeds, masks, target = gather_padded_prompt_batch(
        [embeds_a, embeds_b],
        [mask_a, mask_b],
        max_sequence_length=16,
    )
    assert target == 9
    assert embeds.shape == (2, 9, 8)
    assert masks.shape == (2, 9)
    assert rope_txt_seq_lens(embeds) == [9, 9]


def test_supports_request_batch_enabled():
    assert QwenImagePipelineWithLogProb.supports_request_batch is True
