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

"""CPU tests for the instruction-following plus source-consistency edit reward."""

import io

import pytest
import torch
from PIL import Image

from verl_omni.utils.reward_score.edit_consistency_reward import compute_score_edit_consistency


def _png_bytes(color) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buffer, format="PNG")
    return buffer.getvalue()


class _FakeRewardModel:
    def __init__(self, consistency: float):
        self.consistency = consistency

    async def infer(self, prompts, images, sources):
        assert prompts == ["make the sky purple"]
        assert len(images) == 1 and isinstance(images[0], Image.Image)
        assert len(sources) == 1 and sources[0].getpixel((0, 0)) == (10, 20, 30)
        consistency = self.consistency
        return [
            {
                "pickscore_text_embedding": torch.tensor([1.0, 0.0]),
                "pickscore_image_embedding": torch.tensor([0.5, 0.75**0.5]),
                "pickscore_logit_scale": torch.tensor(26.0),
                "dino_output_embedding": torch.tensor([consistency, (1 - consistency**2) ** 0.5]),
                "dino_source_embedding": torch.tensor([1.0, 0.0]),
            }
        ]


_IMAGE = torch.zeros((3, 4, 4), dtype=torch.uint8)


def _extra_info(source=None) -> dict:
    return {"media_kind": "image", "source_image": {"bytes": _png_bytes((10, 20, 30))} if source is None else source}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("consistency", "penalty"),
    [(0.95, 0.0), (0.8, 0.0), (0.3, 0.5)],
)
async def test_edit_consistency_penalizes_only_below_threshold(consistency, penalty) -> None:
    result = await compute_score_edit_consistency(
        data_source="edit_consistency",
        solution_image=_IMAGE,
        ground_truth="make the sky purple",
        extra_info=_extra_info(),
        reward_model=_FakeRewardModel(consistency),
        consistency_weight=2.0,
        consistency_threshold=0.8,
    )

    assert result["pickscore_raw"] == pytest.approx(0.5)
    assert result["dino_consistency_raw"] == pytest.approx(consistency)
    assert result["consistency_penalty"] == pytest.approx(penalty)
    assert result["score"] == pytest.approx(0.5 - 2.0 * penalty)


@pytest.mark.asyncio
async def test_edit_consistency_accepts_raw_bytes_and_pil_sources() -> None:
    for source in (_png_bytes((10, 20, 30)), Image.new("RGB", (4, 4), (10, 20, 30))):
        result = await compute_score_edit_consistency(
            data_source="edit_consistency",
            solution_image=_IMAGE,
            ground_truth="make the sky purple",
            extra_info=_extra_info(source),
            reward_model=_FakeRewardModel(0.9),
        )
        assert result["consistency_penalty"] == 0.0


@pytest.mark.asyncio
async def test_edit_consistency_requires_source_image() -> None:
    with pytest.raises(ValueError, match="source_image"):
        await compute_score_edit_consistency(
            data_source="edit_consistency",
            solution_image=_IMAGE,
            ground_truth="make the sky purple",
            extra_info={"media_kind": "image"},
            reward_model=_FakeRewardModel(0.9),
        )
