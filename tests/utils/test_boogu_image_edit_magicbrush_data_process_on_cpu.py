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

"""Locks the MagicBrush edit-row contract the edit-consistency reward depends on."""

import importlib.util
import io
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image


def _load_module():
    path = Path(__file__).parents[2] / "examples/flowgrpo_trainer/data_process/boogu_image_edit_magicbrush.py"
    spec = importlib.util.spec_from_file_location("boogu_image_edit_magicbrush", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


converter = _load_module()


def _png(size=(64, 32)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (200, 10, 10)).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_shard(path: Path, rows: list[tuple[str, int, str]]) -> None:
    image = {"bytes": _png(), "path": None}
    table = pa.table(
        {
            "img_id": [img_id for img_id, _, _ in rows],
            "turn_index": [turn for _, turn, _ in rows],
            "instruction": [instruction for _, _, instruction in rows],
            "source_img": [image] * len(rows),
        }
    )
    pq.write_table(table, path)


@pytest.mark.parametrize(
    ("instruction", "removal"),
    [("Remove the pears", True), ("get rid of the dog", True), ("add a guide dog", False), ("make it rainy", False)],
)
def test_removal_instructions_are_detected(instruction, removal) -> None:
    assert converter.is_removal_instruction(instruction) is removal


def test_rows_carry_source_image_and_instruction_ground_truth(tmp_path) -> None:
    shard = tmp_path / "dev-00000.parquet"
    _write_shard(shard, [("1", 1, "add a hat"), ("1", 2, "make it red"), ("2", 1, "Remove the cup")])

    frame = converter.convert_split([shard], "test", -1, 32, first_turn_only=True, keep_removal=False)

    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["reward_model"]["ground_truth"] == "add a hat"
    assert row["prompt"][1]["content"] == "Picture 1: <image>add a hat"
    assert "<image>" not in row["negative_prompt"][1]["content"]
    source = row["extra_info"]["source_image"]["bytes"]
    assert source == row["images"][0]["bytes"]
    with Image.open(io.BytesIO(source)) as image:
        assert image.size == (32, 32)
