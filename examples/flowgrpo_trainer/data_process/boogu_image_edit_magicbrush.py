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
"""Convert MagicBrush to parquet for Boogu-Image-Edit RL with the edit-consistency reward.

MagicBrush (``osunlp/MagicBrush``) pairs real COCO photos with free-form edit
instructions. Download the parquet shards first, e.g.::

    hf download osunlp/MagicBrush --repo-type dataset --local-dir ~/data/magicbrush_raw \\
        --include "data/dev-*" $(printf 'data/train-%05d-* ' $(seq 0 4 48))

That is 13 of the 51 train shards (2,071 rows after filtering), enough for the
1,440 training rows the recipe consumes.

Training rows come from the ``train`` shards (every turn); validation rows come
from the ``dev`` shards, first turn only, so validation sources are untouched
photos and share no image with training.

``reward_model.ground_truth`` holds the instruction (PickScore's prompt) and
``extra_info["source_image"]`` the resized source, which
``edit_consistency_reward.py`` compares the edited output against.

Removal instructions ("remove the pears") are dropped by default: CLIP-family
scores do not model negation, so the instruction text scores *higher* when the
object is still present and the reward would push the policy the wrong way.
"""

import argparse
import io
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image, ImageOps

# Same template as boogu_image_edit_ocr.py (upstream SYSTEM_PROMPT_4_TI2I_UNIFIED).
BOOGU_SYSTEM_PROMPT_TI2I = (
    "Describe the key features of the input image (color, shape, size, texture, objects, background), "
    "then explain how the user's text instruction should alter or modify the image. Generate a new image "
    "that meets the user's requirements while maintaining consistency with the original input where appropriate."
)

_REMOVAL_PATTERN = re.compile(r"\b(remove|delete|erase|get rid|take away|take out|eliminate|without)\b", re.IGNORECASE)


def is_removal_instruction(instruction: str) -> bool:
    return bool(_REMOVAL_PATTERN.search(instruction))


def load_source_image(image_bytes: bytes, image_size: int) -> bytes:
    """Center-crop to a square so every sample shares one output resolution."""
    with Image.open(io.BytesIO(image_bytes)) as source:
        image = ImageOps.fit(source.convert("RGB"), (image_size, image_size), method=Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def build_row(split: str, index: int, img_id: str, turn: int, instruction: str, image: bytes) -> dict:
    return {
        "data_source": "edit_consistency",
        "prompt": [
            {"role": "system", "content": BOOGU_SYSTEM_PROMPT_TI2I},
            {"role": "user", "content": f"Picture 1: <image>{instruction}"},
        ],
        # Text-only on purpose; see the negative-prompt note in boogu_image_edit_ocr.py.
        "negative_prompt": [
            {"role": "system", "content": BOOGU_SYSTEM_PROMPT_TI2I},
            {"role": "user", "content": ""},
        ],
        "ability": "image_edit",
        "images": [{"bytes": image}],
        "reward_model": {"style": "model", "ground_truth": instruction},
        "extra_info": {
            "split": split,
            "index": index,
            "instruction": instruction,
            "img_id": img_id,
            "turn_index": turn,
            "source_image": {"bytes": image},
        },
    }


def convert_split(
    shards: list[Path], split: str, max_samples: int, image_size: int, first_turn_only: bool, keep_removal: bool
) -> pd.DataFrame:
    rows = []
    for shard in shards:
        table = pq.read_table(shard, columns=["img_id", "turn_index", "instruction", "source_img"]).to_pylist()
        for example in table:
            if max_samples >= 0 and len(rows) >= max_samples:
                return pd.DataFrame(rows)
            instruction = str(example["instruction"]).strip()
            turn = int(example["turn_index"])
            if not instruction or (first_turn_only and turn != 1):
                continue
            if not keep_removal and is_removal_instruction(instruction):
                continue
            image = load_source_image(example["source_img"]["bytes"], image_size)
            rows.append(build_row(split, len(rows), str(example["img_id"]), turn, instruction, image))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir", type=Path, required=True, help="MagicBrush snapshot holding data/*.parquet")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_size", type=int, default=-1)
    parser.add_argument("--val_size", type=int, default=256)
    parser.add_argument("--image_size", type=int, default=512, help="Square edge; a multiple of 16.")
    parser.add_argument("--keep_removal", action="store_true", help="Keep removal instructions (see module doc).")
    args = parser.parse_args()

    if args.image_size % 16 or not 0 < args.image_size <= 2048:
        raise ValueError(f"--image_size must be a multiple of 16 in (0, 2048]; got {args.image_size}")
    data_dir = args.input_dir.expanduser() / "data"
    train_shards = sorted(data_dir.glob("train-*.parquet"))
    dev_shards = sorted(data_dir.glob("dev-*.parquet"))
    if not train_shards or not dev_shards:
        raise FileNotFoundError(f"expected train-*.parquet and dev-*.parquet under {data_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = convert_split(train_shards, "train", args.train_size, args.image_size, False, args.keep_removal)
    validation = convert_split(dev_shards, "test", args.val_size, args.image_size, True, args.keep_removal)
    train.to_parquet(args.output_dir / "train.parquet", row_group_size=200)
    validation.to_parquet(args.output_dir / "test.parquet", row_group_size=200)
    print(f"Wrote {len(train)} training and {len(validation)} validation samples to {args.output_dir}")


if __name__ == "__main__":
    main()
