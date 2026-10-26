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
"""Write a layer-truncated checkpoint of an omni model's LLM.

The output keeps the original format so the HF actor and the vLLM-Omni rollout
load it unchanged. Families:

- ``qwen3_omni``: ``Qwen3OmniMoeForConditionalGeneration``. Talker and code2wav
  weights are dropped and ``enable_audio_output`` is set to false; the audio and
  vision towers are kept because both loaders build them.
- ``minicpm_o``: MiniCPM-o (remote code). Only LLM layers are dropped; the
  vision/audio/TTS modules are kept because the remote ``__init__`` builds them.

Usage:
    python -m rsi.tools.make_truncated_ckpt --src <snapshot> --dst <dir> --num-layers 4 [--family minicpm_o]
"""

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import defaultdict

from safetensors import safe_open
from safetensors.torch import save_file

_SHARD_BYTES = 4 * 1024**3

# family: (regex for an LLM layer key, key prefixes kept (None = all))
_FAMILIES = {
    "qwen3_omni": (re.compile(r"^thinker\.model\.layers\.(\d+)\."), ("thinker.",)),
    "minicpm_o": (re.compile(r"^llm\.model\.layers\.(\d+)\."), None),
}


def keep_key(key: str, num_layers: int, family: str = "qwen3_omni") -> bool:
    layer_re, prefixes = _FAMILIES[family]
    if prefixes is not None and not key.startswith(prefixes):
        return False
    m = layer_re.match(key)
    return m is None or int(m.group(1)) < num_layers


def _truncate_text_config(text: dict, num_layers: int) -> None:
    if num_layers > text["num_hidden_layers"]:
        raise ValueError(f"num_layers={num_layers} exceeds {text['num_hidden_layers']}")
    text["num_hidden_layers"] = num_layers
    if isinstance(text.get("mlp_only_layers"), list):
        text["mlp_only_layers"] = [i for i in text["mlp_only_layers"] if i < num_layers]
    if isinstance(text.get("layer_types"), list):
        text["layer_types"] = text["layer_types"][:num_layers]


def truncate_config(config: dict, num_layers: int, family: str = "qwen3_omni") -> dict:
    config = json.loads(json.dumps(config))
    if family == "qwen3_omni":
        _truncate_text_config(config["thinker_config"]["text_config"], num_layers)
        config["enable_audio_output"] = False
    else:
        _truncate_text_config(config, num_layers)
        if isinstance(config.get("max_window_layers"), int):
            config["max_window_layers"] = min(config["max_window_layers"], num_layers)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--family", choices=sorted(_FAMILIES), default="qwen3_omni")
    args = parser.parse_args()

    if os.path.exists(args.dst) and os.listdir(args.dst):
        raise FileExistsError(f"{args.dst} is not empty")
    os.makedirs(args.dst, exist_ok=True)

    with open(os.path.join(args.src, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    by_shard = defaultdict(list)
    for key, shard in weight_map.items():
        if keep_key(key, args.num_layers, args.family):
            by_shard[shard].append(key)

    out_map, buf, buf_bytes, shard_idx, total = {}, {}, 0, 0, 0
    shard_names = []

    def flush():
        nonlocal buf, buf_bytes, shard_idx
        if not buf:
            return
        shard_idx += 1
        name = f"model-{shard_idx:05d}.safetensors"
        save_file(buf, os.path.join(args.dst, name), metadata={"format": "pt"})
        os.chmod(os.path.join(args.dst, name), 0o644)  # save_file writes 0600; the agent user must read it
        shard_names.append(name)
        for k in buf:
            out_map[k] = name
        buf, buf_bytes = {}, 0

    for shard in sorted(by_shard):
        with safe_open(os.path.join(args.src, shard), framework="pt") as f:
            for key in sorted(by_shard[shard]):
                t = f.get_tensor(key)
                nbytes = t.numel() * t.element_size()
                if buf_bytes + nbytes > _SHARD_BYTES:
                    flush()
                buf[key] = t
                buf_bytes += nbytes
                total += nbytes
    flush()

    n = len(shard_names)
    final_map = {}
    for i, name in enumerate(shard_names, 1):
        new = f"model-{i:05d}-of-{n:05d}.safetensors"
        os.rename(os.path.join(args.dst, name), os.path.join(args.dst, new))
        final_map[name] = new
    out_map = {k: final_map[v] for k, v in out_map.items()}
    with open(os.path.join(args.dst, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(out_map.items()))}, f, indent=2)

    with open(os.path.join(args.src, "config.json")) as f:
        config = truncate_config(json.load(f), args.num_layers, args.family)
    with open(os.path.join(args.dst, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    for name in os.listdir(args.src):
        if name.endswith(".safetensors") or name in ("model.safetensors.index.json", "config.json"):
            continue
        src = os.path.join(args.src, name)
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(args.dst, name))

    digest = hashlib.sha256(json.dumps(out_map, sort_keys=True).encode()).hexdigest()
    with open(os.path.join(args.dst, "rsi_truncation.json"), "w") as f:
        json.dump(
            {
                "src": os.path.realpath(args.src),
                "family": args.family,
                "num_layers": args.num_layers,
                "tensors": len(out_map),
                "total_bytes": total,
                "weight_map_sha256": digest,
            },
            f,
            indent=2,
        )
    print(f"wrote {len(out_map)} tensors, {total / 1024**3:.2f} GiB, {n} shards to {args.dst}")


if __name__ == "__main__":
    main()
