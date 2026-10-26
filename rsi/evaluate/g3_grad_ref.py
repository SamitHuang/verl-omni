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
"""Reference side of the gradient gate; independent of any candidate.

Runs with only the harness on ``PYTHONPATH`` (stock transformers from the venv):

- ``ref``: fp32 language model, SDPA, one row per forward (no packing);
- ``stock``: bf16 language model, flash-attention-2, rows packed as the actor packs them.

The language model is the Qwen3-Omni thinker or MiniCPM-o's ``llm`` (a stock
``Qwen3ForCausalLM``), by ``--family``.

``stock`` calibrates how far ordinary bf16 training is from the fp32 gradient.

    python -m rsi.evaluate.g3_grad_ref --ckpt C --fixture F --out O.pt [--family minicpm_o]
"""

import argparse

import torch

from rsi.evaluate.grad_common import ROWS, batched_positions, boundaries, canonical, loss_from_logprobs, packed_inputs


def _logprobs(logits, labels):
    return torch.log_softmax(logits.float(), -1).gather(-1, labels[:, None]).squeeze(-1)


def _grads(thinker):
    return {canonical(n): p.grad.detach().float().cpu() for n, p in thinker.named_parameters() if p.grad is not None}


def _load_qwen3_omni(ckpt, dtype, attn):
    from transformers import Qwen3OmniMoeForConditionalGeneration

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(ckpt, dtype=dtype, attn_implementation=attn)
    thinker = model.thinker.cuda().train()
    for module in (getattr(model, "talker", None), getattr(model, "code2wav", None)):
        if module is not None:
            module.to("meta")
    return thinker


def _load_minicpm_o(ckpt, dtype, attn):
    """MiniCPM-o's ``llm`` as a stock ``Qwen3ForCausalLM``, without the remote code or the towers."""
    import json
    import os

    from safetensors import safe_open
    from transformers import Qwen3Config, Qwen3ForCausalLM

    with open(os.path.join(ckpt, "config.json")) as f:
        config = Qwen3Config.from_dict(json.load(f))
    config._attn_implementation = attn
    with torch.device("meta"):
        model = Qwen3ForCausalLM(config)
    with open(os.path.join(ckpt, "model.safetensors.index.json")) as f:
        shards = sorted(set(json.load(f)["weight_map"].values()))
    state = {}
    for shard in shards:
        with safe_open(os.path.join(ckpt, shard), "pt") as f:
            state.update({k[len("llm.") :]: f.get_tensor(k).to(dtype) for k in f.keys() if k.startswith("llm.")})
    if config.tie_word_embeddings:
        state.setdefault("lm_head.weight", state["model.embed_tokens.weight"])
    model.load_state_dict(state, strict=True, assign=True)
    if config.tie_word_embeddings:
        model.tie_weights()
    model.model.rotary_emb = type(model.model.rotary_emb)(config=config)
    return model.cuda().train()


_LOADERS = {"qwen3_omni": _load_qwen3_omni, "minicpm_o": _load_minicpm_o}


def reference(ckpt, fields, load):
    thinker = load(ckpt, torch.float32, "sdpa")
    _, _, weights = packed_inputs(fields)
    total, start = None, 0
    for r in ROWS:
        ids = fields["input_ids"][r][None].cuda()
        pos = batched_positions(fields["position_ids"][r]).cuda()
        length = ids.shape[1]
        logits = thinker(input_ids=ids, position_ids=pos, use_cache=False).logits[0, :-1]
        loss = loss_from_logprobs(_logprobs(logits, ids[0, 1:]), weights[start : start + length - 1])
        loss = loss * (weights[start : start + length - 1] != 0).sum() / (weights != 0).sum()
        loss.backward()
        total = loss.item() if total is None else total + loss.item()
        start += length
    grads = _grads(thinker)
    del thinker
    torch.cuda.empty_cache()
    return grads, total


def stock(ckpt, fields, load):
    thinker = load(ckpt, torch.bfloat16, "flash_attention_2")
    ids, pos, weights = packed_inputs(fields)
    weights[boundaries(fields)] = 0
    logits = thinker(input_ids=ids, position_ids=pos, use_cache=False).logits[0, :-1]
    loss = loss_from_logprobs(_logprobs(logits, ids[0, 1:]), weights)
    loss.backward()
    grads = _grads(thinker)
    del thinker
    torch.cuda.empty_cache()
    return grads, loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--family", default="qwen3_omni", choices=sorted(_LOADERS))
    args = parser.parse_args()
    load = _LOADERS[args.family]
    fields = torch.load(args.fixture, weights_only=False)["fields"]
    ref, ref_loss = reference(args.ckpt, fields, load)
    st, st_loss = stock(args.ckpt, fields, load)
    torch.save({"ref": ref, "stock": st, "loss": {"ref": ref_loss, "stock": st_loss}, "rows": list(ROWS)}, args.out)
    print(f"wrote {args.out}: {len(ref)} params, loss ref={ref_loss:.6f} stock={st_loss:.6f}")


if __name__ == "__main__":
    main()
