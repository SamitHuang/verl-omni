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
"""Inputs and loss shared by the gradient gate's reference and candidate sides.

The loss is a policy-gradient surrogate on fixed rows of a captured batch:
``-sum(adv_row * logprob(token)) / num_response_tokens`` over response tokens,
with a fixed per-row advantage of +1 or -1.
"""

import re

import torch

ROWS = (0, 1)
ADVANTAGES = (1.0, -1.0)
_WRAPPERS = re.compile(r"(_fsdp_wrapped_module|_checkpoint_wrapped_module|_orig_mod)\.")


def batched_positions(pos):
    """Per-row ``position_ids`` ([T], or [C, T] for M-RoPE) with the batch axis added: [1, T] or [C, 1, T]."""
    return pos[None] if pos.dim() == 1 else pos[:, None]


def packed_inputs(fields, rows=ROWS, device="cuda"):
    """Packed ``input_ids`` [1, T], ``position_ids`` ([1, T] or [C, 1, T]), and per-label-position weights [T-1]."""
    ids = torch.cat([fields["input_ids"][r] for r in rows])[None].to(device)
    pos = batched_positions(torch.cat([fields["position_ids"][r] for r in rows], dim=-1)).to(device)
    weights = torch.zeros(ids.shape[1] - 1, device=device)
    start = 0
    for r, adv in zip(rows, ADVANTAGES, strict=True):
        length = fields["input_ids"][r].shape[-1]
        mask = fields["response_mask"][r].to(device).float()
        prompt = length - mask.shape[0]
        # Label index j predicts token j + 1; response token k sits at start + prompt + k.
        weights[start + prompt - 1 : start + length - 1] = adv * mask
        start += length
    return ids, pos, weights


def boundaries(fields, rows=ROWS):
    """Label positions that predict the first token of the next packed row (excluded from the loss)."""
    out, start = [], 0
    for r in rows[:-1]:
        start += fields["input_ids"][r].shape[-1]
        out.append(start - 1)
    return out


def loss_from_logprobs(logprobs, weights):
    return -(logprobs * weights).sum() / (weights != 0).sum()


def canonical(name):
    """Parameter name relative to the language model, without FSDP / checkpoint / PEFT wrapper prefixes.

    The language model is ``thinker`` in Qwen3-Omni and ``llm`` in MiniCPM-o.
    """
    name = _WRAPPERS.sub("", name)
    name = name[len("base_model.model.") :] if name.startswith("base_model.model.") else name
    for prefix in ("thinker.", "llm."):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def lora_base_name(name):
    """``...q_proj.lora_B.default.weight`` -> ``...q_proj.weight`` (canonical), else None."""
    m = re.match(r"(.*)\.lora_[AB]\.[^.]+\.weight$", name)
    return f"{m.group(1)}.weight" if m else None


def grad_stats(ref, stock, cand, device="cpu"):
    """Per-parameter stats for the gate: cosines to ``ref`` and squared norms.

    ``cand`` maps names to tensors or None (no gradient). Only names in ``ref``
    with a nonzero reference gradient are reported.
    """
    import torch.nn.functional as F

    out = {}
    for name, r in ref.items():
        r = r.to(device, torch.float64)
        r2 = r.pow(2).sum().item()
        if r2 == 0:
            continue
        s = stock[name].to(device, torch.float64)
        c = cand.get(name)
        row = {
            "r_norm2": r2,
            "s_norm2": s.pow(2).sum().item(),
            "cos_s": F.cosine_similarity(s.flatten(), r.flatten(), dim=0).item(),
        }
        if c is None or not c.any():
            row.update(c_norm2=0.0, cos_c=None)
        else:
            c = c.to(device, torch.float64)
            row.update(c_norm2=c.pow(2).sum().item(), cos_c=F.cosine_similarity(c.flatten(), r.flatten(), dim=0).item())
        out[name] = row
    return out


def expected_lora_grads(full_grads, lora_meta):
    """LoRA-B gradients implied by full-weight gradients when B == 0 (merged weight == base weight).

    With ``W' = W + s * B @ A``: ``dL/dB = s * dL/dW' @ A^T``; ``dL/dA = s * B^T @ dL/dW' = 0``.
    """
    return {
        name: meta["scaling"] * full_grads[meta["base"]].double() @ meta["A"].double().T
        for name, meta in lora_meta.items()
    }
