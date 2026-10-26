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
"""Candidate side of the gradient gate.

Builds the candidate's actor exactly as g1 does (so ``configure_model`` and any
candidate patches apply), runs the packed fixture rows through the module and
verl's ``logprobs_from_logits`` as used by the FSDP engine, and compares the
gradients with the cached reference on the GPU. Only per-parameter stats are
written (full gradients would be gigabytes per candidate).

For LoRA the reference gradient of each B is derived from the full-weight
reference (``grad_common.expected_lora_grads``), which requires B == 0.

    python -m rsi.evaluate.g3_grad_actor --worktree W --harness H --ckpt C --fixture F --ref R --out O.pt
"""

import argparse
import json

import torch

from rsi.evaluate.g1_actor import build_worker
from rsi.evaluate.grad_common import (
    boundaries,
    canonical,
    expected_lora_grads,
    grad_stats,
    lora_base_name,
    loss_from_logprobs,
    packed_inputs,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--harness", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lora", default="false", help="scope.lora as JSON")
    parser.add_argument("--family", default="qwen3_omni")
    args = parser.parse_args()

    worker = build_worker(args.worktree, args.harness, args.ckpt, lora=json.loads(args.lora), family=args.family)
    module = worker.actor.engine.module
    module.train()
    fields = torch.load(args.fixture, weights_only=False)["fields"]
    ids, pos, weights = packed_inputs(fields)
    weights[boundaries(fields)] = 0

    import verl.workers.engine.fsdp.transformer_impl as transformer_impl

    engine = worker.actor.engine
    inputs = engine.model_adapter_cls.prepare_model_inputs(
        {"input_ids": ids, "position_ids": pos, "use_cache": False}, None, engine.model_config
    )
    logits = module(**inputs).logits[0, :-1]
    logprobs = transformer_impl.logprobs_from_logits(logits, ids[0, 1:])
    loss = loss_from_logprobs(logprobs.float(), weights)
    loss.backward()
    del logits, logprobs

    def full(t):
        return (t.full_tensor() if hasattr(t, "full_tensor") else t).detach()

    grads, lora_meta, frozen = {}, {}, 0
    for name, p in module.named_parameters():
        if not p.requires_grad:
            frozen += 1
            continue
        cname = canonical(name)
        grads[cname] = None if p.grad is None else full(p.grad)
        if ".lora_B." in cname:
            owner = module.get_submodule(name.rsplit(".lora_B.", 1)[0])
            adapter = name.rsplit(".", 2)[-2]
            lora_meta[cname] = {
                "base": lora_base_name(cname),
                "A": full(getattr(owner.lora_A, adapter).weight).float().cpu(),
                "B_norm": full(p).float().norm().item(),
                "scaling": float(owner.scaling[adapter]),
            }

    ref = torch.load(args.ref, weights_only=False, mmap=True)
    if lora_meta:
        ref_g, stock_g = expected_lora_grads(ref["ref"], lora_meta), expected_lora_grads(ref["stock"], lora_meta)
        cand_g = {n: grads.get(n) for n in lora_meta}
    else:
        ref_g, stock_g, cand_g = ref["ref"], ref["stock"], grads
    stats = grad_stats(ref_g, stock_g, cand_g, device="cuda")
    out = {
        "stats": stats,
        "lora": {n: {k: v for k, v in m.items() if k != "A"} for n, m in lora_meta.items()},
        "trainable": len(grads),
        "frozen": frozen,
        "loss": loss.item(),
    }
    torch.save(out, args.out)
    print(f"wrote {args.out}: {len(stats)} params compared ({len(lora_meta)} LoRA pairs), loss={loss.item():.6f}")


if __name__ == "__main__":
    main()
