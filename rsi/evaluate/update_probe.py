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
"""Is the actor's forward inside ``update_actor`` bitwise equal to its ``compute_log_prob``?

Bypass mode (old log-probs = rollout log-probs) is exactly on-policy only if it is.
Runs the captured step-1 batch through the candidate's actor in four ways and
compares response log-probs with ``!=`` against ``compute_log_prob``:

* ``train_nograd_infer_mb``: train mode, no autograd, inference micro-batching
* ``train_nograd``: train mode, no autograd, training micro-batching
* ``train_grad``: train mode with autograd and backward (as in the update), training micro-batching
* ``rollout``: the rollout's own log-probs from the capture

    python -m rsi.evaluate.update_probe --worktree W --harness H --fixture batch_step1.pt --ckpt C --out o.json
"""

import argparse
import json

from rsi.evaluate.g1_actor import build_td, build_worker


def _response(td, log_probs):
    from verl.workers.utils.padding import response_from_nested

    resp = response_from_nested(log_probs, td["response_mask"])
    return [r.detach().float().cpu() for r in (resp.unbind() if resp.is_nested else resp.unbind(0))]


def _engine_pass(worker, fields, rows, max_token_len, grad):
    import torch
    from verl.utils import tensordict_utils as tu

    actor = worker.actor
    td = build_td(fields, rows)
    tu.assign_non_tensor(
        td,
        use_remove_padding=actor.model_config.get("use_remove_padding", False),
        use_dynamic_bsz=actor.engine_config.use_dynamic_bsz,
        max_token_len_per_gpu=max_token_len,
        micro_batch_size_per_gpu=actor.engine_config.micro_batch_size_per_gpu,
        use_fused_kernels=actor.engine_config.use_fused_kernels,
        calculate_entropy=False,
        return_model_output=True,
    )

    def loss_fn(model_output, data, dp_group=None):
        return model_output["log_probs"].values().sum() * 0.0, {}

    engine = actor.engine
    with engine.train_mode():
        engine.optimizer_zero_grad()
        if grad:
            out = engine.forward_backward_batch(td, loss_fn, forward_only=False)
        else:
            with torch.no_grad():
                out = engine.forward_backward_batch(td, None, forward_only=True)
        engine.optimizer_zero_grad()
    return _response(td, out["model_output"]["log_probs"])


def _compare(ref, other):
    mismatched = sum(int((a != b).sum()) for a, b in zip(ref, other, strict=True))
    total = sum(a.numel() for a in ref)
    max_abs = max(float((a - b).abs().max()) if a.numel() else 0.0 for a, b in zip(ref, other, strict=True))
    return {"mismatch_total": mismatched, "tokens_compared": total, "abs_diff_max": max_abs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--harness", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lora", default="false")
    parser.add_argument("--family", default="qwen3_omni")
    args = parser.parse_args()

    import torch
    from verl.utils import tensordict_utils as tu

    worker = build_worker(args.worktree, args.harness, args.ckpt, lora=json.loads(args.lora), family=args.family)
    fields = torch.load(args.fixture, weights_only=False)["fields"]
    rows = list(range(len(fields["input_ids"])))

    td = build_td(fields, rows)
    infer = _response(td, tu.get(worker.compute_log_prob(td), "log_probs"))
    cfg = worker.actor.engine_config
    passes = {
        "train_nograd_infer_mb": _engine_pass(worker, fields, rows, cfg.infer_max_token_len_per_gpu, grad=False),
        "train_nograd": _engine_pass(worker, fields, rows, cfg.max_token_len_per_gpu, grad=False),
        "train_grad": _engine_pass(worker, fields, rows, cfg.max_token_len_per_gpu, grad=True),
    }
    result = {
        "rows": len(rows),
        "max_token_len": {"infer": cfg.infer_max_token_len_per_gpu, "train": cfg.max_token_len_per_gpu},
    }
    result.update({name: _compare(infer, lp) for name, lp in passes.items()})
    if "rollout_log_probs" in fields:
        rollout = [fields["rollout_log_probs"][i].float()[: len(infer[i])] for i in rows]
        result["rollout"] = _compare(infer, rollout)
    print(json.dumps(result, indent=2))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
