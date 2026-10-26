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
"""Actor side of the benchmark: log-prob recompute and forward+backward throughput.

Builds the candidate's actor as g1 does (the g2 Hydra config) on ``--ckpt``
and tiles the fixture rows to about ``--tokens`` tokens per rank. Under
torchrun every rank runs the same batch through FSDP2, so the timings include
its all-gathers and reduce-scatters and whatever NCCL settings the candidate's
code leaves in place; throughput counts the tokens of all ranks. Times:

- ``logprob``: ``worker.compute_log_prob`` on the whole batch (the loop's
  old-log-prob path, dynamic micro-batching included);
- ``fwd_bwd``: module forward, verl's ``logprobs_from_logits`` and backward of
  the mean log-prob, on packed chunks of at most ``--chunk-tokens`` tokens.
  No optimizer step, so the weights never change between repeats.

Usage:
    python -m torch.distributed.run --nproc-per-node N -m rsi.evaluate.bench_actor \
        --worktree W --harness H --ckpt C --fixture F --out O.json
"""

import argparse
import json
import statistics
import time


def _tile(fields, copies):
    n = len(fields["input_ids"])
    return {
        k: [v[i % n] for i in range(n * copies)]
        for k, v in fields.items()
        if hasattr(v, "__getitem__") and not isinstance(v, str | bytes | dict)
    }


def _chunks(fields, limit):
    out, cur, size = [], [], 0
    for i, ids in enumerate(fields["input_ids"]):
        length = int(ids.shape[-1])
        if cur and size + length > limit:
            out.append(cur)
            cur, size = [], 0
        cur.append(i)
        size += length
    return out + ([cur] if cur else [])


def _sync():
    import torch
    import torch.distributed as dist

    torch.cuda.synchronize()
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()


def _timed(fn, repeats):
    fn()  # warmup
    walls = []
    for _ in range(repeats):
        _sync()
        start = time.perf_counter()
        fn()
        _sync()
        walls.append(time.perf_counter() - start)
    return walls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--harness", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--chunk-tokens", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--lora", default="false", help="scope.lora as JSON")
    parser.add_argument("--family", default="qwen3_omni")
    args = parser.parse_args()

    import torch

    from rsi.evaluate.g1_actor import build_td, build_worker
    from rsi.evaluate.grad_common import batched_positions

    worker = build_worker(args.worktree, args.harness, args.ckpt, lora=json.loads(args.lora), family=args.family)
    raw = torch.load(args.fixture, weights_only=False)["fields"]
    per_copy = sum(int(t.shape[-1]) for t in raw["input_ids"])
    fields = _tile(raw, max(1, round(args.tokens / per_copy)))
    rows = list(range(len(fields["input_ids"])))
    tokens = sum(int(t.shape[-1]) for t in fields["input_ids"])

    td = build_td(fields, rows)
    logprob_walls = _timed(lambda: worker.compute_log_prob(td), args.repeats)

    import verl.workers.engine.fsdp.transformer_impl as transformer_impl

    engine = worker.actor.engine
    module = engine.module
    module.train()
    chunks = _chunks(fields, args.chunk_tokens)

    def fwd_bwd():
        module.zero_grad(set_to_none=True)
        for chunk in chunks:
            ids = torch.cat([fields["input_ids"][r] for r in chunk])[None].cuda()
            pos = batched_positions(torch.cat([fields["position_ids"][r] for r in chunk], dim=-1)).cuda()
            inputs = engine.model_adapter_cls.prepare_model_inputs(
                {"input_ids": ids, "position_ids": pos, "use_cache": False}, None, engine.model_config
            )
            logits = module(**inputs).logits[0, :-1]
            logprobs = transformer_impl.logprobs_from_logits(logits, ids[0, 1:])
            loss = -logprobs.float().mean()
            loss.backward()
            del logits, logprobs, loss

    fwd_bwd_walls = _timed(fwd_bwd, args.repeats)

    import torch.distributed as dist

    world = dist.get_world_size() if dist.is_initialized() else 1
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    def summary(walls):
        med = statistics.median(walls)
        return {"wall_s": walls, "tok_s": world * tokens / med, "spread": (max(walls) - min(walls)) / med}

    out = {
        "world_size": world,
        "tokens": world * tokens,
        "rows": len(rows),
        "chunks": len(chunks),
        "max_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "logprob": summary(logprob_walls),
        "fwd_bwd": summary(fwd_bwd_walls),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(
        f"logprob {out['logprob']['tok_s']:.0f} tok/s, fwd_bwd {out['fwd_bwd']['tok_s']:.0f} tok/s, "
        f"{tokens} tokens, peak {out['max_memory_gib']:.1f} GiB"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
