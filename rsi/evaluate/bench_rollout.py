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
"""Rollout side of the benchmark: decode throughput of the candidate's engine.

Rebuilds the engine from the ``AsyncOmni(**engine_args)`` call that g2 captured,
with the model swapped for ``--ckpt`` (full depth). The fixture's prompts are
tiled to each batch size and decoded with ``ignore_eos`` to exactly
``--max-tokens`` tokens, so every candidate generates the same token count.
Sampling matches the loop (temperature 1, log-prob of the sampled token).

Usage:
    python -m rsi.evaluate.bench_rollout --engine-args E.pkl --fixture F --ckpt C --out O.json [--cudagraph]
"""

import argparse
import json
import os
import pickle
import statistics
import tempfile
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-args", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cudagraph", action="store_true", help="override enforce_eager=False")
    parser.add_argument("--batch-sizes", default="32,128")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument(
        "--pipeline", default=None, help="verl-omni rollout adapter (engine_kwargs.vllm_omni.pipeline_name)"
    )
    args = parser.parse_args()

    import torch

    with open(args.engine_args, "rb") as f:
        captured = pickle.load(f)
    kwargs = dict(captured["kwargs"])
    for k, v in captured.get("env", {}).items():
        os.environ.setdefault(k, v)
    if "deploy_config_text" in captured:
        path = os.path.join(tempfile.mkdtemp(prefix="rsi_bench_"), "deploy.yaml")
        with open(path, "w") as f:
            f.write(captured["deploy_config_text"])
        kwargs["deploy_config"] = path
    kwargs["model"] = args.ckpt
    kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.cudagraph:
        kwargs["enforce_eager"] = False
        kwargs.pop("compilation_config", None)

    if args.pipeline:
        import verl_omni.pipelines  # noqa: F401
        from verl_omni.pipelines.model_base import OmniRolloutPipelineBase

        adapter = OmniRolloutPipelineBase.get_class(args.pipeline)
        if adapter is not None:
            adapter.ensure_pipeline_registered("thinker_only")

    from vllm import SamplingParams
    from vllm_omni.entrypoints.omni import Omni

    t0 = time.perf_counter()
    omni = Omni(*captured.get("args", ()), **kwargs)
    startup_s = time.perf_counter() - t0

    fields = torch.load(args.fixture, weights_only=False)["fields"]
    unique, seen = [], set()
    for p in fields["prompts"]:
        key = tuple(p.tolist())
        if key not in seen:
            seen.add(key)
            unique.append(list(key))
    params = SamplingParams(
        max_tokens=args.max_tokens,
        min_tokens=args.max_tokens,
        ignore_eos=True,
        temperature=1.0,
        top_k=-1,
        top_p=1.0,
        logprobs=0,
        detokenize=False,
    )

    def run(bs):
        prompts = [{"prompt_token_ids": unique[i % len(unique)]} for i in range(bs)]
        start = time.perf_counter()
        outs = omni.generate(prompts, [params], use_tqdm=False)
        wall = time.perf_counter() - start
        tokens = sum(len((getattr(o, "request_output", None) or o).outputs[0].token_ids) for o in outs)
        return wall, tokens

    results = {}
    for bs in [int(b) for b in args.batch_sizes.split(",")]:
        run(bs)  # warmup: first-shape compilation, allocator growth
        walls, tokens = [], None
        for _ in range(args.repeats):
            wall, n = run(bs)
            walls.append(wall)
            if tokens is not None and n != tokens:
                raise SystemExit(f"token count changed between repeats: {tokens} vs {n}")
            tokens = n
        if tokens != bs * args.max_tokens:
            raise SystemExit(f"expected {bs * args.max_tokens} tokens at batch {bs}, got {tokens}")
        med = statistics.median(walls)
        results[str(bs)] = {
            "tokens": tokens,
            "wall_s": walls,
            "tok_s": tokens / med,
            "spread": (max(walls) - min(walls)) / med,
        }
        print(f"batch {bs}: {tokens / med:.0f} tok/s (walls {[round(w, 2) for w in walls]})", flush=True)

    out = {
        "mode": "cudagraph" if args.cudagraph else "eager",
        "startup_s": startup_s,
        "max_tokens": args.max_tokens,
        "prompts": len(unique),
        "batches": results,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")
    omni.close()


if __name__ == "__main__":
    main()
