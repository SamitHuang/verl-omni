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
"""Rollout side of gate g1: rebuild the loop's vLLM-Omni engine and replay a batch.

The engine is built offline from the ``AsyncOmni(**engine_args)`` call that
g2 captured, with only the worker extension wrapped for probing. For each
fixture row the full sequence is prefilled with ``prompt_logprobs`` so the
response tokens get log-probs from one forward pass.

Usage:
    python -m rsi.evaluate.g1_rollout --engine-args E.pkl --fixture F --out O.pt \
        [--cudagraph] [--probe-row 0] [--weights W.pt] [--pipeline P]
"""

import argparse
import os
import pickle
import tempfile


def _response_logprobs(request_output, prompt_len, response):
    plp = request_output.prompt_logprobs
    out = []
    for j, tok in enumerate(response.tolist()):
        entry = plp[prompt_len + j]
        out.append(entry[tok].logprob)
    return out


def _generate(omni, prompts, params):
    from vllm import SamplingParams

    outs = omni.generate([{"prompt_token_ids": p} for p in prompts], [SamplingParams(**params)], use_tqdm=False)
    by_index = {}
    for o in outs:
        ro = getattr(o, "request_output", None) or o
        idx = int(str(ro.request_id).split("_", 1)[0])
        by_index[idx] = ro
    return [by_index[i] for i in range(len(prompts))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-args", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cudagraph", action="store_true", help="override enforce_eager=False")
    parser.add_argument("--batch-copies", type=int, default=4, help="fixture rows repeated into one batch")
    parser.add_argument("--probe-row", type=int, default=None)
    parser.add_argument("--weights", default=None, help="weights_sent_sync{k}.pt from g2, applied after load")
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
    tmp = tempfile.mkdtemp(prefix="rsi_g1_")
    if "deploy_config_text" in captured:
        path = os.path.join(tmp, "deploy.yaml")
        with open(path, "w") as f:
            f.write(captured["deploy_config_text"])
        kwargs["deploy_config"] = path
    base_ext = kwargs.get("worker_extension_cls") or ""
    os.environ["RSI_BASE_WORKER_EXT"] = base_ext
    kwargs["worker_extension_cls"] = "rsi.evaluate.probe_ext.RsiProbeExtension"
    if args.cudagraph:
        kwargs["enforce_eager"] = False
        kwargs.pop("compilation_config", None)

    if args.pipeline:
        # Runtime-registered pipelines (e.g. thinker-only clones) exist only after the adapter registers them.
        import verl_omni.pipelines  # noqa: F401
        from verl_omni.pipelines.model_base import OmniRolloutPipelineBase

        adapter = OmniRolloutPipelineBase.get_class(args.pipeline)
        if adapter is not None:
            adapter.ensure_pipeline_registered("thinker_only")

    from vllm_omni.entrypoints.omni import Omni

    omni = Omni(*captured.get("args", ()), **kwargs)
    if args.weights:
        loaded = omni.engine.collective_rpc("rsi_load_weights", args=(args.weights,), stage_ids=[0])
        print(f"weight sync replay from {args.weights}: {loaded}")
        if any(isinstance(r, dict) and "error" in r for r in loaded):
            raise SystemExit(f"weight sync replay failed: {loaded}")

    fields = torch.load(args.fixture, weights_only=False)["fields"]
    prompts_len = [int(p.shape[-1]) for p in fields["prompts"]]
    seqs = [ids.tolist() for ids in fields["input_ids"]]
    responses = fields["responses"]
    n = len(seqs)
    params = dict(max_tokens=1, temperature=1.0, top_k=-1, top_p=1.0, prompt_logprobs=0, logprobs=0, detokenize=False)

    alone = []
    for i in range(n):
        ro = _generate(omni, [seqs[i]], params)[0]
        alone.append(torch.tensor(_response_logprobs(ro, prompts_len[i], responses[i]), dtype=torch.float32))

    batch_ros = _generate(omni, seqs * args.batch_copies, params)
    in_batch = [
        [
            torch.tensor(_response_logprobs(batch_ros[c * n + i], prompts_len[i], responses[i]), dtype=torch.float32)
            for i in range(n)
        ]
        for c in range(args.batch_copies)
    ]

    result = {
        "prefill_alone": alone,
        "prefill_in_batch": in_batch,
        "engine_kwargs": {k: repr(v) for k, v in kwargs.items()},
    }

    if args.probe_row is not None:
        L = len(seqs[args.probe_row])
        omni.engine.collective_rpc("rsi_probe_start", args=(L,), stage_ids=[0])
        _generate(omni, [seqs[args.probe_row]], params)
        probe_path = os.path.join(tmp, "probe.pt")
        omni.engine.collective_rpc("rsi_probe_stop", args=(probe_path,), stage_ids=[0])
        result["probe"] = torch.load(probe_path, weights_only=False)

    torch.save(result, args.out)
    print(f"wrote {args.out}")
    omni.close()


if __name__ == "__main__":
    main()
