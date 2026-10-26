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
"""Per-op output capture on the HF actor and the vLLM rollout, and first-divergence search.

Both sides record the output of the same logical ops for one prefill of one
sequence. Compared in forward order, the first op whose output differs while
every earlier output is equal received equal inputs, so it is where the two
implementations diverge.

Op names are ``embed``, ``L{i}.{op}`` for ``input_norm``, ``q_norm``,
``k_norm``, ``attn_core`` (input of ``o_proj``), ``attn_out``, ``post_norm``,
``router``, ``mlp``, ``out`` (layer output; vLLM's ``hidden + residual``),
then ``final_norm`` and ``logits``.
"""

import re

import torch

_SKIP = re.compile(r"audio_tower|visual|talker|code2wav|(?:^|\.)(?:vpm|apm|resampler|tts)(?:\.|$)")

# (regex on module name, op template, capture). ``capture`` is "out" or "in".
_OPS = [
    (re.compile(r"(?:^|\.)embed_tokens$"), "embed", "out"),
    (re.compile(r"(?:^|\.)layers\.(\d+)\.input_layernorm$"), "L{}.input_norm", "out"),
    (re.compile(r"(?:^|\.)layers\.(\d+)\.self_attn\.q_norm$"), "L{}.q_norm", "out"),
    (re.compile(r"(?:^|\.)layers\.(\d+)\.self_attn\.k_norm$"), "L{}.k_norm", "out"),
    (re.compile(r"(?:^|\.)layers\.(\d+)\.self_attn\.o_proj$"), "L{}.attn_core", "in"),
    (re.compile(r"(?:^|\.)layers\.(\d+)\.self_attn\.o_proj$"), "L{}.attn_out", "out"),
    (re.compile(r"(?:^|\.)layers\.(\d+)\.post_attention_layernorm$"), "L{}.post_norm", "out"),
    (re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.gate$"), "L{}.router", "out"),
    (re.compile(r"(?:^|\.)layers\.(\d+)\.mlp$"), "L{}.mlp", "out"),
    (re.compile(r"(?:^|\.)layers\.(\d+)$"), "L{}.out", "layer"),
    (re.compile(r"(?:^|\.)model\.norm$|(?:^|\.)language_model\.norm$"), "final_norm", "out"),
    (re.compile(r"(?:^|\.)lm_head$|(?:^|\.)logits_processor$"), "logits", "out"),
]


def op_order(num_layers):
    per_layer = ["input_norm", "q_norm", "k_norm", "attn_core", "attn_out", "post_norm", "router", "mlp", "out"]
    return ["embed", *[f"L{i}.{op}" for i in range(num_layers) for op in per_layer], "final_norm", "logits"]


def _first_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, tuple | list) and x and isinstance(x[0], torch.Tensor):
        return x[0]
    return None


def _layer_output(x):
    # vLLM decoder layers return (hidden_states, residual); the layer's full
    # output is their sum. HF returns the full hidden state (maybe in a tuple).
    if isinstance(x, tuple | list) and len(x) == 2 and all(isinstance(t, torch.Tensor) for t in x):
        if x[0].shape == x[1].shape:
            return x[0] + x[1]
    return _first_tensor(x)


def _to_tokens(t, num_tokens):
    t = t.detach()
    if t.dim() >= 2 and t.shape[0] == 1 and t.shape[1] == num_tokens:
        t = t[0]
    if t.shape[0] < num_tokens:
        return None
    return t[:num_tokens].reshape(num_tokens, -1).cpu()


class Recorder:
    """Record the first output of each logical op on ``module``."""

    def __init__(self, module, num_tokens):
        self.module = module
        self.num_tokens = num_tokens
        self.records = {}
        self._handles = []

    def _store(self, name, tensor):
        if tensor is None or name in self.records:
            return
        t = _to_tokens(tensor, self.num_tokens)
        if t is not None:
            self.records[name] = t

    def __enter__(self):
        for mod_name, mod in self.module.named_modules():
            if _SKIP.search(mod_name):
                continue
            for pattern, template, capture in _OPS:
                m = pattern.search(mod_name)
                if not m:
                    continue
                op = template.format(*m.groups())
                if capture == "in":
                    self._handles.append(
                        mod.register_forward_pre_hook(lambda _m, args, op=op: self._store(op, _first_tensor(args)))
                    )
                elif capture == "layer":
                    self._handles.append(
                        mod.register_forward_hook(lambda _m, _a, out, op=op: self._store(op, _layer_output(out)))
                    )
                else:
                    self._handles.append(
                        mod.register_forward_hook(lambda _m, _a, out, op=op: self._store(op, _first_tensor(out)))
                    )
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False


class ActorProbe(Recorder):
    """Recorder for the actor; the token count is taken from the first embed call."""

    def __init__(self, module):
        super().__init__(module, num_tokens=None)

    def _store(self, name, tensor):
        if tensor is not None and self.num_tokens is None and name == "embed":
            t = tensor.detach()
            self.num_tokens = t.shape[1] if t.dim() == 3 else t.shape[0]
        if self.num_tokens is None:
            return
        super()._store(name, tensor)


class ProbeWorkerExtension:
    """vLLM worker extension: ``collective_rpc("rsi_probe_start", args=(num_tokens,))``."""

    def _rsi_model(self):
        runner = self.model_runner
        return runner.get_model() if hasattr(runner, "get_model") else runner.model

    def rsi_probe_start(self, num_tokens):
        self._rsi_recorder = Recorder(self._rsi_model(), num_tokens).__enter__()
        return True

    def rsi_probe_stop(self, path):
        rec = self._rsi_recorder
        rec.__exit__(None, None, None)
        torch.save(rec.records, path)
        return sorted(rec.records)

    def rsi_load_weights(self, path, bucket_bytes=1 << 30):
        """Replay a captured weight sync through the loop's own ``update_weights_from_ipc``.

        Only the IPC receiver is replaced, by one that yields the captured tensors in buckets.
        """
        import verl.workers.rollout.vllm_rollout.bucketed_weight_transfer as transfer

        dump = torch.load(path, weights_only=False)
        if dump["peft"]:
            raise NotImplementedError("unmerged LoRA sync replay is not supported")
        device = torch.device("cuda", torch.cuda.current_device())
        items = list(dump["tensors"].items())

        class _Replay:
            def __init__(self, *args, **kwargs):
                pass

            def receive_weights(self, on_bucket_received):
                bucket, size = [], 0
                for i, (name, t) in enumerate(items):
                    bucket.append((name, t.to(device)))
                    size += t.numel() * t.element_size()
                    last = i == len(items) - 1
                    if size >= bucket_bytes or last:
                        on_bucket_received(bucket, is_last=last)
                        bucket, size = [], 0

        original = transfer.BucketedWeightReceiver
        transfer.BucketedWeightReceiver = _Replay
        try:
            self.update_weights_from_ipc(peft_config=None, base_sync_done=False)
        finally:
            transfer.BucketedWeightReceiver = original
        return {"tensors": len(items), "changed": len(dump.get("changed", []))}


def first_divergence(actor, rollout, num_layers):
    """Compare per-op records in forward order.

    Returns ``(first, rows)`` where ``first`` is the first op that differs
    (or None) and ``rows`` lists every op with its comparison.
    """
    rows, first = [], None
    for op in op_order(num_layers):
        a, r = actor.get(op), rollout.get(op)
        if a is None or r is None:
            rows.append({"op": op, "status": "missing", "actor": a is not None, "rollout": r is not None})
            continue
        if a.shape != r.shape:
            rows.append({"op": op, "status": "shape", "actor": list(a.shape), "rollout": list(r.shape)})
            if first is None:
                first = op
            continue
        same_dtype = a.dtype == r.dtype
        neq = (a != r) if same_dtype else (a.float() != r.float())
        diff = (a.double() - r.double()).abs()
        row = {
            "op": op,
            "status": "equal" if same_dtype and not neq.any() else "differs",
            "dtype": [str(a.dtype), str(r.dtype)],
            "elements_differ": int(neq.sum()),
            "tokens_differ": int(neq.any(-1).sum()),
            "tokens": a.shape[0],
            "abs_diff_max": float(diff.max()),
        }
        rows.append(row)
        if row["status"] != "equal" and first is None:
            first = op
    return first, rows
