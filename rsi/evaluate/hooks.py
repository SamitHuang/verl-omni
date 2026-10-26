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
"""Evaluator-owned capture hooks for the verl training loop.

Loaded in every verl process through ``VERL_USE_EXTERNAL_MODULES`` and active
only when ``RSI_DUMP_DIR`` is set. Patches are installed lazily when their
target module is first imported, so loading this module never imports the
trainer itself.

Captures:

- per training step, every TransferQueue field of the batch right after
  ``PPOTrainer._compute_old_log_prob`` (token ids, position ids, masks,
  ``rollout_log_probs`` and ``old_log_probs``) to
  ``$RSI_DUMP_DIR/batch_step{N}.pt``;
- per weight sync, a BLAKE2b digest of every tensor the actor sends
  (``weights_sent_*.json``, rank 0) and every tensor the rollout receives
  (``weights_recv_*.json``);
- at sync ``$RSI_STATE_SYNC`` (default 1, the weights step 2 runs with), the
  actor's gathered trainable parameters (``actor_state_sync{k}.pt``) and every
  tensor sent at that sync (``weights_sent_sync{k}.pt``), so g1 can replay that
  step offline;
- the exact ``AsyncOmni(**engine_args)`` call of each rollout server, plus its
  deploy config text and ``VLLM*`` environment (``engine_args_*.pkl``), so g1
  can rebuild the same engine offline.
"""

import functools
import hashlib
import importlib.abc
import json
import os
import sys
import time

DUMP_DIR_ENV = "RSI_DUMP_DIR"
STATE_SYNC_ENV = "RSI_STATE_SYNC"

_PATCHES = {}


def _on_import(module_name):
    def register(fn):
        _PATCHES[module_name] = fn
        return fn

    return register


class _PostImportFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in _PATCHES:
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            find_spec = getattr(finder, "find_spec", None)
            spec = find_spec(fullname, path, target) if find_spec else None
            if spec is not None and spec.loader is not None:
                break
        else:
            return None
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            _PATCHES[fullname](module)

        spec.loader.exec_module = exec_and_patch
        return spec


def _to_cpu(value):
    import torch

    if isinstance(value, torch.Tensor):
        if value.is_nested:
            return [t.detach().cpu() for t in value.unbind()]
        return value.detach().cpu()
    return None


@_on_import("verl.trainer.ppo.v1.trainer_base")
def _patch_trainer_base(module):
    original = module.PPOTrainer._compute_old_log_prob

    @functools.wraps(original)
    def _compute_old_log_prob(self, batch, metrics):
        import torch

        out = original(self, batch, metrics)
        data = module.tq.kv_batch_get(keys=out.keys, partition_id=out.partition_id, select_fields=None)
        fields = {}
        for key in data.keys():
            converted = _to_cpu(data.get(key))
            if converted is not None:
                fields[key] = converted
        step = getattr(self, "global_steps", -1)
        dump_dir = os.environ[DUMP_DIR_ENV]
        os.makedirs(dump_dir, exist_ok=True)
        torch.save({"step": step, "fields": fields}, os.path.join(dump_dir, f"batch_step{step}.pt"))
        return out

    module.PPOTrainer._compute_old_log_prob = _compute_old_log_prob


def _digest(tensor):
    import torch

    t = tensor.detach().contiguous().cpu().reshape(-1).view(torch.uint8)
    return f"{tensor.dtype}:{tuple(tensor.shape)}:" + hashlib.blake2b(t.numpy().tobytes(), digest_size=16).hexdigest()


def _write_digests(kind, digests):
    dump_dir = os.environ[DUMP_DIR_ENV]
    os.makedirs(dump_dir, exist_ok=True)
    name = f"weights_{kind}_{time.time_ns()}_{os.getpid()}.json"
    with open(os.path.join(dump_dir, name), "w") as f:
        json.dump({"pid": os.getpid(), "time_ns": time.time_ns(), "digests": digests}, f)


def _dump_trainable(engine, path, is_rank0):
    """Full (gathered) trainable parameters of the actor; collective, so every rank calls it."""
    import torch
    from torch.distributed.tensor import DTensor

    state = {}
    device = torch.cuda.current_device()
    for name, p in engine.module.named_parameters():
        if not p.requires_grad:
            continue
        full = p.to(device).full_tensor() if isinstance(p, DTensor) else p
        if is_rank0:
            state[name] = full.detach().cpu().clone()
    if is_rank0:
        torch.save(state, path)


@_on_import("verl_omni.workers.engine.fsdp.omni_impl")
def _patch_actor_send(module):
    cls = module.OmniFSDPEngine
    original = cls.get_per_tensor_param

    @functools.wraps(original)
    def get_per_tensor_param(self, *args, **kwargs):
        import torch
        import torch.distributed as dist

        sync = self._rsi_sync = getattr(self, "_rsi_sync", -1) + 1
        is_rank0 = not dist.is_initialized() or dist.get_rank() == 0
        dump_dir = os.environ[DUMP_DIR_ENV]
        # Sync k carries the weights step k+1 runs with; g1 replays that step from these dumps.
        capture = sync == int(os.environ.get(STATE_SYNC_ENV, "1"))
        if capture:
            os.makedirs(dump_dir, exist_ok=True)
            _dump_trainable(self, os.path.join(dump_dir, f"actor_state_sync{sync}.pt"), is_rank0)
        per_tensor, peft = original(self, *args, **kwargs)
        if not is_rank0:
            return per_tensor, peft

        def tap():
            digests, sent = {}, {}
            first = getattr(self, "_rsi_first_digests", None)
            for name, tensor in per_tensor:
                digests[name] = _digest(tensor)
                if capture:
                    sent[name] = tensor.detach().cpu().clone()
                yield name, tensor
            if first is None:
                self._rsi_first_digests = digests
            if capture:
                # Everything, not only what changed: fused-MoE reload in vLLM works on whole layers.
                changed = sorted(n for n in digests if first is None or first.get(n) != digests[n])
                torch.save(
                    {"sync": sync, "peft": peft is not None, "tensors": sent, "changed": changed},
                    os.path.join(dump_dir, f"weights_sent_sync{sync}.pt"),
                )
            _write_digests("sent", digests)

        return tap(), peft

    cls.get_per_tensor_param = get_per_tensor_param


@_on_import("verl.workers.rollout.vllm_rollout.bucketed_weight_transfer")
def _patch_rollout_recv(module):
    cls = module.BucketedWeightReceiver
    original = cls.receive_weights

    @functools.wraps(original)
    def receive_weights(self, on_bucket_received, *args, **kwargs):
        digests = {}

        def tapped(weights, *cb_args, **cb_kwargs):
            items = weights.items() if isinstance(weights, dict) else weights
            for name, tensor in items:
                digests[name] = _digest(tensor)
            return on_bucket_received(weights, *cb_args, **cb_kwargs)

        result = original(self, tapped, *args, **kwargs)
        _write_digests("recv", digests)
        return result

    cls.receive_weights = receive_weights


@_on_import("vllm_omni.entrypoints.async_omni")
def _patch_async_omni(module):
    cls = module.AsyncOmni
    original = cls.__init__

    @functools.wraps(original)
    def __init__(self, *args, **kwargs):
        import pickle

        dump_dir = os.environ[DUMP_DIR_ENV]
        os.makedirs(dump_dir, exist_ok=True)
        captured = {
            "args": args,
            "kwargs": kwargs,
            "env": {k: v for k, v in os.environ.items() if k.startswith("VLLM")},
        }
        deploy = kwargs.get("deploy_config")
        if isinstance(deploy, str) and os.path.isfile(deploy):
            with open(deploy) as f:
                captured["deploy_config_text"] = f.read()
        path = os.path.join(dump_dir, f"engine_args_{os.getpid()}.pkl")
        try:
            with open(path, "wb") as f:
                pickle.dump(captured, f)
        except Exception as e:  # unpicklable values are recorded by repr
            captured["kwargs"] = {k: repr(v) for k, v in kwargs.items()}
            captured["pickle_error"] = repr(e)
            with open(path, "wb") as f:
                pickle.dump(captured, f)
        original(self, *args, **kwargs)

    cls.__init__ = __init__


def install():
    if not os.environ.get(DUMP_DIR_ENV):
        return
    if any(isinstance(f, _PostImportFinder) for f in sys.meta_path):
        return
    for name, patch in _PATCHES.items():
        if name in sys.modules:
            patch(sys.modules[name])
    sys.meta_path.insert(0, _PostImportFinder())


install()
