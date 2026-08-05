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
"""Compatibility layer for the vllm-omni 0.26 diffusion output envelope.

vllm-omni 0.26 removed ``DiffusionOutput.custom_output`` and stopped
forwarding it in ``DiffusionEngine.postprocess_output``. verl-omni rollout
pipelines ship RL training tensors (per-step latents / log-probs / timesteps,
prompt embeddings) alongside the generated media, so this module restores
that carrier end-to-end:

- :class:`RolloutDiffusionOutput` re-adds the ``custom_output`` field,
  including ``to_cpu`` handling, for pipeline-side construction.
- An import-time patch on ``DiffusionEngine.postprocess_output`` forwards the
  field to ``OmniRequestOutput.custom_output`` on the engine side, preserving
  the consumption path in ``vllm_omni_async_server``.

The engine patch relies on vLLM's default ``fork`` multiprocessing start
method on Linux: ``verl_omni.pipelines`` is imported in the rollout server
process before ``AsyncOmni`` spawns its stage/worker children, so the patch
is inherited everywhere the engine runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from typing import Any

import torch
from vllm_omni.diffusion.data import DiffusionOutput

logger = logging.getLogger(__name__)


def _move_tree_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _move_tree_to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_tree_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_tree_to_cpu(v) for v in value)
    return value


@dataclass
class RolloutDiffusionOutput(DiffusionOutput):
    """``DiffusionOutput`` carrying verl-omni rollout extras.

    ``custom_output`` holds per-request training tensors (``all_latents``,
    ``all_log_probs``, ``all_timesteps``, prompt embeddings, ...). When
    ``to_cpu`` is set, custom tensors are moved to CPU as well, matching the
    pre-0.26 ``DiffusionOutput`` behavior.
    """

    custom_output: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.to_cpu and self.custom_output:
            self.custom_output = {k: _move_tree_to_cpu(v) for k, v in self.custom_output.items()}


def rollout_output_from(output: DiffusionOutput, **changes: Any) -> RolloutDiffusionOutput:
    """Copy a (base or rollout) ``DiffusionOutput`` with *changes* applied.

    Replacement for ``dataclasses.replace(output, custom_output=...)``, which
    no longer works now that ``custom_output`` is not a declared field on the
    base class.
    """
    kwargs = {f.name: getattr(output, f.name) for f in fields(DiffusionOutput)}
    if isinstance(output, RolloutDiffusionOutput):
        kwargs["custom_output"] = output.custom_output
    kwargs.update(changes)
    return RolloutDiffusionOutput(**kwargs)


_ENGINE_PATCHED = False


def _patch_diffusion_engine_custom_output() -> None:
    """Forward ``DiffusionOutput.custom_output`` to ``OmniRequestOutput``.

    vllm-omni 0.26's ``DiffusionEngine.postprocess_output`` drops the rollout
    payload; re-attach it so ``vllm_omni_async_server`` can keep reading
    ``OmniRequestOutput.custom_output``. Idempotent.
    """
    global _ENGINE_PATCHED
    if _ENGINE_PATCHED:
        return
    _ENGINE_PATCHED = True

    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

    original = DiffusionEngine.postprocess_output

    def postprocess_output_with_custom(self, request, output):
        results = original(self, request, output)
        custom = getattr(output, "custom_output", None)
        if custom:
            for result in results:
                result.custom_output = custom
        return results

    DiffusionEngine.postprocess_output = postprocess_output_with_custom


_patch_diffusion_engine_custom_output()
