#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Performance benchmark for Qwen-Image text-to-image rollout.

Compares the **vllm-omni** rollout backend (the one used by
``examples/flowgrpo_trainer/run_qwen_image_ocr*.sh``) against a plain
**diffusers** ``QwenImagePipeline``. Both backends generate the same workload:

    total_images  = num_prompts * rollout_n

This mirrors the flowgrpo end-to-end Qwen-Image OCR example, where each
training step processes ``train_batch_size * rollout.n`` images (e.g.
``32 * 16 = 512``).  For a single-GPU benchmark the defaults are scaled
down (``num_prompts=2``, ``rollout_n=16`` → 32 images), and can be tuned
via CLI flags.  All other knobs (resolution, ``max_sequence_length``,
``num_inference_steps``, ``true_cfg_scale``) default to values consistent
with ``run_qwen_image_ocr.sh`` (resolution overridden to 512 per task).

The script logs:

* The diffusion attention backend in use for each engine
  (vllm-omni: ``DIFFUSION_ATTENTION_BACKEND`` selection;
  diffusers: ``_AttentionBackendRegistry._active_backend``).
* Setup, warmup and steady-state generation times.
* Throughput (images/s) and per-image latency.
* Peak GPU memory.

Example usage (run from the repo root)::

    # vllm-omni only (one B=N forward per prompt)
    python scripts/bench_qwen_image_rollout.py --backend vllm_omni \
        --vllm-omni-mode batched --num-prompts 2 --rollout-n 16 --iters 3

    # vllm-omni only (N concurrent B=1 requests per prompt, mirrors agent loop)
    python scripts/bench_qwen_image_rollout.py --backend vllm_omni \
        --vllm-omni-mode concurrent --num-prompts 2 --rollout-n 16 --iters 3

    # diffusers only
    python scripts/bench_qwen_image_rollout.py --backend diffusers \
        --num-prompts 2 --rollout-n 16 --iters 3

    # both (sequential; vllm-omni first, then diffusers)
    python scripts/bench_qwen_image_rollout.py --backend both \
        --num-prompts 1 --rollout-n 8 --iters 2 \
        --output bench_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import torch

# ---------------------------------------------------------------------------
# Defaults aligned with examples/flowgrpo_trainer/run_qwen_image_ocr.sh.
#   - rollout.n            = 16
#   - num_inference_steps  = 50 (val_kwargs.pipeline.num_inference_steps)
#   - max_sequence_length  = 256
#   - true_cfg_scale       = 1.0 (non-CFG variant)
# Image resolution is fixed to 512x512 per the perf-test request.
# ---------------------------------------------------------------------------

DEFAULT_PROMPTS: list[str] = [
    "a beautiful sunset over the ocean with vibrant orange and purple clouds "
    "reflecting on the calm water surface near a rocky coastline",
    "a fluffy orange cat sitting on a wooden windowsill looking outside at "
    "a garden full of colorful flowers on a bright sunny afternoon",
    "a majestic mountain landscape covered with fresh white snow under a "
    "clear blue sky with pine trees in the foreground and a frozen lake",
    "a futuristic city at night with neon lights glowing on tall glass "
    "skyscrapers and flying vehicles soaring between the buildings",
    "a peaceful japanese garden with cherry blossom trees in full bloom "
    "surrounding a small wooden bridge over a koi pond at sunrise",
    "a steaming bowl of ramen noodles topped with a soft-boiled egg, "
    "scallions, and slices of pork on a wooden table in a cozy restaurant",
    "a vintage red sports car parked on a cobblestone street in a quaint "
    "european town with old buildings and flower boxes in the windows",
    "a friendly golden retriever puppy playing fetch in a green grassy "
    "field with a yellow tennis ball on a warm summer afternoon",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    backend: str
    num_prompts: int
    rollout_n: int
    total_images: int
    height: int
    width: int
    num_inference_steps: int
    true_cfg_scale: float
    max_sequence_length: int
    dtype: str
    diffusion_attention_backend: str
    setup_time_s: float
    warmup_time_s: float
    bench_times_s: list[float]
    mean_iter_time_s: float
    median_iter_time_s: float
    throughput_images_per_s: float
    seconds_per_image: float
    peak_gpu_memory_mb: float
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def _setup_logging() -> None:
    """Install a dedicated handler on the ``bench`` logger.

    Ray and vLLM reconfigure the root logger during their initialization, which
    can swallow subsequent ``logging.basicConfig`` output. Attaching our own
    handler with ``propagate=False`` ensures ``bench`` log records survive.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    bench_logger = logging.getLogger("bench")
    bench_logger.setLevel(logging.INFO)
    bench_logger.propagate = False
    if not any(isinstance(h, _BenchStreamHandler) for h in bench_logger.handlers):
        handler = _BenchStreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                fmt="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        bench_logger.addHandler(handler)


class _BenchStreamHandler(logging.StreamHandler):
    """Marker subclass so we can detect (and avoid double-adding) our handler."""


def _peak_gpu_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def _summarize(times: list[float], total_images: int) -> tuple[float, float, float, float]:
    mean_t = sum(times) / len(times)
    med_t = statistics.median(times)
    throughput = total_images / mean_t
    s_per_img = mean_t / total_images
    return mean_t, med_t, throughput, s_per_img


def _print_result(result: BenchResult) -> None:
    log = logging.getLogger(f"bench.{result.backend}")
    log.info("=" * 78)
    log.info("BENCH RESULT [%s]", result.backend)
    log.info(
        "  workload          : %d prompts x rollout_n=%d = %d images",
        result.num_prompts,
        result.rollout_n,
        result.total_images,
    )
    log.info(
        "  resolution        : %dx%d  steps=%d  cfg_scale=%.2f  max_seq_len=%d  dtype=%s",
        result.height,
        result.width,
        result.num_inference_steps,
        result.true_cfg_scale,
        result.max_sequence_length,
        result.dtype,
    )
    log.info("  attention backend : %s", result.diffusion_attention_backend)
    log.info("  setup_time        : %.2fs", result.setup_time_s)
    log.info("  warmup_time       : %.2fs", result.warmup_time_s)
    log.info("  iter_times        : %s", ", ".join(f"{t:.2f}s" for t in result.bench_times_s))
    log.info("  mean iter         : %.2fs  (median %.2fs)", result.mean_iter_time_s, result.median_iter_time_s)
    log.info(
        "  throughput        : %.3f images/s  (%.3f s/image)", result.throughput_images_per_s, result.seconds_per_image
    )
    log.info("  peak GPU memory   : %.0f MB", result.peak_gpu_memory_mb)
    if result.extra:
        log.info("  extra             : %s", result.extra)
    log.info("=" * 78)


# ---------------------------------------------------------------------------
# Attention backend introspection
# ---------------------------------------------------------------------------


def _detect_vllm_omni_attn_backend(override: Optional[str]) -> str:
    """Resolve which diffusion attention backend vllm-omni will use.

    Mirrors the platform-side selection logic in
    ``vllm_omni.platforms.cuda.platform.CudaOmniPlatform.get_diffusion_attn_backend_cls``:
    FLASH_ATTN if compute capability is in [8.0, 10.0) and a FA package is
    importable, otherwise TORCH_SDPA. The optional ``override`` matches the
    behavior of the ``DIFFUSION_ATTENTION_BACKEND`` env var.
    """
    if not torch.cuda.is_available():
        return "N/A (no CUDA)"

    if override is not None:
        return override.upper()

    major, minor = torch.cuda.get_device_capability(0)
    capability = major * 10 + minor
    compute_supported = 80 <= capability < 100

    # Match the package detection in vllm_omni.diffusion.envs.PackagesEnvChecker
    fa_available = False
    for mod in ("fa3_fwd_interface", "flash_attn_interface", "flash_attn"):
        try:
            __import__(mod)
            fa_available = True
            break
        except (ImportError, ModuleNotFoundError):
            continue

    if compute_supported and fa_available:
        return "FLASH_ATTN"
    return "TORCH_SDPA"


def _detect_diffusers_attn_backend() -> str:
    """Resolve the active diffusers attention backend.

    Combines:
    - the registry's currently active backend (``native`` by default), and
    - PyTorch's SDPA preferred kernel when applicable, so the user can tell
      whether ``F.scaled_dot_product_attention`` is dispatching to FlashAttention,
      mem-efficient, math, or cuDNN under the hood.
    """
    try:
        from diffusers.models.attention_dispatch import _AttentionBackendRegistry

        active = _AttentionBackendRegistry._active_backend
        active_str = active.value if hasattr(active, "value") else str(active)
    except Exception as e:  # noqa: BLE001
        return f"<unknown: {e}>"

    sdpa_extra = ""
    if active_str.startswith("native"):
        # PyTorch SDPA dispatches dynamically; report which kernels are enabled.
        try:
            from torch.backends.cuda import (
                flash_sdp_enabled,
                math_sdp_enabled,
                mem_efficient_sdp_enabled,
            )

            enabled = []
            if flash_sdp_enabled():
                enabled.append("flash")
            if mem_efficient_sdp_enabled():
                enabled.append("mem_efficient")
            if math_sdp_enabled():
                enabled.append("math")
            try:
                from torch.backends.cuda import cudnn_sdp_enabled

                if cudnn_sdp_enabled():
                    enabled.append("cudnn")
            except ImportError:
                pass
            sdpa_extra = f" (torch.SDPA dispatch: {','.join(enabled) or 'none'})"
        except Exception:  # noqa: BLE001
            pass
    return active_str + sdpa_extra


# ---------------------------------------------------------------------------
# vllm-omni backend
# ---------------------------------------------------------------------------


def run_vllm_omni(args: argparse.Namespace, prompts: list[str]) -> BenchResult:
    """Run the benchmark against a Ray-launched ``vLLMOmniHttpServer``."""
    import ray
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from verl.utils.tokenizer import normalize_token_ids
    from verl.workers.rollout.replica import RolloutMode

    # Importing this triggers registration of QwenImagePipelineWithLogProb
    # which is what the flowgrpo example uses.
    import verl_omni.pipelines  # noqa: F401
    from verl_omni.workers.rollout.replica import DiffusionOutput
    from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import (
        vLLMOmniHttpServer,
    )

    log = logging.getLogger("bench.vllm_omni")
    log.info("Launching vllm-omni rollout server (this may take a while)…")

    setup_t0 = time.time()

    ray_env: dict[str, str] = {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "INFO",
    }
    if args.attn_backend:
        ray_env["DIFFUSION_ATTENTION_BACKEND"] = args.attn_backend

    ray.init(runtime_env={"env_vars": ray_env}, ignore_reinit_error=True)

    total_images = args.num_prompts * args.rollout_n

    rollout_cfg = OmegaConf.create(
        {
            "_target_": "verl_omni.workers.config.diffusion.DiffusionRolloutConfig",
            "name": "vllm_omni",
            "mode": "async",
            "tensor_model_parallel_size": args.tensor_parallel_size,
            "data_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_num_batched_tokens": 8192,
            "max_num_seqs": max(total_images, 64),
            "max_model_len": args.max_sequence_length + 64,
            "dtype": args.dtype,
            "load_format": "auto",
            "enforce_eager": args.enforce_eager,
            "enable_chunked_prefill": False,
            "enable_prefix_caching": False,
            "enable_sleep_mode": False,
            "free_cache_engine": True,
            "disable_log_stats": True,
            "n": 1,
            "pipeline": {
                "_target_": "verl_omni.workers.config.diffusion.rollout.DiffusionPipelineConfig",
                "height": args.height,
                "width": args.width,
                "num_inference_steps": args.num_inference_steps,
            },
        }
    )
    model_cfg = OmegaConf.create(
        {
            "_target_": "verl_omni.workers.config.diffusion.DiffusionModelConfig",
            "path": args.model_path,
            "tokenizer_path": os.path.join(args.model_path, "tokenizer"),
            "trust_remote_code": True,
            "load_tokenizer": True,
            "algorithm": "flow_grpo",
        }
    )

    actor_env: dict[str, str] = {
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
        "NCCL_CUMEM_ENABLE": "0",
    }
    if args.attn_backend:
        actor_env["DIFFUSION_ATTENTION_BACKEND"] = args.attn_backend

    server_cls = ray.remote(vLLMOmniHttpServer)
    server = server_cls.options(
        runtime_env={"env_vars": actor_env},
        max_concurrency=max(total_images + 8, 32),
    ).remote(
        config=rollout_cfg,
        model_config=model_cfg,
        rollout_mode=RolloutMode.STANDALONE,
        workers=[],
        replica_rank=0,
        node_rank=0,
        gpus_per_node=args.tensor_parallel_size,
        nnodes=1,
        cuda_visible_devices=args.cuda_visible_devices,
    )

    ray.get(server.launch_server.remote())
    setup_time = time.time() - setup_t0
    log.info("vllm-omni server up in %.2fs", setup_time)

    attn_backend = _detect_vllm_omni_attn_backend(args.attn_backend)
    log.info("vllm-omni diffusion attention backend: %s", attn_backend)

    # Tokenize prompts once. Matches ``tests/workers/.../test_vllm_omni_generate.py``.
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.model_path, "tokenizer"), trust_remote_code=True)
    selected_prompts = prompts[: args.num_prompts]
    prompt_ids_list: list[list[int]] = []
    for p in selected_prompts:
        messages = [{"role": "user", "content": p}]
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        prompt_ids_list.append(normalize_token_ids(ids))

    def _common_sampling_params(p_idx: int, s_idx: int, seed_base: int) -> dict[str, Any]:
        return {
            "num_inference_steps": args.num_inference_steps,
            "true_cfg_scale": args.true_cfg_scale,
            "height": args.height,
            "width": args.width,
            "max_sequence_length": args.max_sequence_length,
            "logprobs": False,
            # Match the val_kwargs.algo.noise_level=0.0 setting
            # in run_qwen_image_ocr.sh, so the SDE scheduler
            # behaves like a deterministic flow-match-Euler.
            "noise_level": 0.0,
            "seed": seed_base + p_idx * 1000 + s_idx,
        }

    def _submit_concurrent(seed_base: int) -> list[DiffusionOutput]:
        """``rollout_n`` independent B=1 requests per prompt, fanned out concurrently.

        Mirrors what ``DiffusionAgentLoopWorker`` does today.
        """
        refs = []
        for p_idx, ids in enumerate(prompt_ids_list):
            for s_idx in range(args.rollout_n):
                rid = f"bench_c_{p_idx}_{s_idx}_{uuid4().hex[:6]}"
                refs.append(
                    server.generate.remote(
                        prompt_ids=ids,
                        sampling_params=_common_sampling_params(p_idx, s_idx, seed_base),
                        request_id=rid,
                    )
                )
        outs = ray.get(refs, timeout=3600)
        assert len(outs) == total_images, f"expected {total_images} outputs, got {len(outs)}"
        for o in outs:
            assert isinstance(o, DiffusionOutput)
        return outs

    def _submit_batched(seed_base: int) -> list[DiffusionOutput]:
        """One batched request per prompt with ``num_outputs_per_prompt=rollout_n``.

        Uses :meth:`vLLMOmniHttpServer.generate_batched`, which submits a single
        engine request and lets the diffusion transformer run B=rollout_n in one
        forward pass instead of serializing ``rollout_n`` B=1 forwards.
        """
        refs = []
        for p_idx, ids in enumerate(prompt_ids_list):
            rid = f"bench_b_{p_idx}_{uuid4().hex[:6]}"
            # Seed is shared across the batch; per-sample variation comes from
            # the in-pipeline generator's stream advancement.
            refs.append(
                server.generate_batched.remote(
                    prompt_ids=ids,
                    sampling_params=_common_sampling_params(p_idx, 0, seed_base),
                    request_id=rid,
                    num_outputs_per_prompt=args.rollout_n,
                )
            )
        outs_per_prompt = ray.get(refs, timeout=3600)
        flat: list[DiffusionOutput] = []
        for batch in outs_per_prompt:
            assert isinstance(batch, list), f"expected list[DiffusionOutput], got {type(batch)}"
            assert len(batch) == args.rollout_n, f"expected {args.rollout_n} outputs per prompt, got {len(batch)}"
            for o in batch:
                assert isinstance(o, DiffusionOutput)
                flat.append(o)
        assert len(flat) == total_images, f"expected {total_images} outputs, got {len(flat)}"
        return flat

    if args.vllm_omni_mode == "batched":
        _submit_batch = _submit_batched
    else:
        _submit_batch = _submit_concurrent

    log.info("Warmup iteration (%d images)…", total_images)
    t0 = time.time()
    _submit_batch(seed_base=0)
    warmup_time = time.time() - t0
    log.info("Warmup done in %.2fs", warmup_time)

    times: list[float] = []
    for i in range(args.iters):
        t0 = time.time()
        _submit_batch(seed_base=(i + 1) * 100_000)
        dt = time.time() - t0
        times.append(dt)
        log.info(
            "Iter %d/%d: %.2fs (%.3f img/s)",
            i + 1,
            args.iters,
            dt,
            total_images / dt,
        )

    mean_t, med_t, thr, spi = _summarize(times, total_images)
    result = BenchResult(
        backend="vllm_omni",
        num_prompts=args.num_prompts,
        rollout_n=args.rollout_n,
        total_images=total_images,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        true_cfg_scale=args.true_cfg_scale,
        max_sequence_length=args.max_sequence_length,
        dtype=args.dtype,
        diffusion_attention_backend=attn_backend,
        setup_time_s=setup_time,
        warmup_time_s=warmup_time,
        bench_times_s=times,
        mean_iter_time_s=mean_t,
        median_iter_time_s=med_t,
        throughput_images_per_s=thr,
        seconds_per_image=spi,
        # Driver-process memory is irrelevant; engine runs in Ray actor.
        peak_gpu_memory_mb=0.0,
        extra={
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": args.enforce_eager,
            "algorithm": "flow_grpo",
            "scheduler": "FlowMatchSDEDiscreteScheduler (noise_level=0)",
            "submission_mode": args.vllm_omni_mode,
            "submission_shape": (
                f"{args.num_prompts}x B=1 concurrent"
                if args.vllm_omni_mode == "concurrent"
                else f"{args.num_prompts}x B={args.rollout_n} batched"
            ),
        },
    )

    # Best-effort teardown; vllm-omni servers can be slow to shut down.
    try:
        ray.kill(server, no_restart=True)
    except Exception as e:  # noqa: BLE001
        log.warning("ray.kill failed: %s", e)
    ray.shutdown()
    return result


# ---------------------------------------------------------------------------
# diffusers backend
# ---------------------------------------------------------------------------


def run_diffusers(args: argparse.Namespace, prompts: list[str]) -> BenchResult:
    """Run the benchmark against a stock ``diffusers.QwenImagePipeline``."""
    from diffusers import QwenImagePipeline

    log = logging.getLogger("bench.diffusers")
    log.info("Loading diffusers QwenImagePipeline…")
    setup_t0 = time.time()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if args.dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype for diffusers: {args.dtype}")
    torch_dtype = dtype_map[args.dtype]

    pipe = QwenImagePipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype)
    pipe = pipe.to("cuda")

    if args.diffusers_attn_backend:
        from diffusers.models.attention_dispatch import _AttentionBackendRegistry

        _AttentionBackendRegistry.set_active_backend(args.diffusers_attn_backend)
        log.info("Forced diffusers attention backend → %s", args.diffusers_attn_backend)

    if args.compile_transformer:
        log.info("torch.compile(transformer, mode=%s)", args.compile_mode)
        pipe.transformer = torch.compile(pipe.transformer, mode=args.compile_mode)

    setup_time = time.time() - setup_t0
    log.info("diffusers pipeline ready in %.2fs", setup_time)

    attn_backend = _detect_diffusers_attn_backend()
    log.info("diffusers attention backend: %s", attn_backend)

    selected_prompts = prompts[: args.num_prompts]
    total_images = args.num_prompts * args.rollout_n

    # Chunk so we never OOM on huge batches. Default chunk size equals the
    # full workload (== one giant batched call, the most throughput-friendly
    # config) but can be reduced via --diffusers-mini-batch.
    chunk_prompts = max(1, args.diffusers_mini_batch // args.rollout_n)
    log.info(
        "diffusers chunking: %d prompts × num_images_per_prompt=%d per call (=> %d images/call)",
        chunk_prompts,
        args.rollout_n,
        chunk_prompts * args.rollout_n,
    )

    def _run_one_batch(seed_base: int, _pipe=pipe) -> None:
        # ``pipe`` is bound as a default arg so ruff doesn't flag F821 due to
        # the explicit ``del pipe`` later in this function.
        torch.cuda.synchronize()
        for i in range(0, len(selected_prompts), chunk_prompts):
            batch = selected_prompts[i : i + chunk_prompts]
            generator = torch.Generator(device="cuda").manual_seed(seed_base + i)
            _ = _pipe(
                prompt=batch,
                num_inference_steps=args.num_inference_steps,
                true_cfg_scale=args.true_cfg_scale,
                height=args.height,
                width=args.width,
                num_images_per_prompt=args.rollout_n,
                max_sequence_length=args.max_sequence_length,
                generator=generator,
                output_type="pt",  # avoid PIL postprocessing in the hot loop
            )
        torch.cuda.synchronize()

    log.info("Warmup iteration (%d images)…", total_images)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    _run_one_batch(seed_base=0)
    warmup_time = time.time() - t0
    log.info("Warmup done in %.2fs", warmup_time)

    times: list[float] = []
    for i in range(args.iters):
        t0 = time.time()
        _run_one_batch(seed_base=(i + 1) * 100_000)
        dt = time.time() - t0
        times.append(dt)
        log.info(
            "Iter %d/%d: %.2fs (%.3f img/s)",
            i + 1,
            args.iters,
            dt,
            total_images / dt,
        )

    peak_mem = _peak_gpu_memory_mb()
    mean_t, med_t, thr, spi = _summarize(times, total_images)

    result = BenchResult(
        backend="diffusers",
        num_prompts=args.num_prompts,
        rollout_n=args.rollout_n,
        total_images=total_images,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        true_cfg_scale=args.true_cfg_scale,
        max_sequence_length=args.max_sequence_length,
        dtype=args.dtype,
        diffusion_attention_backend=attn_backend,
        setup_time_s=setup_time,
        warmup_time_s=warmup_time,
        bench_times_s=times,
        mean_iter_time_s=mean_t,
        median_iter_time_s=med_t,
        throughput_images_per_s=thr,
        seconds_per_image=spi,
        peak_gpu_memory_mb=peak_mem,
        extra={
            "compile_transformer": args.compile_transformer,
            "compile_mode": args.compile_mode if args.compile_transformer else None,
            "diffusers_mini_batch": args.diffusers_mini_batch,
            "scheduler": pipe.scheduler.__class__.__name__,
        },
    )

    del pipe
    torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--backend",
        choices=["vllm_omni", "diffusers", "both"],
        default="both",
        help="Which engine(s) to benchmark.",
    )

    # Model / data
    p.add_argument(
        "--model-path",
        default=os.environ.get("QWEN_IMAGE_PATH", "Qwen/Qwen-Image"),
        help="Local path or HF hub id of Qwen-Image. "
        "Override with $QWEN_IMAGE_PATH or this flag (default: %(default)s).",
    )
    p.add_argument(
        "--prompts-file",
        default=None,
        help="Optional newline-separated prompts file; falls back to a built-in list.",
    )

    # Workload (mirrors run_qwen_image_ocr.sh)
    p.add_argument(
        "--num-prompts", type=int, default=2, help="Distinct prompts per iteration (= train_batch_size in flowgrpo)."
    )
    p.add_argument("--rollout-n", type=int, default=16, help="Generations per prompt (=rollout.n in flowgrpo).")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="=val_kwargs.pipeline.num_inference_steps in run_qwen_image_ocr.sh",
    )
    p.add_argument(
        "--true-cfg-scale", type=float, default=1.0, help="=rollout.pipeline.true_cfg_scale (1.0 disables CFG)."
    )
    p.add_argument("--max-sequence-length", type=int, default=256, help="=rollout.pipeline.max_sequence_length")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])

    # Iteration control
    p.add_argument("--iters", type=int, default=2, help="Number of measured iterations after the warmup pass.")

    # vllm-omni knobs
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    p.add_argument("--enforce-eager", action="store_true", help="Disable CUDA graphs/compile for fast startup (debug).")
    p.add_argument("--cuda-visible-devices", default="0", help="CUDA_VISIBLE_DEVICES forwarded to the Ray actor.")
    p.add_argument(
        "--attn-backend",
        default=None,
        choices=[None, "FLASH_ATTN", "TORCH_SDPA", "SAGE_ATTN"],
        help="Override vllm-omni DIFFUSION_ATTENTION_BACKEND (default: platform default).",
    )
    p.add_argument(
        "--vllm-omni-mode",
        choices=["concurrent", "batched"],
        default="concurrent",
        help="How to submit rollout-n samples per prompt to vllm-omni: "
        "'concurrent' (rollout_n independent B=1 requests, like DiffusionAgentLoopWorker), "
        "or 'batched' (one request per prompt with num_outputs_per_prompt=rollout_n, "
        "which runs a single B=rollout_n transformer forward).",
    )

    # diffusers knobs
    p.add_argument(
        "--diffusers-attn-backend",
        default=None,
        help="Force diffusers attention backend (e.g. 'native', 'flash', '_native_cudnn'). "
        "Default: leave as registry default ('native'/SDPA).",
    )
    p.add_argument(
        "--diffusers-mini-batch",
        type=int,
        default=None,
        help="Max images per diffusers pipe() call. Default: num_prompts * rollout_n (one big batched call).",
    )
    p.add_argument("--compile-transformer", action="store_true", help="torch.compile the diffusers transformer.")
    p.add_argument("--compile-mode", default="reduce-overhead", choices=["default", "reduce-overhead", "max-autotune"])

    # Output
    p.add_argument("--output", default=None, help="Optional JSON file path to dump structured results.")

    args = p.parse_args(argv)

    if args.diffusers_mini_batch is None:
        args.diffusers_mini_batch = args.num_prompts * args.rollout_n
    return args


def _load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompts_file:
        p = Path(args.prompts_file).expanduser()
        prompts = [line.strip() for line in p.read_text().splitlines() if line.strip()]
        if not prompts:
            raise ValueError(f"No prompts found in {p}")
    else:
        prompts = list(DEFAULT_PROMPTS)
    if args.num_prompts > len(prompts):
        # Repeat the list to satisfy num_prompts (each repeat still distinct in
        # the workload sense from the engine's POV).
        prompts = (prompts * ((args.num_prompts // len(prompts)) + 1))[: args.num_prompts]
    return prompts


def main(argv: Optional[list[str]] = None) -> int:
    _setup_logging()
    args = _parse_args(argv)

    log = logging.getLogger("bench")
    log.info("Benchmark settings: %s", vars(args))

    if args.attn_backend:
        # Mirror this in the driver process too so any subsequent introspection
        # via vllm_omni APIs picks it up.
        os.environ["DIFFUSION_ATTENTION_BACKEND"] = args.attn_backend

    prompts = _load_prompts(args)
    log.info("Loaded %d candidate prompts (using first %d)", len(prompts), args.num_prompts)

    results: list[BenchResult] = []

    if args.backend in ("vllm_omni", "both"):
        try:
            results.append(run_vllm_omni(args, prompts))
        except Exception:
            logging.exception("vllm-omni benchmark failed")
            if args.backend == "vllm_omni":
                return 1

    if args.backend in ("diffusers", "both"):
        try:
            results.append(run_diffusers(args, prompts))
        except Exception:
            logging.exception("diffusers benchmark failed")
            if args.backend == "diffusers":
                return 1

    for r in results:
        _print_result(r)

    # Side-by-side speedup line when we ran both.
    if len(results) == 2:
        by_backend = {r.backend: r for r in results}
        if "vllm_omni" in by_backend and "diffusers" in by_backend:
            vo, df = by_backend["vllm_omni"], by_backend["diffusers"]
            log.info(
                "[SUMMARY] vllm-omni %.3f img/s vs diffusers %.3f img/s (vllm-omni is %.2fx %s)",
                vo.throughput_images_per_s,
                df.throughput_images_per_s,
                max(vo.throughput_images_per_s, df.throughput_images_per_s)
                / max(min(vo.throughput_images_per_s, df.throughput_images_per_s), 1e-9),
                "faster" if vo.throughput_images_per_s >= df.throughput_images_per_s else "slower",
            )

    if args.output:
        Path(args.output).expanduser().write_text(json.dumps([asdict(r) for r in results], indent=2))
        log.info("Wrote results to %s", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
