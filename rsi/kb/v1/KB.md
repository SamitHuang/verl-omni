# Knowledge base v1 (from T0: Qwen3-Omni thinker, text, bf16, TP=1, full-parameter)

Each entry: the lesson, how to check it, the evidence it rests on, and the scope it
was verified on. Entries are claims to re-check, not facts: verify on your stack.

## A. General: train/rollout 0-diff

**A1. Make the rollout's kernels the reference; the actor calls them.**
Patch the actor's forward to call the exact functions the rollout runs (same kernel,
same shapes, same op order), wrapped in `torch.autograd.Function` whose backward is
ordinary torch math. Do not try to make the rollout reproduce HF numerics.
Check: g1 first divergence moves past every op you rebind.
Evidence: T0 solved this way (g2 0/16384, g1 0/65536, full 48 layers 0/2899).
Scope: vLLM 0.30, FSDP2 actor, H200.

**A2. Turn on the rollout's batch invariance first, and check it is really invariant.**
vLLM has `VLLM_BATCH_INVARIANT=1` (set it in `CANDIDATE_ENV`; it reaches both the
rollout and the actor processes). It switches linear/lm_head to a persistent Triton
matmul, RMSNorm without residual to a Triton kernel, FA3 to `num_splits=1`, MoE to a
fixed Triton config, and `log_softmax` to a Triton kernel. Before building on it,
check that the rollout alone gives the same result for a request alone vs in a batch
and in prefill vs decode (g1 checks `alone_vs_batch_*`, `prefill_vs_decode`).
Check: `rsi_tools/kernel_invariance.py` prints 0 everywhere.
Scope: vLLM 0.30, FA3 build (vLLM's FA2 build does not load on this box).

**A3. "Same function" is not "same kernel": find what the rollout actually runs.**
Under `enforce_eager`, vLLM custom ops are enabled, so e.g. rotary runs
`forward_cuda` (a Triton kernel), not the native torch path; RMSNorm with a residual
runs the CUDA `fused_add_rms_norm`, without one the Triton batch-invariant kernel.
Model packages may override vLLM classes (vLLM-Omni has its own M-RoPE class).
Check: read the class actually instantiated in the rollout worker, not the generic one.

**A4. vLLM CustomOps need a vLLM config context when built outside the engine.**
Constructing one in the actor (e.g. `get_rope`) asserts "Current vLLM config is not
set". Wrap construction in `with set_current_vllm_config(VllmConfig()):`, with the
target dtype as default dtype and the target device.

**A5. A 0-diff forward says nothing about gradients.**
A detached op or wrong backward keeps the forward identical; only the training goes
wrong. Gate g3 now checks gradients against an fp32 reference with stock bf16 as the
bar. Check each custom backward in isolation first: its error vs fp32 should match
plain bf16 autograd (`rsi_tools/backward_check.py`); rotations: `R^T R q == q`.
Evidence: a candidate with the MoE output detached had the same loss to 6 digits and
failed g3 (min cosine 0.15, norm ratio 0.64); the T0 solution passed (0.984 vs stock 0.984).

**A6. Sanity-check the grad norm against the baseline run.**
Baseline T0 grad norms are about 1.8 to 2.0 on both steps. A run that shows 108 or 15
is suspect even if every gate passes; find out why before moving on.

**A7. Backward speed: no Python loops over experts.**
A per-expert loop with `unique().tolist()` made actor updates 5x slower. Recompute
the expert MLP with `torch._grouped_mm` over expert-sorted tokens (offsets from
`bincount().cumsum()`): 17x to 110x faster, weight gradients bitwise equal to the loop.

**A8. Sampling is not reproducible run to run.**
Even with seed 42, rollout responses differ across runs (async scheduling), so g2
mismatch counts move by a token or two between identical candidates and grad norms
vary with the samples. Compare runs on their gates, not on raw sampled values.

**A9. The evaluator's static check reads names.**
Integrity patterns match words like `top_k`, `temperature` followed by `=` or `:` on
changed lines. A MoE "top-k" parameter trips it; name it `num_experts`.

**A10. Full-depth acceptance needs memory planning.**
The 48-layer thinker (30B MoE) with FSDP2 and optimizer offload OOMs on 2 H200 in the
optimizer step (Adam state for a 15B shard is about 120 GB); 4 GPUs work. Weight
loading takes about 25 minutes.

## B. Qwen3-Omni thinker specifics (vLLM-Omni 12e9280, transformers 5.13)

**B1. Norm paths.** Layer 0 input norm: no residual (Triton kernel). Every other
decoder norm and the final norm: fused residual add + norm (CUDA). q/k norms: Triton.
The fused kernel normalizes the bf16 sum, so the actor can reproduce it with
`fused_add_rms_norm(h.clone(), zeros_like(h), w, eps)` on the HF residual-added state.

**B2. Rotary.** Interleaved M-RoPE, sections [24, 20, 20], theta 1e6, head 128. Use the
rollout's bf16 cos/sin cache built on CUDA and `triton_mrope`; the backward is the same
kernel with `-sin`. HF passes text positions (row 0 of the [4, B, T] ids) to the
layers and the 3 M-RoPE rows to rotary.

**B3. Attention.** Fused QKV GEMM on concatenated q/k/v weights, then FA3 varlen causal
with `num_splits=1`; cu_seqlens from the text position ids. Backward: the flash-attn
varlen backward.

**B4. MoE.** All layers sparse (128 experts, top 8, renormalized). Router logits from
the batch-invariant matmul, `fused_topk` (softmax), `fused_experts` with the HF
`gate_up_proj` [E, 2I, H] / `down_proj` [E, H, I] layouts as-is.

**B5. Log-probs.** vLLM computes `log_softmax(logits, float32)` (Triton under batch
invariance). verl's FSDP engine calls `transformer_impl.logprobs_from_logits`; route it
to the same kernel, chunked over rows (chunking does not change per-row results).

**B6. DeepStack (image/video input; not verified yet).** vLLM-Omni adds visual
features to the layer output before the deferred residual add; HF adds after.
Mathematically equal, rounded differently in bf16.

**B7. Cost.** Full model, warm step: rollout generation +12% (batch-invariant
kernels), actor log-prob and update flat, memory flat. CUDA graphs do not start on
this stack even at baseline (buffer assignment during forward).
