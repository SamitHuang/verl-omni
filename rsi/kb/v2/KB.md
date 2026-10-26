# Knowledge base v2 (T0 and T2 by hand and by agents; bench, bench_loop and convergence runs)

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

## C. Throughput under 0-diff

**C1. Know where the step goes before optimizing a kernel.**
On MiniCPM-o LoRA (2 GPUs, eager rollout, 32 prompts x 8, about 210 response tokens
mean, 1000 max), stock spends 54% of the RL step in generation, 7% in log-prob
recompute, 16% in the actor update and 22% in weight sync. A 0-diff candidate that is
33% slower at decode costs about +20% step time there; on a workload with short
responses and CUDA graphs (generation about 20% of the step) the same decode loss
costs much less. Check: the brief's loop phase table; compare phases, not just the step.
Evidence: `bench_loop` stock 32.5 s (gen 17.6), T2 KB candidate 39.2 s (gen 26.8).
Scope: MiniCPM-o 4.5, vLLM 0.30, H200.

**C2. Rollout batch invariance is the decode cost.**
`VLLM_BATCH_INVARIANT=1` routes every linear layer (and lm_head) through a persistent
Triton matmul instead of cuBLAS, plus Triton RMSNorm and log-softmax and FA3 with
`num_splits=1`. On unmodified stock code it alone decodes at 67% / 70% of stock eager
(batch 32 / 128) and 53% / 87% with CUDA graphs; the T2 KB candidate built on it is
only 3-4 points lower (63% / 66%, 51% / 83%). The actor-side parity code costs the
rollout almost nothing; the batch-invariant kernels are where the time goes.
Faster batch-invariant GEMMs (tile configs for decode shapes, split-K off) keep the
result bitwise as long as the reduction order per output element does not depend on
the batch size. Check: bench with your change vs `VLLM_BATCH_INVARIANT=1` alone.
Evidence: evaluator bench, all four measured in one session.

**C3. Forcing every custom op on under Inductor costs nothing measurable.**
Removing `VLLM_OMNI_BATCH_INVARIANT_KERNELS` (which sets `custom_ops` to `all` and
makes the RMSNorm opaque) from the T2 KB candidate leaves CUDA-graph decode unchanged
(51% / 83%) and eager 3 points lower, and it breaks the CUDA-graph 0-diff (C4). An
earlier cross-session comparison suggested a large cost; that was session drift (C5).

**C4. Inductor regenerates user Triton kernels with an fp64 `eps`.**
A Triton RMSNorm launched directly takes `eps` as fp32; inside an Inductor graph the
same kernel is regenerated with `eps` typed fp64, which changes `sqrt(mean(x^2)+eps)`
in the last bit for some rows. Wrapping the kernel as an opaque `torch.library`
custom op keeps one launch path. Check: compare the kernel's output inside and
outside `torch.compile` with `!=`.
Evidence: the T2 KB candidate (CUDA-graph acceptance passes with it).

**C5. Measurement noise: compare back to back.** Within one session the 3 bench
repeats agree to 0.1% (actor) and 1-4% (rollout). Stock CUDA-graph decode at batch 128
measured 9249, 7808 and 9299 tok/s in three sessions (7808 with four benches sharing
the node) while the KB candidate stayed at about 6450, so its ratio read anywhere from
69% to 83%. The evaluator benches stock right after every candidate on the same GPUs;
do the same in your own measurements.

**C6. Weight sync was faster in both 0-diff candidates** (3.6-3.7 s vs 7.1 s stock per
step on MiniCPM-o). Not explained; do not assume a change to sync is free or neutral,
measure it.

**C7. With a 0-diff stack the old-log-prob recompute can be skipped.**
`algorithm.rollout_correction.bypass_mode=true` takes old log-probs from the rollout
and saves the actor's `compute_log_prob` (2.2 s of a 32 s step on MiniCPM-o). With a
0-diff candidate the 150-step GSM8K run matched stock with recompute (validation
0.941 both); with stock's mismatch it clipped 2-3x more and lagged early.
Check that the update is exactly on-policy: with bypass and one mini-batch per step
(`ppo_mini_batch_size` = train batch), `actor/ppo_kl` and `actor/pg_clipfrac` must be
exactly 0 (they are for the T2 KB candidate at full depth). With 2 mini-batches the
second runs after an optimizer step, so `ppo_kl` is nonzero by design.
Evidence: evaluator convergence runs (150 steps, n = 1 per arm) and on-policy check.

## D. From a human 0-diff design for Qwen3-Omni (vLLM 0.32, H20)

**D1. Do not run vLLM's batch-invariance init in the actor.**
`override_envs_for_invariance()` (called by `init_batch_invariance()`) unconditionally
sets `NCCL_MAX_NCHANNELS=1`, `NCCL_NTHREADS=1`, `NCCL_PROTO=Simple`,
`NCCL_ALGO=allreduce:tree` and disables NVLS. In the rollout (TP=1) that is harmless;
in an FSDP actor it slows every all-gather and reduce-scatter. Call the batch-invariant
kernel functions directly in the actor instead of the init. Check: print those
variables inside an actor process; compare bench actor fwd+bwd with and without.
Scope: vLLM 0.30 code reading; reported large actor speedup on Qwen3-Omni.

**D2. MoE: use vLLM's fused MoE kernels on both sides** (Qwen3-Omni only; MiniCPM-o is
dense). Moving rollout and actor to the same fused MoE kernel took rollout throughput
from -40% to -10% vs non-bitwise in that design; the remaining cost was about +10% RL
step time with CUDA graphs on vLLM 0.32, H20.
