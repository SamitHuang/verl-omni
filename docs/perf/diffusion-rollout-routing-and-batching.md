# Diffusion rollout: request routing and batching

Last updated: 06/14/2026

How FlowGRPO (and other diffusion RL jobs) route rollout HTTP requests across omni
server replicas (verl-omni) and fuse them into GPU forwards (vllm-omni).

Two layers — do not confuse them:

| Layer | Config | Decides |
|-------|--------|---------|
| **Replica routing** | `actor_rollout_ref.rollout.server_routing` | Which of N `vLLMOmniHttpServer` actors handles a request |
| **Per-server batching** | `engine_kwargs.vllm_omni.*`, `max_num_seqs` | How many requests already on **one** server fuse into one forward |

There is **no cross-replica fusion**. Server 0 cannot batch a request that landed on server 1.

---

## 1. How 512 requests are formed

Each training step (`run_qwen_image_ocr_lora.sh` defaults):

```text
train_batch_size = 32 prompts
rollout.n        = 16 stochastic copies per prompt
→ 512 rollout rows per step
```

In `RayDiffusionTrainer`:

1. Load 32 OCR samples from the dataloader.
2. Assign one `uid` per prompt: `batch.non_tensor_batch["uid"] = uuid4()` × 32.
3. Expand: `gen_batch.repeat(n=16, interleave=True)` → row order  
   `uid_A×16, uid_B×16, …, uid_32×16`.
4. Dispatch: `AgentLoopManager.generate_sequences(gen_batch_output)`.

`AgentLoopManager` splits 512 rows across 4 `DiffusionAgentLoopWorker` actors
(`agent.num_workers=4`): **4 × 128 rows**. Each worker runs
`asyncio.gather` over its 128 `DiffusionSingleTurnAgentLoop.run()` tasks.

---

## 2. Replica routing (verl-omni)

### Config

```yaml
# verl_omni/trainer/config/rollout/server_routing.yaml
actor_rollout_ref.rollout.server_routing:
  policy: prompt_uid_affinity
  routing_key_field: uid
```

Enable at launch:

```bash
actor_rollout_ref.rollout.server_routing.policy=prompt_uid_affinity
```

### Components

```text
OmniLLMServerManager
  └─ 4× vLLMOmniHttpServer (hybrid, 1 per GPU)
  └─ ConfigurableRequestLoadBalancer (global Ray actor)

OmniLLMServerClient (one per agent worker)
  └─ acquire_server(request_id, routing_key)
  └─ server.generate.remote(...)
  └─ release_server(server_id)
```

Code: `verl_omni/workers/rollout/omni_llm_server.py`,
`verl_omni/workers/rollout/request_routing.py`,
`verl_omni/agent_loop/single_turn_agent_loop.py`.

### HTTP call site

```python
# single_turn_agent_loop.py
routing_key = str(kwargs["uid"])   # same for all 16 copies of one prompt

output = await self.server_manager.generate(
    request_id=uuid4().hex,          # new every call — NOT used for uid affinity
    routing_key=routing_key,         # passed to load balancer
    prompt_ids=...,
    sampling_params={..., "seed": per_copy_seed},
)
```

Inside `OmniLLMServerClient`, the engine receives yet another fresh `request_id`
for its internal request state. **Replica choice uses `routing_key` only.**

---

## 3. `prompt_uid_affinity` — concrete behavior

Policy implementation: `ConfigurableRequestLoadBalancer._acquire_sticky(sticky_key)`
where `sticky_key = routing_key` (= batch `uid`).

### State the load balancer keeps

```text
_request_id_to_server: LRUCache   # sticky_key (uid) → server_id
_inflight_requests:    dict        # server_id → count of in-flight HTTP generates
_servers:              dict        # server_id → Ray actor handle
```

`inflight` increments on `acquire_server`, decrements on `release_server` after the
full diffusion generate returns (seconds per request).

### Concrete example (4 replicas, 32 prompts, rollout.n=16)

Assume replicas `R0`–`R3`, prompts `P0`–`P31` with uids `u0`–`u31`.  
Each `ui` will be requested **16 times** (one per rollout copy).

**First time a uid is seen** (cache miss):

1. LB looks up `ui` in `_request_id_to_server` → miss.
2. Pick replica with minimum `_inflight_requests` (tie → lexicographically smallest
   `server_id`, e.g. `R0`).
3. Store `_request_id_to_server[ui] = Rk`.
4. Increment `_inflight_requests[Rk]`, return `Rk`.

**Subsequent copies of the same prompt** (cache hit):

1. LB looks up `ui` → `Rk` (even if `Rk` now has high inflight).
2. Increment `_inflight_requests[Rk]`, return `Rk`.
3. **All 16 HTTP generates for `ui` go to the same replica**, regardless of which
   agent worker issued them or what `request_id` uuid was passed.

Example mapping after the first wave of first-arriving uids (all inflight start at 0):

| uid | First copy routed to | Copies 2–16 |
|-----|----------------------|-------------|
| `u0` | `R0` (inflight was 0) | `R0` (sticky) |
| `u1` | `R1` (R0 inflight=1) | `R1` (sticky) |
| `u2` | `R2` | `R2` |
| `u3` | `R3` | `R3` |
| `u4` | `R0` (all inflight=1, min is R0) | `R0` |
| … | … | same replica as first copy |

Over the full step: **32 uids / 4 replicas ≈ 8 uids per replica**, each uid contributing
**16 concurrent-compatible requests** on that replica's waiting queue.

### Contrast with `least_inflight` (old default)

| | `least_inflight` | `prompt_uid_affinity` |
|--|------------------|------------------------|
| Sticky key | `request_id` (= new uuid4 per call) | `routing_key` (= batch `uid`) |
| 16 copies of `u0` | Spread across replicas as inflight shifts | **Always same replica** |
| Typical per-server queue at schedule | 1–2 requests | 8–16+ from uid clustering |

### Other policies (reference)

| Policy | Sticky / shard key | Use case |
|--------|-------------------|----------|
| `least_inflight` | `request_id` | LLM serving fairness |
| `prompt_uid_affinity` | `routing_key` (`uid`) | Diffusion RL with `rollout.n > 1` |
| `prompt_hash_sharding` | `hash(routing_key) % N` | Stateless sharding |
| `round_robin` | none | Even rotation, ignores load |

---

## 4. Per-server request batching (vllm-omni)

After routing, each `vLLMOmniHttpServer` runs an independent `DiffusionEngine`.

### Engine settings (current defaults)

Set in `vllm_omni_async_server._preprocess_engine_kwargs`:

| Knob | Default | Meaning |
|------|---------|---------|
| `step_execution` | `false` | Request-mode `execute_batch` path |
| `request_batch_max_wait_ms` | `250` | Max wait before `schedule()` (cap; typical ~10–50ms on burst) |
| `max_num_seqs` | config (32 in e2e run) | Max fused requests per scheduler step |

### Batching pipeline (one replica)

```text
HTTP generate
  → OmniDiffusionRequest enqueued on local waiting queue
  → admission window (optional wait for more arrivals)
  → scheduler: group by SamplingParamsKey
       (height, width, true_cfg_scale, LoRA id, …; seed NOT in key)
  → execute_model_batch (width ≤ max_num_seqs)
  → QwenImagePipelineWithLogProb.forward()
       logs: [flowgrpo_reqbatch] pipeline.forward num_reqs=N
```

The 16 copies of one OCR prompt are **batch-compatible** (same resolution/CFG, different
`seed`). Uid affinity puts them on one queue; admission + scheduler fuse them.

### Multi-wave draining

One replica may serve ~128 requests per step in several waves:

```text
Wave 1: schedule 32 → forward (num_reqs=32) while GPU busy
Wave 2: schedule 32 → …
…
Tail:   num_reqs=1..16 for stragglers
```

---

## 5. End-to-end diagram

```text
512 rows (32 uid × 16 copies)
        │
        ▼
AgentLoopManager.chunk(4)  →  4 × 128 async tasks
        │
        ▼
DiffusionSingleTurnAgentLoop × 512
  routing_key = uid
  request_id  = uuid4()  (ignored for affinity)
        │
        ▼
ConfigurableRequestLoadBalancer  (prompt_uid_affinity)
  uid → R0 | R1 | R2 | R3
        │       │       │       │
        ▼       ▼       ▼       ▼
   local scheduler + admission + execute_batch
        │
        ▼
   GPU forward  num_reqs ≤ 32
```

---

## 6. Observability

```bash
export FLOWGRPO_BATCH_VERIFY_LOG=/path/to/verify.log
```

Look for:

```text
[flowgrpo_reqbatch] pipeline.forward num_reqs=32
[RequestBatch] admission wait done waiting=… scheduled_new_reqs=32
```

Histogram:

```bash
grep -oP 'num_reqs=\K\d+' "$FLOWGRPO_BATCH_VERIFY_LOG" | sort -n | uniq -c | sort -rn
```

Tests:

- `tests/workers/rollout/test_rollout_server_routing.py` — LB policy semantics
- `tests/workers/rollout/rollout_vllm/test_vllm_omni_request_batch_flowgrpo.py` — single-server batching
- `tests/workers/rollout/rollout_vllm/test_rollout_server_routing_perf.py` — 4-server comparison

---

## 7. E2e verification result (uid-affinity run)

Run: `uid_affinity_flowgrpo_150_20260613_120025`  
W&B: https://wandb.ai/samithuang/flow_grpo/runs/m6isr2pu

- Rollout waves reached **`num_reqs=32`** on all 4 omni servers.
- **~343 s/step** (vs ~480 s baseline with thin per-server queues).
- 150/150 steps completed.

---

## Related files

| File | Role |
|------|------|
| `verl_omni/trainer/diffusion/ray_diffusion_trainer.py` | `uid` assignment, `repeat(n)` |
| `verl_omni/agent_loop/single_turn_agent_loop.py` | `routing_key=uid` |
| `verl_omni/workers/rollout/request_routing.py` | `ConfigurableRequestLoadBalancer` |
| `verl_omni/workers/rollout/omni_llm_server.py` | `OmniLLMServerManager` / `Client` |
| `verl_omni/trainer/config/rollout/server_routing.yaml` | Routing defaults |
| `verl_omni/workers/rollout/vllm_rollout/vllm_omni_async_server.py` | Engine kwargs |
| `verl_omni/pipelines/qwen_image_flow_grpo/vllm_omni_rollout_adapter.py` | `supports_request_batch`, logs |

See also: [request-batch-multi-server-routing.md](./request-batch-multi-server-routing.md) for the original ingress analysis.
