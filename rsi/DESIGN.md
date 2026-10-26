# RSI demo: design (draft)

Status: draft for discussion. Nothing here is implemented yet.

## 1. Goal

Show a loop in which a coding agent improves the RL framework against a
verifiable objective, and in which the agent's own tools and knowledge base
improve across tasks, so that later tasks get cheaper. The second property is
what distinguishes this from an autotuner and is measured explicitly (section 9).

First demo, in order:

1. **Calibration (T0).** Reach 0-diff train-rollout log-probs for the
   Qwen3-Omni thinker (4-layer truncation, TP=1, full-parameter, text-only)
   starting from the current pins, without access to the human solution in
   verl-omni#637. Compare cost with the human effort.
2. **First new result (T1).** Starting from a 0-diff stack, maximize rollout
   throughput while keeping exactly 0 mismatches.

Non-goals for the demo: diffusion models, convergence claims, multi-node,
automatic PR submission. Every loop ends with a branch, a report and the
ledger, handed to a human (AGENTS.md section 1).

## 2. Components

```
                  +-------------------------------+
                  | orchestrator (trusted)        |
 task spec  --->  |  budget, GPU locks, worktrees |
                  |  ledger, KB, evaluator calls  |
                  +---------------+---------------+
                                  | one iteration = one worktree
                                  v
                  +-------------------------------+
                  | agent worker (untrusted)      |
                  |  coding-agent CLI, edits code |
                  |  inside the editable surface  |
                  +---------------+---------------+
                                  | candidate commit
                                  v
                  +-------------------------------+
                  | evaluator (trusted, read-only)|
                  |  gates -> result.json         |
                  +-------------------------------+
```

- **Trusted:** orchestrator, evaluator, task spec, ledger. Live outside the
  worker's worktree; the worker cannot modify them.
- **Untrusted:** the agent worker. Any coding-agent CLI (Cursor CLI first,
  Claude Code headless as an alternative). Swappable so agents can be
  compared on the same tasks.

## 3. Task spec

One YAML file per task. The orchestrator refuses to start if a field is
missing.

The T0 spec is `rsi/tasks/t0_qwen3omni_thinker_zero_diff.yaml`. Its main
fields:

```yaml
scope:                         # fixed; g0 rejects changes to it
  checkpoint: /home/public/yx/rsi_ckpts/qwen3omni_thinker_4l   # rsi.tools.make_truncated_ckpt
  truncate_layers: 4
  rollout_tp: 1
  lora: false
  modalities: [text]
  dtype: bfloat16
  sampling: {temperature: 1.0, top_k: -1, top_p: 1.0}
start:
  python: /home/public/yx/verl-omni/.venv/bin/python
  repos:                       # worktree per repo, put first on PYTHONPATH
    verl-omni: {path: /home/public/yx/verl-omni, sha: c53b387c}
    verl: {path: /home/public/yx/verl, sha: fefb0802}
    vllm-omni: {path: /home/public/yx/vllm-omni-pin-12e9280, sha: 12e928001}
editable: [verl-omni/verl_omni/**, verl/verl/**, vllm-omni/vllm_omni/**]
protected: [verl-omni/rsi/**, "*/tests/**"]
objective:
  primary: {metric: mismatch_total, goal: "== 0"}
gates: [g0_static, g2_loop_zero_diff, g1_offline_zero_diff]
budget: {gpu_hours: 40, iterations: 60, wall_clock_h: 72, iteration_timeout_min: 20}
context: {allow_web: false, kb: none}
```

The worktrees override the editable installs in the shared venv through
`PYTHONPATH`. This works because the three repos are pure Python; vLLM's
compiled kernels come from the venv and are not editable.

T1 differs in `start` (post-fix pins), `objective.primary` (`rollout_tok_s`,
maximize), and adds `constraints: [{metric: mismatch_total, goal: "== 0"}]`.

## 4. Evaluator contract

Single entry point, run by the orchestrator on a candidate worktree:

```
python -m rsi.evaluate --task <spec.yaml> --worktree <dir with the 3 repos> \
    --gate <gate_id> --out-dir <dir> --gpus 0,1 [--from-dir <g2 out-dir>] [--acceptance]
```

`<out-dir>/result.json` (abridged from the baseline g1 run):

```json
{
  "task_id": "t0_qwen3omni_thinker_zero_diff",
  "candidate": {"verl-omni": "<HEAD>", "verl": "<HEAD>", "vllm-omni": "<HEAD>", "patch_id": "base"},
  "gate": "g1_offline_zero_diff",
  "status": "pass | fail | invalid | error | timeout",
  "metrics": {"mismatch_total": 24597, "tokens_compared": 65536, "first_divergence": "L0.input_norm"},
  "checks": [
    {"name": "prefill_vs_actor", "mismatch_total": 8192, "tokens_compared": 8192, "abs_diff_max": 3.14, "abs_diff_mean": 0.070},
    {"name": "first_divergence", "row": 0, "op": "L0.input_norm", "ops": [{"op": "embed", "status": "equal"}, "..."]}
  ],
  "env": {"gpu_name": "NVIDIA H200", "compute_cap": "9.0", "gpus": "0,1"},
  "artifacts": ["..."],
  "wall_clock_s": 324
}
```

`invalid` means the measurement itself is untrustworthy (weights not synced,
no real update, or the actor replay does not reproduce the loop), as opposed
to `fail`, which means a valid measurement with mismatches.

Rules:

- Mismatch is computed by the evaluator from raw log-prob tensors dumped by
  both sides: same dtype, all finite, element-wise `!=` on valid response
  tokens. No `exp`, no tolerance. It does not use any metric code the worker
  can edit.
- `first_divergence` comes from forward hooks on both sides
  (`rsi/evaluate/probe.py`) over a fixed op list: `embed`, then per layer
  `input_norm, q_norm, k_norm, attn_core, attn_out, post_norm, router, mlp,
  out`, then `final_norm` and `logits`. The first differing op is the first
  whose inputs were equal. This is the main diagnostic signal the agent gets.
  MSProbe (`.claude/skills/train-infer-consistency`) is the fallback for
  deeper dumps.
- Dump hooks are trusted code loaded in every process via
  `VERL_USE_EXTERNAL_MODULES=verl_omni,rsi.evaluate.hooks` and active only
  when the evaluator sets `RSI_DUMP_DIR`.
- Failures are structured (`status`, `checks`), never "read the log".

## 5. Gates

Run order is g0, g3, g2, g1. g3 is cheap and a failure makes the candidate
`invalid` whatever its forward does, so it runs before the loop. g2 comes
before g1 because g1 replays exactly what g2 captured (the step-1 batch and
the rollout engine arguments), so the offline suite cannot drift from the
real loop's configuration. g1 runs even when g2 fails, because it produces
the diagnostics. The baseline runs every gate even when g3 fails.

| gate | GPUs | wall (baseline) | content |
|---|---|---|---|
| `g0_static` | 0 | seconds | changed files inside `editable` and outside `protected`; added or removed lines must not touch integrity patterns (section 8); `CANDIDATE_ENV` keys allowed by `candidate_env_rules` |
| `g3_grad` | 1 | ~3 min | actor backward on two fixed rows (surrogate loss, advantages +1/-1) vs an fp32 SDPA per-row reference, with stock bf16 as the yardstick: every parameter with a nonzero reference gradient has a nonzero gradient; per-parameter cosine ≥ min(stock − 0.05, 0.99); total norm ratio in [0.8, 1.25]. For LoRA, B = 0 at init, so the expected dL/dB is `scaling · dL/dW · Aᵀ` from the same full reference |
| `g2_loop_zero_diff` | 2 | ~12 min | verl loop, FSDP2 + vLLM-Omni, 2 steps with real optimizer updates and weight syncs; 0/8192 mismatches at both steps; BLAKE2b digest of every tensor the actor sends equals what each rollout replica receives, at every sync; tensors change between syncs; `grad_norm` nonzero |
| `g1_offline_zero_diff` | 2 | ~5.5 min | replays the first step g2 found mismatched (step 1 if none), with that step's weights (below); actor replay of the captured batch reproduces the loop exactly (else `invalid`); vLLM prefill vs actor; loop decode vs actor; prefill vs decode; actor alone vs packed; vLLM row alone vs inside a batch of 4 copies; per-op first divergence on one row; with `--acceptance`, prefill under CUDA graphs |
| `bench` | `bench.actor.gpus` | ~11-16 min | throughput for the `maximize` objective; full-depth checkpoints from the task's `bench` section, from the candidate's g2 capture (`--from-dir`). Rollout: the captured engine with the model swapped, fixture prompts tiled to batch 32 and 128, `ignore_eos` decode of exactly 512 tokens (same token count for every candidate), eager and CUDA graph. Actor: torchrun FSDP2, step-1 batch tiled to 32k tokens per rank; `compute_log_prob` and module forward + `logprobs_from_logits` + backward in ≤8k-token packed chunks, no optimizer step. 1 warmup + 3 repeats, median; `spread` = (max − min) / median wall time per metric |

`bench_loop` runs `g2_loop.sh` with `RSI_HOOKS=0` (no capture hooks, so no
per-sync tensor hashing or state dumps) on the full checkpoint with the
task's `bench.loop` settings (default 32 prompts x 8 responses, 5 steps,
step 1 dropped as warmup) and reports the median step and phase times
(`gen`, `old_log_prob`, `update_actor`, `update_weights`), response lengths
and verl's `training/rollout_probs_diff_max` (0 for a 0-diff candidate).
Response lengths are natural (the vLLM-Omni AR rollout ignores verl's
`ignore_eos`), so compare phase times together with response lengths.
The bench actor now runs under torchrun on `bench.actor.gpus` GPUs (FSDP2),
so NCCL settings count: vLLM's `override_envs_for_invariance()`
unconditionally sets `NCCL_MAX_NCHANNELS=1`, `NCCL_NTHREADS=1`,
`NCCL_PROTO=Simple`, `NCCL_ALGO=allreduce:tree` and more whenever
`init_batch_invariance()` runs in a process (reported by cr-gao as a large
actor slowdown when it runs in the actor).

Named profiles: `--bench-profile P` runs `bench_loop` with `bench.P`
layered over `bench.loop` (YAML aliases to another profile's override list
are flattened), e.g. `converge` / `converge_bypass` on T2 (150 steps, real
GSM8K reward, validation every 25 steps); the summary adds the first/last
10-step mean of `critic/score/mean` and the first/last `val-core` score.
Stock Qwen3-Omni full fine-tuning at depth 48 on 4 GPUs runs out of memory
in `update_actor` with the default 16k-token micro-batch (optimizer
offload on, rollout memory 0.6), and again at 8k tokens (about 136 GB in
use, still in `update_actor`). Gradient checkpointing was on. Full-parameter
Qwen3-Omni loop numbers are therefore not available on 4 GPUs. The
Qwen3-Omni throughput task uses LoRA instead (no full-size fp32 gradients).

Benchmark caveats (v1): the actor ran at world size 1 (no FSDP communication),
the rollout at `gpu_memory_utilization` 0.85 rather than the loop's 0.5,
and the fixture's responses come from the truncated model, so MoE routing
sees text-like but not natural sequences. Candidates differ by the same
inputs only. A repeat of stock MiniCPM-o in a separate session moved by 3%;
within a session the 3 repeats agree to about 1% (up to 10% at batch 128).
`python -m rsi.tools.bench_candidates --spec S.json --out D` runs saved
candidates one GPU each and writes a table relative to stock.

The truncated model's GSM8K reward is always 0, which gives zero advantages
and no update. g2 therefore uses `rsi/evaluate/reward.py`, a deterministic
hash of the response mapped to [0, 1), so step 2 runs after a real update.

Why g1 replays later steps: on T0-LoRA the baseline is 0-diff at step 1
(B = 0, LoRA has no effect) and 8190/8192 at step 2, so a step-1 replay
passes every check and the first-divergence part of the progress key says
nothing. At sync k (default 1, the weights step 2 runs with) g2's hooks
save the actor's gathered trainable parameters and every tensor the actor
sends. g1 loads the first into the rebuilt actor (`actor_replay_vs_loop`
then verifies the restore) and replays the second through the rollout
extension's own `update_weights_from_ipc`, with only the IPC receiver
replaced. The dumps can be a full model, so g1 deletes them after use.

Why g3 exists: 0-diff only constrains the forward pass. Solving T0 by hand
went through intermediate candidates with good log-prob matching but grad
norms of 15 and 108 (baseline: about 2), and a detached MoE output
(`runs/g3_neg`) gives the same loss to 6 digits while its gradients have
cosine 0.15. The fp32 reference and the
stock run are cached per (checkpoint, fixture, gate code) and are
independent of the candidate; the candidate subprocess loads the cache and
writes only per-parameter statistics. Calibration: stock passes, the T0
solution passes (min cosine 0.9842 vs stock 0.9836, norm ratio 1.086), the
negative control fails every check.

## 6. Orchestrator loop

`python -m rsi.orchestrator.loop --task T.yaml --run-dir R --agent {noop|cursor[:model]} --gpus 0,1,2,3`

```
evaluate baseline (start shas + start.patches, all gates) -> ledger line 0
for i in 1..budget.iterations (stop on GPU-hour / wall-clock budget):
    workspace.reset(best.patches)              # detached worktrees at start shas + parent patches (+ CANDIDATE_ENV)
    copy toolset to rsi_tools/ (--toolset)     # KB tools snapshot, optional
    write RSI_BRIEF.md                         # task, scope, environment, last gate results, divergence table, recent ledger, KB (--kb)
    lease agent.gpus; agent.run(workspace)     # as user rsi-agent, timeout, GPU-seconds counted
    audit(agent.log)                           # flags web tools, URLs, git history, hidden paths (section 8)
    patches = workspace.diff()                 # repo diffs + CANDIDATE_ENV, saved under iter_i/patches/, never committed
    g0 -> g3 -> g2 -> g1                       # GPU leases, hard timeouts, process-tree cleanup
    verdict = compare(progress(results), best) # improved | neutral | regressed | invalid | error
    if g2 and g1 pass: re-run from a clean workspace -> solved | unconfirmed
    ledger.append(...); update best on improved/solved
report.md
```

`maximize` tasks (`objective.primary.type: maximize`, `orchestrator/objective.py`)
change three things. At session start the loop runs stock (no patches)
through g2 (engine capture) and `bench_loop` into `stock_<time>/`. Every
candidate that passes the correctness gates then runs `bench` paired with
a stock `bench` (below). The key is
`(every task gate passes, worst ratio to stock over objective.primary.metrics)`,
and a candidate is `improved` only if the ratio beats the parent's by more
than max(`min_gain`, 2 × the worst repeat spread of the candidate, the
parent and their paired stock runs, over the scored metrics only); within
that band it is `neutral`. An improvement, or any candidate whose ratio
reaches `target`, is confirmed by `bench_loop` at full depth, which must
pass with `rollout_probs_diff_max == 0` (else `unconfirmed`); `solved` when
the ratio reaches `target`. The first T1 runs (h7) took the spread over every
bench metric and checked the target only after `improved`. Stock CUDA-graph
decode at batch 32 (not scored) varied by 38% between repeats, so the
required gain became 77%. Three candidates above the 0.9 target were left
`neutral`: no-KB iteration 5 at 0.955, and KB iterations 4 and 5 at 0.93
and 0.944. h8 fixes both, and a run resumed with `--resume-from` first
confirms earlier candidates at the target, best first, as new ledger entries
(`agent: confirm`). The brief shows a per-metric stock / parent / ratio
table, the full-depth phase times of both and the noise threshold.

Paired stock. One stock measurement per session is not enough: in the
rollout ablation (`rsi_work/bench/ablation_out`) stock CUDA-graph decode at
batch 128 was 16% lower than in the v2 session while the KB candidate was
unchanged, moving its ratio from 71% to 83%. So every candidate's `bench` is
followed by a stock `bench` on the same leased GPUs (`bench_stock/`, from a
stock worktree under the run dir), and its ratio and noise threshold use
that pair. The session's stock setup only provides the engine capture and
the full-depth loop phase times; a resumed run needs no re-measurement.

Batch-invariance ablation (one session, % of stock): stock +
`VLLM_BATCH_INVARIANT=1` alone decodes at 67% / 70% eager (batch 32 / 128)
and 53% / 87% with CUDA graphs; the T2 KB candidate at 63% / 66% and
51% / 83%; without its forced custom ops 60% / 63% and 51% / 83%. The
decode cost is vLLM's batch-invariant kernels, not the candidate's
parity code.

A candidate is a patch set plus an optional `CANDIDATE_ENV` file
(`KEY=VALUE` lines) at the workspace root. The T0 solution needs
`VLLM_BATCH_INVARIANT=1`, so opt-in switches have to be part of the
candidate; g0 checks the keys against `candidate_env_rules` (allow and deny
full-match regexes, deny wins) and the evaluator applies them to every gate
subprocess without leaking into the next candidate.

The agent gets `agent.gpus` GPUs for its own experiments during its turn.
Solving T0 by hand relied on many short GPU experiments (per-op kernel
checks, backward checks); an agent that only sees gate results would spend a
1.4 GPU-hour evaluation per hypothesis.

Rejected candidates. The workspace restarts from the parent, so an agent
whose candidate was rejected has neither the reason nor its own code. The
ledger stores the first failing gate's reason (g0's exact flagged line, g3's
failing checks), the brief shows it, and the rejected patches are copied to
`rsi_previous_candidate/`. Found on the T0 control (harness h1, which lacks
this): iteration 4 was rejected by a g0 false positive
(`self.top_k = text_config.num_experts_per_tok`) and the next brief said
only `verdict=invalid`; agents also read their own earlier CLI transcripts,
apparently to recover what the brief left out.

Validated end to end: the hand-made T0 solution replayed through the loop
with the scripted agent (`runs/refrun_t0`) passes g0, g3, g2 (0/16384), g1
and the clean re-confirm, verdict `solved`, 1.37 GPU-hours.

Progress key (larger is better): `(g2 passes, index of the first diverging
op, -g2 mismatch fraction rounded to 1%)`. Fixing one op usually leaves
every token mismatched, because later ops still differ, so the mismatch
count alone would hide real progress; moving the first divergence later is
the intermediate signal. The rounding is needed because rollout sampling is
not reproducible across runs even with `seed=42` (async scheduling changes
the sampled tokens and row order), so the raw count moves by a few tokens
between identical candidates.

Harness snapshots. Gate subprocesses import the harness from disk on every
evaluation, so editing `rsi/` during a run would change how later
candidates in that run are judged. Each run is launched from a frozen copy
(`rsi_work/harness_snapshots/hN`, reached through `rsi_work/harness/rsi`);
development uses `rsi_work/harness_dev`. `run_config.json` records the
snapshot path and a hash of its code.

Pre-flight. Before the first turn the loop checks, as the agent user, that
the checkpoint, the Python environment and `agent.readable` paths are
readable. The first T0 control turn found the checkpoint's weight files
root-only and fell back to the full upstream snapshot.

Engineering: candidates are patch sets against the start commits, so
nothing is committed to the source repos; GPUs are leased with `flock` lock
files plus an idle-memory check; every gate subprocess runs in its own
session and the whole tree is killed on timeout; every evaluation is
reproducible from (task spec, patches, gate). Gate subprocesses run from
`/` unless a gate passes another directory (`proc.run` default), and the CLIs
make their path arguments absolute: `python -m` puts the working directory
first on `sys.path`, and the bench v1 measurements were invalid because a
working directory holding the live `verl_omni` shadowed the candidate's
(`test_proc_on_cpu.py` checks this).

Concurrent loops on one host: verl names its weight-sync sockets
`/tmp/rl-colocate-zmq-{ray job id}-replica-{r}-rank-{k}.sock`, and the first
driver of every fresh local Ray cluster gets job id `01000000`. The first two
concurrent T1 runs therefore bound the same sockets, and both stock
references crashed at the step-2 weight sync (an NCCL watchdog timeout in one
run, a vLLM `TypeError` in the other). A private `/tmp` per gate is not
possible because the container lacks `CAP_SYS_ADMIN` and user namespaces
(`unshare` fails). Instead the harness puts `rsi/isolation/sitecustomize.py`
on every gate's `PYTHONPATH`; it appends `RSI_RAY_JOB_SUFFIX` (random per
gate, set by `gate_env` and `g2_loop.sh`) to `RuntimeContext.get_job_id()`.
Ray never calls that method internally, and in verl and verl-omni it only
names these sockets, so candidate code is unchanged. Agents' own loops still
use the default ID. Two agents running as different users get a bind error
instead of cross-talk, because neither can replace the other's socket in the
sticky `/tmp`.

## 7. Ledger and knowledge base

**Ledger** (`ledger.jsonl`, append-only, one line per iteration): iteration,
agent id, hypothesis, diff stat and patch path, gate results, verdict
(`improved | neutral | regressed | invalid`), GPU-seconds used.

**Knowledge base** (`kb/`): short, checkable lessons distilled from the ledger
after each task, e.g. "HF Qwen3 RMSNorm casts to bf16 before the weight
multiply; vLLM multiplies in fp32". Each entry records the evidence (ledger
lines, gate results) and the scope it was verified on. Mature entries are
promoted, by a human, into `.claude/skills/` or
`docs/contributing/common_pitfalls.md`.

Tools count as knowledge too: if the agent writes a reusable diagnostic
(for example a per-op diff harness), it is reviewed and added to the toolset
for later tasks. KB and toolset are versioned as snapshots so runs can be
compared with and without them.

Snapshots live in `kb/vN/{KB.md, tools/}` and are immutable once a run has
used them; a run records which snapshot it got in `run_config.json`.
`kb/v1` was distilled from solving T0 by hand: general entries (rollout
kernels are the reference; batch invariance; same function is not same
kernel; custom ops need the vLLM config; 0-diff is not correct gradients;
grad-norm sanity; backward speed; sampling nondeterminism; full-depth
memory) and Qwen3-Omni entries, plus `kernel_invariance.py` and
`backward_check.py`. Entries are written as checkable claims with evidence
and scope. The T0 control run gets no KB; transfer tasks are run with and
without it.

## 8. Integrity

The worker may reach "0 mismatches" or "faster" by changing the measurement
instead of the system. The trusted side checks for:

- edits outside `editable`, or to evaluator, task spec, tests used by gates;
- actor reusing rollout log-probs (or vice versa) in the measured path;
- changed sampling parameters, batch size, sequence length, layer count,
  dtype, or disabled CUDA graphs when the task requires them;
- weights not actually synced (digest of every sent and received tensor,
  per sync).

g0 checks file scope and integrity patterns on added and removed lines
(`rsi/tests/test_g0_on_cpu.py` covers each violation type). It is a
tripwire, not a proof: an obfuscated edit can slip past a regex. The
runtime checks (actor replay reproduces the loop, weights match, real
update, solved claims re-run from clean) are the backstop, and every
solved candidate gets human review.

Any "solved" claim is re-run by the orchestrator from a clean worktree before
it is reported.

Blindness for T0: no web access for the worker, and the pinned commits
predate verl#8025, verl#8026 and vllm-omni#8186. If web access is needed
later, every fetched URL is logged and reported.

Isolation. The Cursor CLI sandbox needs Linux ≥ 6.2 and the host runs 5.10;
mount namespaces need `CAP_SYS_ADMIN`. The agent therefore runs as a
dedicated unprivileged user, `rsi-agent`, via `runuser`:

- the evaluator, harness, runs, reference solutions, the hand-solved dev
  tree and `/root` are mode 700 (owner root), so the agent cannot read the
  gates' code, previous ledgers or the human solution;
- the workspace (`agent.workspace_root/<run>`) is chowned to the agent each
  iteration; private `TMPDIR`/`RAY_TMPDIR`; `CUDA_VISIBLE_DEVICES` set to the
  leased GPUs; GPU lock files are not writable by the agent;
- every process of the user is killed after each turn, so concurrent runs
  use distinct agent users (`--agent cursor:MODEL@rsi-agent2`); the agent
  runs under `umask 077` and each workspace is mode 700, so one run cannot
  read another's work or its scratch files in `/tmp`;
- mode 700 does not separate runs of the same user, and root-owned
  workspaces were world-readable (T2 KB iteration 4 listed `agent_ws/`,
  which held `refrun_t0` and the finished no-KB workspace; no run opened
  another's files). Now: `agent_ws` is mode 711, a finished run's workspace
  is chowned to root (`agent.lock`), and the preflight refuses to start if
  the agent user can read the runs dir, `agent_ws` or any other workspace;
- at the start of a run the user's `~/.cursor` (CLI chats, transcripts) and
  `~/tmp` are wiped, so a "no KB" run cannot inherit an earlier run's
  history;
- network stays on (the model API needs it), so the stream-json log is
  audited after each turn: web tools, URLs, `git log`/`reflog`-style history
  reads, plus per-task patterns (human-solution identifiers, hidden paths).
  Audit hits are recorded in the ledger and the report, not blocking.

## 9. What we measure

| question | measurement |
|---|---|
| does the loop solve T0? | reached 0-diff within budget; GPU-hours, iterations |
| how does it compare with the human? | causes found vs the #637 per-op table; wall-clock vs the RFC timeline (approximate) |
| does T1 improve throughput? | median tok/s vs the HF-numerics baseline, at 0 mismatches |

**Throughput of the existing 0-diff solutions, v2** (`bench`,
`rsi_work/bench/v2`, candidate code on the path, actor FSDP2 on 2 GPUs;
% of stock on the same pins):

| candidate | rollout eager bs32 / bs128 | rollout CUDA graph bs32 / bs128 | actor log-prob | actor fwd+bwd |
|---|---|---|---|---|
| Qwen3-Omni stock (tok/s) | 679 / 2640 | n/a | 44850 | 10148 |
| T0 by hand | 87% / 87% | n/a | 125% | 99% |
| T0 no-KB agent | 62% / 63% | n/a | 88% | 93% |
| MiniCPM-o stock (tok/s) | 1959 / 6364 | 3817 / 9249 | 48070 | 15662 |
| T2 no-KB agent | 45% / 48% | 64% / 73% | 97% | 96% |
| T2 KB agent | 61% / 67% | 40% / 71% | 93% | 81% |

**Full-depth RL step time, MiniCPM-o** (`bench_loop`, `rsi_work/bench/loop_v2`,
2 GPUs, eager rollout, LoRA, 32 prompts x 8, median of steps 2-5, seconds):

| candidate | step | gen | old_log_prob | update_actor | update_weights | diff max |
|---|---|---|---|---|---|---|
| stock | 32.5 | 17.6 | 2.3 | 5.1 | 7.1 | 0.47 |
| T2 no-KB agent | 54.5 (+68%) | 42.6 | 2.4 | 5.1 | 3.6 | 0 |
| T2 KB agent | 39.2 (+20%) | 26.8 | 2.5 | 5.9 | 3.7 | 0 |

Weight sync is faster in both 0-diff candidates than in stock; not
investigated. For comparison, cr-gao reports +10.6% step time for his
Qwen3-Omni 0-diff stack (vllm-omni#8186), with CUDA graphs, vLLM 0.32, short
responses (generation is about 17-27% of his step vs 54% here).

**Convergence: skip the log-prob recompute?** (`bench_loop --bench-profile
converge[_bypass]`, `rsi_work/bench/converge`, MiniCPM-o LoRA GSPO on GSM8K,
150 steps, 2 GPUs per arm). Arms: A stock with the actor's recompute; B
stock with `algorithm.rollout_correction.bypass_mode=true` (old log-probs
= rollout log-probs, no recompute); C the T2 KB 0-diff candidate with
bypass; D the same candidate with recompute (separates "0-diff code" from
"skip recompute"). On verl's v1 trainer bypass only replaces
`old_log_probs`; the loss stays GSPO. n = 1 per arm; validation on 256 rows
(standard error about 0.015 near 0.92).

| arm | val step 0 / 25 / 150 | ppo_kl (mean) | clipfrac steps 1-25 / 126-150 | step s | old_log_prob s |
|---|---|---|---|---|---|
| A stock, recompute | 0.309 / 0.918 / 0.941 | 1e-6 to 2e-4 | 0.119 / 0.042 | 32.1 | 2.22 |
| B stock, bypass | 0.305 / 0.852 / 0.922 | 2.3e-3 | 0.278 / 0.105 | 29.9 | 0.11 |
| C 0-diff, bypass | 0.297 / 0.918 / 0.941 | 7.4e-4 | 0.113 / 0.041 | 38.0 | 0.12 |
| D 0-diff, recompute | 0.297 / 0.902 / 0.934 | 8.0e-4 | 0.118 / 0.035 | 40.3 | 2.36 |

GSM8K saturates for MiniCPM-o by step 25 (mostly the `####` format), so this
tests stability, not learning speed. Skipping the recompute with stock's
mismatch (B) clips 2-3x more and is behind at step 25; with the 0-diff
candidate (C) it is indistinguishable from A and saves the 2.2 s recompute.
D keeps `rollout_probs_diff_max = 0` for all 150 full-depth steps.

C and D have the same steady `ppo_kl` (about 8e-4, vs about 1e-5 to 2e-4
for A). It is not a train/inference forward mismatch: with one mini-batch
per step (`ppo_mini_batch_size` = 32) and bypass, the candidate logs
`ppo_kl = 0.0` and clip fraction 0.0 on every step at full depth on 2 GPUs
(`rsi_work/scratch/update_probe`), and `rsi.evaluate.update_probe` finds
`compute_log_prob`, the train-mode forward (with and without backward,
either micro-batching) and the rollout bitwise equal on the truncated
model. The KL comes from the second mini-batch, after one optimizer step.
Why it is larger than stock's there is unverified; a candidate: the 0-diff
actor computes LoRA through the merged bf16 weight W + sBA (as vLLM serves
it), so a small update shows up only where it crosses a bf16 rounding step,
while stock's unmerged fp32 LoRA moves smoothly. One mini-batch per step
with bypass is a cheap end-to-end check that the update is exactly
on-policy (`ppo_kl` and clip fraction exactly 0).

**INVALID, superseded by v2:** v1 ran the bench jobs with cwd = the live
`verl-omni` checkout, which `python -m` puts first on `sys.path`, so
`verl_omni` came from the live repo for every candidate (stock included) and
no candidate's `verl-omni` patches were measured; only `CANDIDATE_ENV`,
`verl` and `vllm-omni` were the candidate's. Bench jobs now run with
cwd = the candidate worktree.

**Throughput of the existing 0-diff solutions** (`bench`, `rsi_work/bench/v1`,
% of stock on the same pins; Qwen3-Omni CUDA graphs cannot start on these
pins for any candidate: the thinker modifies a module buffer in forward):

| candidate | rollout eager bs32 / bs128 | rollout CUDA graph bs32 / bs128 | actor log-prob | actor fwd+bwd |
|---|---|---|---|---|
| Qwen3-Omni stock (tok/s) | 663 / 2533 | n/a | 25973 | 5439 |
| T0 by hand | 91% / 92% | n/a | 100% | 98% |
| T0 no-KB agent | 63% / 61% | n/a | 100% | 99% |
| MiniCPM-o stock (tok/s) | 1994 / 6705 | 3711 / 9207 | 25423 | 8312 |
| T2 no-KB agent | 45% / 43% | 66% / 74% | 98% | 98% |
| T2 KB agent | 63% / 66% | 40% / 70% | 99% | 99% |

(v1 conclusions about the actor were wrong: see v2 above.)
| is it recursive? | cost to solve a task with KB/tool snapshot vs without, on a task not seen when the KB was built |

## 10. Tasks

**T0, calibration.** Start: current pins (verl 0.10.0.dev0, vLLM 0.30.0,
vLLM-Omni 0.30.0rc2 `12e9280`), which contain none of the #637 fixes.
Gates: `g0`, `g1`, `g2`. Note #637 was measured on vLLM 0.28 and H20;
our box is 4x H200. Both are SM90, so the same attention kernels (FA3) are
available, but differences may not be identical on vLLM 0.30, which is
acceptable for a blind run.

Status: solved by hand (g2 0/16384, g1 0/65536, full 48-layer model
0/2899), opt-in via `VLLM_BATCH_INVARIANT=1`. That makes T0 a known-solvable
calibration task: the autonomous control run (fresh agent, no KB) is
measured against it.

**T0-LoRA (`t1_qwen3omni_thinker_lora_zero_diff`), first transfer task.**
Start: the pins plus the T0 solution (`start.patches`). Scope adds LoRA
rank 32, alpha 64, fp32 adapters on q/k/v/o, merged into the synced weights.
The baseline is 0-diff at step 1 (B = 0) but fails g3: the parity attention
builds its QKV matmul from the base weights, so the q/k/v LoRA gradients are
zero (12 of 16 B matrices) and the gradient norm is 0.32 of the reference.
Run with and without `kb/v1` to measure the recursive effect early, before
T2.

**Objective types.** Tasks so far fall into three shapes, which the task
spec and verdict logic should support:

| type | example | objective | constraints |
|---|---|---|---|
| equality | T0, T0-LoRA | mismatch = 0 | g3 |
| maximize under constraint | T1 | tok/s vs a frozen baseline measurement | g1, g2, g3 stay passing |
| convergence | later | reward / eval score at fixed steps, several seeds | 0-diff or bounded mismatch, budget |

The first two are implemented (section 6); the second uses a stock
measurement per session rather than a frozen baseline. The third needs
multi-seed runs and a statistical verdict, since single runs are noisy.

T1 task: `t1_minicpmo_thinker_zero_diff_throughput` starts from the T2 KB
candidate; objective = worst ratio to stock of eager and CUDA-graph decode
at batch 128, actor log-prob and actor fwd+bwd; target 0.9; gates as T2.
The start candidate scores 0.673 (eager decode). Both arms solved it at
iteration 5 (0.955 without KB, 0.944 with KB v2), confirmed at full depth
with zero diff (step 28.2 s vs stock 31.4 s).

T1-Qwen task: `t1_qwen3omni_thinker_zero_diff_throughput` starts from the
T0 control-run (agent) solution and uses LoRA. Full-parameter training does
not fit on 4 GPUs at full depth, see section 6. The objective is the worst
ratio to stock of eager decode at batch 128, actor log-prob and actor
fwd+bwd; CUDA-graph decode is not scored because graphs do not start for
this model on these pins. Target 0.9, 60 GPU-hours. The full-depth
confirmation takes all 4 GPUs, so only one arm runs: KB v3 (KB v2 plus the
MiniCPM-o throughput techniques), testing whether they transfer to an MoE
model.

**T1, throughput under 0-diff.** Start: a 0-diff stack (the #637 PRs once
mergeable, or the T0 result). Objective: maximize `g_bench_rollout` tok/s.
Constraint: `g1` and `g2` stay at 0. Off limits: evaluator, sampling, batch
size, scope. Expected search space: kernels that reproduce the HF operation
order bit for bit (RMSNorm, residual add, rope), faster batch-invariant
kernels, `_grouped_mm` tuning, CUDA-graph capture sizes. First step is to
re-measure the baseline on our hardware (#637 reports 6467 vs 10250 tok/s
with CUDA graphs on H20; H200 numbers will differ).

**T2 (`t2_minicpmo_thinker_zero_diff`), transfer to a new model family:
MiniCPM-o 4.5 thinker 0-diff.** Weights at `/home/public/models/MiniCPM-o-4_5`.
The LLM is a dense Qwen3 (36 layers, hidden 4096, q/k norm, no MoE); the HF
side is `trust_remote_code` modeling with a stock `Qwen3ForCausalLM` inside,
and no human solution exists, so it tests whether the KB generalizes. Start:
the stock pins (the T0 solution is Qwen3-Omni specific). Checkpoint: 4 LLM
layers, towers kept (`make_truncated_ckpt --family minicpm_o`).

What the stock pins required:

- **transformers 5.14.1.** The checkpoint's `processing_minicpmo.py` calls
  `AutoImageProcessor.register` with a string key, which 5.13 rejects; the
  upstream recipe was validated on 5.14.1. T2 runs on a thin venv
  (`rsi_work/venvs/t2_tf5141`): its own site-packages holds only transformers
  5.14.1 and chains to the main venv's site-packages after it. The main venv
  is unchanged.
- **LoRA scope.** Full fine-tuning is unsupported at the pin: the adapter
  leaves `apm`/`vpm`/`resampler` unsharded on the assumption that they are
  frozen, and without LoRA nothing freezes them. T2 uses the recipe's LoRA
  (rank 32, alpha 64, fp32, merged, q/k/v/o).
- **Runtime pipeline registration.** `minicpmo_4_5_thinker_only` exists only
  after verl-omni's rollout adapter registers it, so g1 calls the adapter's
  `ensure_pipeline_registered` before rebuilding the engine, as the rollout
  server does.

Harness changes are keyed by `scope.family` (default `qwen3_omni`):
`loop_config.FAMILY_OVERRIDES` / `FAMILY_PIPELINE`, the g3 reference loader,
1-D vs M-RoPE position ids, and the `thinker.` / `llm.` name prefix. g3's
candidate call goes through the engine adapter's `prepare_model_inputs`
(identity for Qwen3-Omni). Qwen3-Omni equivalence: override set identical,
fp32 g3 reference bitwise identical, stock candidate loss identical. The bf16
stock yardstick is not bitwise reproducible run to run (flash-attention
backward), with or without the change.

Stock baseline: g3 passes (16 LoRA B, min cosine 0.9968 vs 0.9994 stock);
g2 16384/16384 mismatched (max |Δ| 0.29); g1 actor replay = loop, rollout alone
vs in-batch 0, prefill vs actor 8192/8192; first divergence `L0.input_norm`,
as on Qwen3-Omni. Run without and with `kb/v1` (frozen as is; its tools are
partly Qwen3-Omni specific). Coordination note for the verl-omni#345 owners
drafted, not sent.

## 11. Layout

```
rsi/
  DESIGN.md
  tasks/              # task specs (YAML)
  evaluate/           # evaluator and gates (trusted)
  orchestrator/       # loop, worktrees, GPU locks, agent adapters
  kb/                 # knowledge-base snapshots
  runs/               # ledgers and reports (gitignored)
```

Kept framework-agnostic where possible (task spec, evaluator contract,
ledger, orchestrator) so it can move to a separate repo once it has worked on
at least two tasks.

## 12. Phase 1 milestones (about 2 weeks, 4 GPUs)

1. Task spec schema and T0 spec; pinned start environment builds from it. Done.
2. `g1` offline suite and the raw log-prob dump on both sides; verify it
   reports nonzero mismatches on the current pins. Done.
3. `g2` loop check and `g0` scope checks. Done.
4. Orchestrator with one agent adapter, ledger, timeouts. Done (no-op dry run).
5. T0 solved by hand; gaps found along the way closed in the framework
   (candidate env, agent GPUs, isolation and audit, g3, LoRA scope,
   `start.patches`, KB v1). Reference solution validated through the loop.
6. T0 control run (Cursor CLI, Claude Opus 5.5 High, no KB): solved
   autonomously in 11 iterations, 15.4 GPU-hours, 4.7 h wall clock for the
   resumed part (iterations 1-5 on h1, 6-11 on h3 after the g0 fix). Path:
   HF-exact RMSNorm, attention with the actor's bf16 RoPE `inv_freq`, HF-exact
   MoE in the rollout, the actor's log-prob kernel in the rollout, `o_proj`
   padded to 128 rows in decode, and RMSNorm variance padded to 16 rows in
   decode (PyTorch's reduction order changes below 16 rows). All
   rollout-side and opt-in; g3 passes (min cosine 0.976 vs 0.984 stock).
   Iterations 9 and 10 timed out and were scored "improved" on noise.
7. T0-LoRA with vs without `kb/v1` (harness h2, `rsi-agent2`). Both solved
   in one iteration (n = 1 each).
8. T2 MiniCPM-o: model-family support in the harness (harness h4), stock
   baseline on all gates, then runs without and with `kb/v1` on GPUs 2-3
   (`rsi-agent2`). No-KB run solved in 11 iterations, 19.6 GPU-hours,
   including CUDA-graph prefill (opaque custom ops so Inductor cannot fuse
   the parity formulas). Changes on both sides: actor uses vLLM's FA3
   kernel, fp32 RoPE frequencies, fp32 log-softmax and a merged-weight LoRA
   forward; rollout uses HF-exact RMSNorm and RoPE, single-split attention,
   16-row RMSNorm padding in decode and no split-K. Iterations 4-8 were lost
   to the depth-only progress key. KB run: iteration 1 already 0-diff in the
   loop and every eager g1 check (2.4 GPU-hours), all actor-side in
   `verl-omni`, reusing the KB's "actor adopts vLLM numerics +
   `VLLM_BATCH_INVARIANT=1`" pattern; CUDA-graph prefill pending. Its
   iteration 4 made no change and argued the parent meets the objective,
   since its CUDA-graph fix had been rejected by the g0 prose rule.
   Iteration 5 did the same: the brief printed no g1 status, so the failing
   `cudagraph_prefill_vs_actor` row read as optional. Stopped at 8.0 GPU-h
   and resumed from its best candidate as `t2_kb_opus55_r2` on h5 (g1
   status in the brief, g0 prose fix, step-aware progress; g1-g3 unchanged).
   The KB vs no-KB comparison therefore crosses a harness change.
   Solved at iteration 6 on h5 (11.07 GPU-h, confirm re-run passed: g2
   0/16384, g1 0/73728 including CUDA-graph prefill, g3 0.9995 vs stock
   0.9994), vs no-KB at iteration 11 (19.6 GPU-h); n=1 each. Actor side via
   the KB pattern; rollout side via an opt-in `VLLM_OMNI_BATCH_INVARIANT_KERNELS`
   (custom ops on under Inductor; batch-invariant RMSNorm as an opaque op,
   because Inductor re-generates the Triton kernel with `eps` typed fp64).
   The fp64-`eps` finding is new; it is a KB v2 candidate.

### Baseline on the current pins (4x H200)

- g2: 16383 to 16384 of 16384 tokens mismatched over the two steps, mean
  |Δ| ≈ 0.07, max 2.6 to 3.2 depending on the sampled responses (sampling
  differs between runs). Weight sync verified (923 tensors, 3 syncs).
- g1: actor replay = loop (0/8192); vLLM prefill vs actor 8192/8192;
  prefill vs decode 8184/8192; actor alone vs packed 0.
- First divergence: `L0.input_norm` (embedding equal; all 1078 tokens differ,
  max 0.00195), consistent with #637's RMSNorm finding.
- vLLM alone vs in-batch: 0 on most copies, but one run had 29/8192 on one
  copy, so vLLM prefill is not fully batch-invariant on these pins.
- Known gaps: CUDA graphs do not start on this stack at baseline ("assigning
  buffers during forward is not allowed under cudagraph"), so the CUDA-graph
  check reports `unavailable`; the rollout-side `logits` probe op is missing
  because vLLM computes logits for one fewer position.

## 13. Open questions

- Progress noise: g2 samples new responses every run, so its mismatch count
  varies for the same code. T0 r2 iteration 9 (agent timed out; only inert
  debug hooks added over iteration 8) scored "improved" (10648 to 7662 of
  16384). Options: rank progress on a fixed-fixture metric (g1 on a frozen
  batch), or require a margin calibrated from repeated runs of one candidate.
  Iteration 10 repeated it (timeout, only more debug hooks, 7129): one
  functional candidate scored 10648, 7662 and 7129 across three runs.
- g0 false positives on prose: T2 no-KB iteration 10 and KB iteration 2 were
  rejected for ``enforce_eager`` in comments and docstrings (no code touched
  the mode). Fixed for the next snapshot: whole-line comments are skipped and
  the eager/graph rule matches only code uses (attribute, keyword or
  assignment, quoted key).
- CUDA-graph acceptance is uneven across tasks: on Qwen3-Omni the check is
  `unavailable` (graphs do not start on these pins), so T0 was solved
  without it; on MiniCPM-o graphs start and the check counts.
- Agent timeouts: a turn that times out with no hypothesis still costs a full
  evaluation (about 2 GPU-hours on T0) and carries its debug hooks forward
  into the best candidate. Consider marking it `timeout` without gating, and
  telling the agent in the brief to submit before the limit.

- Per-iteration agent budget: 45 minutes (`iteration_timeout_min`); a
  token cap is not enforced yet. Agent: Cursor CLI in print mode as
  `rsi-agent` (section 8), model pinned per run so runs are comparable.
- A full candidate evaluation is about 1.4 GPU-hours (g3 + g2 + g1 +
  confirm). Two lanes on 4 GPUs would halve wall clock; the loop is
  sequential for now.
- Decided: the agent gets GPUs for its own experiments (`agent.gpus`), and
  their GPU-seconds count against the budget.

Decided: 4-layer truncation is a pre-written checkpoint loaded by both
sides; the T0 worker has no web access; the harness contains none of the
#637 fixes.
