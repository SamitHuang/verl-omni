# Knowledge base snapshots

A snapshot is a directory `vN/` with `KB.md` (shown to the agent in the brief via
`--kb`) and `tools/` (copied into the workspace as `rsi_tools/` via `--toolset`).
Snapshots are immutable once a run has used them, so runs with and without a
snapshot stay comparable (DESIGN.md section 9). A new lesson goes into a new
snapshot.

Entry format: the lesson, how to check it, the evidence (run directories, gate
results) and the scope it was verified on. Lessons that hold across tasks are
promoted by a human into `.claude/skills/` or `docs/contributing/`.

| snapshot | built from | contents |
|---|---|---|
| v1 | T0 solved by hand (assistant-driven, outside the loop) | 10 general + 7 Qwen3-Omni entries; kernel-invariance and backward-check tools |
| v2 | v1 + bench / bench_loop / convergence measurements of the T0 and T2 candidates; section D from the vllm-omni#8186 / verl#8026 design (KB arm only) | v1 entries + 7 throughput entries + 2 design entries; v1 tools; frozen for the T1 runs (harness h6) |
| v3 | v2 + the MiniCPM-o throughput runs (T1, both arms), distilled by the assistant from their candidates and notes | v2 entries + 9 entries on recovering throughput with the same bits; v2 tools; for the Qwen3-Omni throughput run |

The `tools/` scripts in v1-v3 later got a license header and lint-only edits
(line wrapping, `zip(strict=True)`); the copies the runs used are in the frozen
harness snapshots.
