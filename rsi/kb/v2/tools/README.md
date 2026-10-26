# rsi_tools (KB v1 toolset)

Diagnostics distilled from T0. Run from the workspace root on your leased GPUs with
`PYTHONPATH=verl-omni:verl:vllm-omni`.

| tool | what it checks |
|---|---|
| `kernel_invariance.py` | vLLM batch-invariant kernels give identical rows across batch sizes, prefill vs decode, packing |
| `backward_check.py` | each custom autograd wrapper in `verl_omni/utils/vllm_parity.py` vs an fp32 reference, with plain bf16 autograd as the bar |

These are not part of the candidate and are not run by the evaluator.
