# Config Explanation

Last updated: 07/30/2026

VeRL-Omni builds on [verl](https://github.com/verl-project/verl) and reuses the
same Hydra config surface for shared RL trainer fields (`data`,
`actor_rollout_ref`, `algorithm`, `trainer`, `reward_model`, and so on).

For the full field-by-field reference of those shared options, see the upstream
verl documentation:

- **[Config Explanation](https://verl.readthedocs.io/en/latest/examples/config.html)**

## Trainer entry points in VeRL-Omni

VeRL-Omni composes configs from two primary trainer YAMLs under
[`verl_omni/trainer/config/`](https://github.com/verl-project/verl-omni/tree/main/verl_omni/trainer/config):

| Trainer | Config | Typical use |
|---------|--------|-------------|
| Diffusion | `diffusion_trainer.yaml` | Image / video / audio diffusion RL (FlowGRPO, MixGRPO, Diffusion-DPO, …) |
| Omni | `omni_trainer.yaml` | Omni-modality models (e.g. Qwen3-Omni GSPO); inherits verl `ppo_trainer` via Hydra `searchpath` |

Recipe scripts under `examples/` pass Hydra overrides on the command line.
Precedence is lowest to highest:

```text
trainer YAML defaults  →  recipe config (if any)  →  CLI overrides
```

## Where to look next

- **Shared PPO / FSDP / rollout knobs** — [verl Config Explanation](https://verl.readthedocs.io/en/latest/examples/config.html)
- **Diffusion algorithm knobs** — algorithm pages under {doc}`algo/flowgrpo` and siblings (each has a Configuration section)
- **Rollout batching** — {doc}`start/rollout_batching`
- **Profiler** — {doc}`perf/profiler`
- **Model catalogue and example scripts** — {doc}`start/models`
