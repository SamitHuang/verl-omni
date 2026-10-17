# Train Boogu-Image with DiffusionNFT

Last updated: 10/17/2026

RL post-training for [Boogu-Image-0.1-Base](https://huggingface.co/Boogu/Boogu-Image-0.1-Base)
(text-to-image) with the DiffusionNFT trainer, using a `vllm-omni` rollout and
`Qwen3-VL-8B-Instruct` as a visual generative reward model on an OCR-style task.

The launcher is [`run_boogu_image_ocr_lora.sh`](https://github.com/verl-project/verl-omni/blob/main/examples/diffusionnft_trainer/boogu_image/run_boogu_image_ocr_lora.sh). It is the Boogu
sibling of [`../qwen_image/run_qwen_image_ocr_lora.sh`](https://github.com/verl-project/verl-omni/blob/main/examples/diffusionnft_trainer/qwen_image/run_qwen_image_ocr_lora.sh)
and differs in the model path, the LoRA targets and FSDP layer prefixes, the guidance knob
(`pipeline.guidance_scale` rather than Qwen's `true_cfg_scale`), rollout TP=1, and the dataset.

## Prerequisites

On top of the [standard install](../../../docs/start/install.md):

```bash
pip install "boogu-image @ git+https://github.com/boogu-project/Boogu-Image.git@$(cat .github/boogu_image_pin.txt)"
```

The training engine loads the checkpoint's canonical `BooguImageTransformer2DModel` through
`diffusers.AutoModel` with `trust_remote_code=True`. The checkpoint's
`transformer/transformer_boogu.py` is a shim that re-exports the class from the `boogu`
package, so that package must be importable on trainer workers. Rollout workers do not need
it — vllm-omni ships its own Boogu pipeline port.

## Data

```bash
python examples/flowgrpo_trainer/data_process/boogu_image_ocr.py \
    --input_dir ~/data/ocr --output_dir ~/data/ocr/boogu_image
```

The converter is shared with the FlowGRPO Boogu recipe and produces
`~/data/ocr/boogu_image/{train,test}.parquet`, which is where the launcher looks by default.
It writes `data_source: "ocr"` — the reward that scores these rows, not the task or the
trainer. Both trainers read `data_source` as `data.reward_fn_key` and the validation trainer
splices it into every metric key as `<val-core|val-aux>/<data_source>/<var>/<metric>`, so
T2I OCR runs log `val-core/ocr/reward/mean@1`. See the Edit section below for the naming rule.
Note the deliberate quirk documented there: the upstream pipeline encodes empty instructions
— including the default negative prompt `""` — with the *TI2I unified* system prompt, not the
T2I one. Do not "fix" this; a template mismatch between the data and the released model
silently shifts the policy's prompt distribution and collapses rewards.

The parquet must carry a `negative_prompt` column. Boogu is a *guided* model
(`pipeline.guidance_scale=4.0`), and both the training adapter and the rollout fail closed
when guidance is active without negative embeddings, rather than silently sampling unguided.

### Edit (TI2I) dataset

[`run_boogu_image_edit_lora.sh`](https://github.com/verl-project/verl-omni/blob/main/examples/diffusionnft_trainer/boogu_image/run_boogu_image_edit_lora.sh) is the TI2I sibling of
the launcher above. It defaults to the edit-specialised
[`Boogu/Boogu-Image-0.1-Edit`](https://huggingface.co/Boogu/Boogu-Image-0.1-Edit)
checkpoint rather than `Boogu-Image-0.1-Base`, which the T2I launcher above uses. It reads
a separate dataset, produced by the edit-specific converter:

```bash
python examples/flowgrpo_trainer/data_process/boogu_image_edit_ocr.py \
    --input_dir ~/data/ocr_edit --output_dir ~/data/ocr/boogu_image_edit_pickscore \
    --image_size 512 --reward pickscore
```

Each row pairs the source image in `images` with an "edit this text" instruction and a
`target_text`. The reward is **PickScore** — the same reward as the verified
[Qwen-Image-Edit TI2I recipe](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/qwen_image_edit/README.md) — which
CLIP-encodes `reward_model.ground_truth` as the prompt and scores its similarity to the
generated image. `ground_truth` is therefore the **instruction**, not the target word; the
converter writes it that way and keeps `target_text` in `extra_info`. With the OCR GenRM
(rather than PickScore) a bare target word would be the right `ground_truth`, so the two are
not interchangeable. `--reward` selects the arm and writes both fields together, so they
cannot drift apart:

| `--reward` | `ground_truth` | `data_source` | scored by |
| --- | --- | --- | --- |
| `pickscore` (default) | the instruction | `pickscore` | `pickscore_reward.py` (CLIP) |
| `ocr` | the target text | `ocr` | `genrm_ocr.py` (Qwen3-VL GenRM) |

`data_source` is the reward, and it is not cosmetic: both trainers read it as
`data.reward_fn_key` and the validation trainer splices it into every metric key as
`<val-core|val-aux>/<data_source>/<var>/<metric>`. Naming it after the task
(`ocr_edit`) or the algorithm (`flow_grpo/...`) makes `val-core/pickscore/reward/mean@1`
impossible to read and mislabels diffusionnft runs as flow_grpo; the T2I/EDIT
distinction belongs in `trainer.experiment_name`.

PickScore runs CLIP locally in the reward workers, so this recipe serves no reward model.

The converter writes a **text-only** negative prompt (`negative_prompt` carries no `<image>`
placeholder) for the edit arm: guided TI2I encodes the negative instruction without the reference
image (upstream default `use_input_images_4_neg_instruct=False`), so a placeholder there would be
tokenized but never expanded into image features — it would satisfy the media-count check while
quietly shifting the guidance. `RLHFDataset._build_messages` therefore permits the negative key to
consume **fewer** media than the row carries (`image_offset <= len(images)`), and only the negative
key; the positive prompt still has to match exactly. The T2I converter (`boogu_image_ocr.py`) and
the edit e2e fixture (`tests/special_e2e/create_dummy_image_edit_data.py --negative-prompt-mode
text-only`) emit the same text-only form. Qwen-Image-Edit's converter instead keeps the placeholder, but that is a different model family
whose negative branch does feed the image; do not copy it here.

The name is baked into the parquet at conversion time, so an existing dataset keeps its old key
until it is regenerated or rewritten in place with
[`set_data_source.py`](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/data_process/set_data_source.py). See
[metrics](../../../docs/start/metrics.md) for what the per-step DiffusionNFT keys mean.

## Launch

```bash
bash examples/diffusionnft_trainer/boogu_image/run_boogu_image_ocr_lora.sh
```

The script is configured for a single node with `4` GPUs. Model-specific constraints baked
into the recipe:

- `tensor_model_parallel_size=1` — the vllm-omni `BooguImagePipeline` supports neither TP nor
  SP nor CFG-parallel. The 20 GB bf16 DiT plus the Qwen3VL encoder and VAE must fit on one
  rollout GPU.
- `pipeline.guidance_scale=4.0` — Boogu uses standard sequential text CFG (upstream default
  `4.0`), driven by `guidance_scale` and not Qwen-style `true_cfg_scale`. Set `1.0` to disable
  CFG (halves rollout NFE).
- `pipeline.height/width` must be multiples of 16 and at most 2048².
- LoRA `target_modules` names the double-stream attention processors (`img_to_*`,
  `instruct_to_*`), single-stream and refiner attention (`to_*`, plus the joint attention's
  output projection), and both feed-forward variants. These are the paths the released
  checkpoint actually exposes and that FSDP layered-summon can transport to the rollout; a
  broader `all-linear` list names top-level embedders that never bind, so the sync would drop
  them. The list is validated at startup rather than allowed to fail silently.
- The objective is rebalanced away from the config defaults: `mix_beta=0.1`,
  `ref_kl_coef=10.0`, `adv_clip_max=1.0`, `clip_ratio=1e-5` (defaults are `0.5`, `0.0`, `5.0`,
  `1e-4`). With the defaults the reward term is compressed by the advantage clip while the
  reward-free contraction dominates the reported loss, and the graded   reward collapses even
  though `positive_loss` looks like it is improving. That collapse was measured with text CFG
  active and an incomplete LoRA sync; the rebalanced objective holds the baseline but has not
  been shown to improve it. The preference recipe below converges with the Qwen-Image values
  and CFG-free training.

### Preference reward (HPSv2.1 + PickScore + CLIPScore)

HPSv2.1 needs `open_clip_torch`, which is not a verl-omni dependency. Version 3.3.0 has been
validated with this recipe:

```bash
pip install open_clip_torch==3.3.0
python examples/flowgrpo_trainer/data_process/boogu_image_pickapic.py
HPSV2_PRETRAINED_PATH=/path/to/open_clip_pytorch_model.bin \
CUSTOM_REWARD_MODEL_PATH=/path/to/HPS_v2.1_compressed.pt \
bash examples/diffusionnft_trainer/boogu_image/run_boogu_image_pickapic_preference_lora_v1.sh
```

The recipe trains on 5,760 Pick-a-Pic v1 prompts and validates on 256 fixed prompts with a
fixed seed. It differs from the OCR recipe in two ways that both matter for convergence:

- `pipeline.guidance_scale=1.0` (CFG-free, as in DiffusionNFT). With text CFG active the
  training forward regresses the guided velocity `v_u + s * (v_c - v_u)` onto the policy's own
  samples. That extrapolated velocity is not the flow of any data distribution, so the update
  erodes the guidance: at `guidance_scale=4.0` the 40-step validation reward fell from 1.811 to
  1.745 within 10 steps and the images washed out. CFG-free also halves rollout and training
  cost.
- `ref_kl_coef=1e-4`, `actor.optim.lr=3e-4`, `adv_clip_max=5.0` (the Qwen-Image values). With
  `ref_kl_coef=10.0` and `lr=1e-4` the reference anchor dominates the gradient and the policy
  stays at the reference: 80 steps left the validation reward flat (1.812 -> 1.808).

Measured on 4 GPUs with the V1 sync trainer (24 prompts x 16 images per update, 10 rollout
steps, 512px):

| Step | Validation reward | HPSv2.1 | PickScore / 26 | CLIPScore | Paired win rate vs step 0 |
|------|-------------------|---------|----------------|-----------|---------------------------|
| 0    | 1.593             | 0.218   | 0.763          | 0.612     | -                         |
| 10   | 1.737             | 0.257   | 0.802          | 0.679     | 88%                       |
| 30   | 1.801             | 0.278   | 0.822          | 0.702     | 96%                       |
| 60   | 1.829             | 0.287   | 0.837          | 0.705     | 98%                       |

At step 60 the CFG-free policy also beats the CFG-4 base model on the same prompts and seeds
(1.829 vs 1.811, paired 95% CI of the difference [+0.003, +0.033]). A training step takes
about 8 minutes and a validation pass (256 prompts, 40 steps) about 4 minutes. The same recipe
on the deprecated V0 trainer gives a bit-identical step-0 validation and reaches 1.844 at step
60; that gap comes from single runs and has not been separated from run-to-run variance.

### Edit preference reward (PickScore + source consistency)

```bash
hf download osunlp/MagicBrush --repo-type dataset --local-dir ~/data/magicbrush_raw \
    --include "data/dev-*" $(printf 'data/train-%05d-* ' $(seq 0 4 48))
python examples/flowgrpo_trainer/data_process/boogu_image_edit_magicbrush.py \
    --input_dir ~/data/magicbrush_raw --output_dir ~/data/boogu_image_edit_magicbrush
bash examples/diffusionnft_trainer/boogu_image/run_boogu_image_edit_lora_v1.sh
```

The edit recipe trains `Boogu/Boogu-Image-0.1-Edit` on real MagicBrush photos with free-form
instructions and validates on 243 first-turn dev edits (removal instructions dropped; CLIP-family
scores do not model negation). It uses the CFG-free setting and the Qwen-Image values of the T2I
preference recipe, and replaces PickScore with
[`edit_consistency_reward.py`](https://github.com/verl-project/verl-omni/blob/main/verl_omni/utils/reward_score/edit_consistency_reward.py):

```text
score = PickScore(instruction, output) / 26 - 0.5 * max(0, 0.4 - cos(DINOv2(source), DINOv2(output)))
```

Both parts were needed:

- PickScore against the instruction never sees the source image. Trained on it alone, the policy
  raised validation PickScore by discarding the source and rendering only what the instruction
  names (a night sky for "add stars", a bare gradient for "blue background").
- The hinge must be low. On MagicBrush's human target edits the DINOv2 source similarity has a
  median of 0.94 but a 5th percentile of 0.36 (background and style changes). At a threshold of
  0.6 the penalty also hit those legitimate edits and the policy drifted toward copying the
  source (near-copies with similarity > 0.97 went from 43% to 61% in 10 steps). At 0.4 it only
  catches outputs that drop the source.

Measured on 4 GPUs with the V1 sync trainer (24 edits x 16 images per update, 10 rollout steps,
512px), 243 fixed dev edits with a fixed seed:

| Step | Validation reward | PickScore / 26 | DINOv2 source similarity | Paired win rate vs step 0 |
|------|-------------------|----------------|--------------------------|---------------------------|
| 0    | 0.764             | 0.766          | 0.893                    | -                         |
| 10   | 0.769             | 0.771          | 0.874                    | 69%                       |
| 30   | 0.777             | 0.778          | 0.866                    | 82%                       |
| 60   | 0.781             | 0.784          | 0.818                    | 87%                       |

At step 60 the policy also beats the CFG-4 base model on the same edits and seeds (0.781 vs
0.766, paired 95% CI of the difference [+0.012, +0.020], 77% wins) at the same source
similarity, and the share of near-copies falls from 43% to 21%. The share of outputs below the
hinge rises from 2% at step 30 to 6% at step 60 (CFG 4: 11%); inspect images before training
longer. A training step takes about 15 minutes and a validation pass about 5 minutes.

## Tests

Model-level CPU tests cover the conventions that break RL silently rather than loudly — the
velocity negation and text CFG that reach `scheduler.step`, the `sigma -> timestep` mapping,
the LoRA name translation, and the DiffusionNFT loss decomposition:

```bash
pytest tests/pipelines/test_boogu_image_diffusion_nft_on_cpu.py \
       tests/pipelines/test_boogu_image_diffusion_nft_engine_on_cpu.py \
       tests/pipelines/test_boogu_image_lora_mapping_on_cpu.py
```

The special E2E test covers parquet loading, vllm-omni rollout, reward computation, FSDP LoRA
training, and weight synchronization, and asserts the rollout engine actually bound every
actor delta (a pass with a dropped sync would be vacuous). It is wired into the GPU smoke
suite as `tests/gpu_smoke/run_gpu_smoke_diffusion_e2e.sh`; run it directly with:

```bash
# T2I
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 bash tests/special_e2e/run_diffusionnft_boogu_image.sh
# Edit (TI2I) — also exercises the reference-latent refiner path
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 MODE=edit bash tests/special_e2e/run_diffusionnft_boogu_image.sh
```
