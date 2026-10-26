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
"""Hydra overrides for the gates' training loop, shared by g1 and g2.

``python -m rsi.evaluate.loop_config`` prints one override per line for
``g2_loop.sh``; ``g1`` composes the same config in-process.
"""

import json
import os
import sys


def lora_overrides(lora) -> list[str]:
    """Overrides for ``scope.lora``: false, or {rank, alpha, dtype, merge, target_modules}."""
    if not lora:
        return ["actor_rollout_ref.model.lora_rank=0"]
    targets = "[" + ",".join(f"'{m}'" for m in lora["target_modules"]) + "]"
    return [
        f"actor_rollout_ref.model.lora_rank={lora['rank']}",
        f"actor_rollout_ref.model.lora_alpha={lora['alpha']}",
        f"actor_rollout_ref.model.lora_dtype={lora.get('dtype', 'float32')}",
        f"actor_rollout_ref.model.lora.merge={bool(lora.get('merge', True))}",
        f"actor_rollout_ref.model.target_modules={targets}",
    ]


# vLLM-Omni pipeline (verl-omni rollout adapter) per ``scope.family``.
FAMILY_PIPELINE = {"qwen3_omni": "qwen3_omni_moe", "minicpm_o": "minicpmo_4_5"}

# Model-specific overrides (``scope.family``); everything else is shared.
FAMILY_OVERRIDES = {
    "qwen3_omni": [
        "actor_rollout_ref.model.exclude_modules=.*talker.*|.*code2wav.*|.*code_predictor.*|.*visual.*|.*audio_tower.*",
        "actor_rollout_ref.actor.freeze_vision_tower=true",
        f"+actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name={FAMILY_PIPELINE['qwen3_omni']}",
        "reward.reward_manager.name=naive",
    ],
    # From examples/gspo_trainer/minicpm/run_minicpmo_4_5_thinker_gspo_lora_avqa_v1.sh.
    "minicpm_o": [
        "actor_rollout_ref.model.trust_remote_code=True",
        "actor_rollout_ref.model.exclude_modules=.*vpm.*|.*apm.*|.*talker.*|.*code2wav.*|.*code_predictor.*|.*codec.*"
        "|.*audio_decoder.*|.*audio_generator.*|.*audio_head.*|.*tts.*|.*vocoder.*",
        "data.custom_cls.path=pkg://verl_omni.utils.dataset.omni_rl_datasets",
        "data.custom_cls.name=MiniCPMORLHFDataset",
        "+data.apply_chat_template_kwargs.enable_thinking=false",
        f"+actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name={FAMILY_PIPELINE['minicpm_o']}",
        "reward.reward_manager.name=minicpm_naive",
    ],
}


def overrides(
    ckpt: str,
    harness: str,
    dump_dir: str,
    data_dir: str,
    steps: int,
    n_gpus: int = 2,
    lora=None,
    family: str = "qwen3_omni",
) -> list[str]:
    return [
        *FAMILY_OVERRIDES[family],
        f"data.train_files={data_dir}/train.parquet",
        f"data.val_files={data_dir}/test.parquet",
        "data.train_batch_size=4",
        "data.max_prompt_length=512",
        "data.max_response_length=1024",
        "data.truncation=error",
        "data.filter_overlong_prompts=true",
        "data.shuffle=false",
        f"actor_rollout_ref.model.path={ckpt}",
        *lora_overrides(lora),
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.actor.strategy=fsdp2",
        "actor_rollout_ref.actor.optim.lr=1e-5",
        "actor_rollout_ref.actor.optim.clip_grad=1.0",
        "actor_rollout_ref.actor.ppo_mini_batch_size=4",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384",
        "actor_rollout_ref.actor.use_kl_loss=false",
        "actor_rollout_ref.actor.policy_loss.loss_mode=gspo",
        "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean",
        "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16",
        "actor_rollout_ref.actor.fsdp_config.param_offload=false",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=false",
        "actor_rollout_ref.rollout.n=2",
        "actor_rollout_ref.rollout.temperature=1.0",
        "actor_rollout_ref.rollout.top_k=-1",
        "actor_rollout_ref.rollout.top_p=1.0",
        "actor_rollout_ref.rollout.calculate_log_probs=True",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.enforce_eager=True",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.5",
        "actor_rollout_ref.rollout.load_format=safetensors",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384",
        "actor_rollout_ref.rollout.enable_prefix_caching=False",
        "actor_rollout_ref.rollout.agent.num_workers=1",
        "+actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode=ar",
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        "reward.reward_manager.source=register",
        f"reward.custom_reward_function.path={harness}/rsi/evaluate/reward.py",
        "reward.custom_reward_function.name=compute_score",
        "trainer.val_before_train=false",
        "trainer.balance_batch=False",
        "trainer.critic_warmup=0",
        'trainer.logger=["console"]',
        "trainer.project_name=rsi",
        "trainer.experiment_name=g2",
        f"trainer.default_local_dir={dump_dir}/ckpt",
        f"trainer.n_gpus_per_node={n_gpus}",
        "trainer.nnodes=1",
        "trainer.save_freq=-1",
        "trainer.test_freq=-1",
        "trainer.resume_mode=disable",
        f"trainer.total_training_steps={steps}",
    ]


if __name__ == "__main__":
    env = os.environ
    for line in overrides(
        ckpt=env["RSI_CKPT"],
        harness=env["RSI_HARNESS"],
        dump_dir=env["RSI_DUMP_DIR"],
        data_dir=env.get("RSI_DATA_DIR", "/home/public/yx/data/gsm8k"),
        steps=int(env.get("RSI_STEPS", "2")),
        lora=json.loads(env.get("RSI_LORA", "false")),
        family=env.get("RSI_FAMILY", "qwen3_omni"),
    ):
        sys.stdout.write(line + "\n")
