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

import torch
from vllm_omni.diffusion.request import OmniDiffusionRequest

QWEN_IMAGE_VAE_SCALE_FACTOR = 8


def extract_custom_prompt(omni_req: OmniDiffusionRequest) -> dict:
    """Return the custom prompt dict from a single diffusion request."""
    prompt = omni_req.prompt
    return prompt if isinstance(prompt, dict) else {}


def coalesce_not_none(value, default):
    return default if value is None else value


def build_img_shapes(
    height: int, width: int, batch_size: int, vae_scale_factor: int
) -> list[list[tuple[int, int, int]]]:
    latent_height = height // vae_scale_factor // 2
    latent_width = width // vae_scale_factor // 2
    return [[(1, latent_height, latent_width)]] * batch_size


def pad_prompt_embeds_to_len(prompt_embeds: torch.Tensor, target_seq_len: int) -> torch.Tensor:
    """Pad or truncate prompt embeddings to a shared batch sequence length."""
    if prompt_embeds.ndim == 2:
        prompt_embeds = prompt_embeds.unsqueeze(0)
    _, seq_len, _ = prompt_embeds.shape
    if seq_len == target_seq_len:
        return prompt_embeds
    if seq_len > target_seq_len:
        return prompt_embeds[:, :target_seq_len]
    out = prompt_embeds.new_zeros((prompt_embeds.shape[0], target_seq_len, prompt_embeds.shape[2]))
    out[:, :seq_len] = prompt_embeds
    return out


def pad_prompt_mask_to_len(prompt_mask: torch.Tensor, target_seq_len: int) -> torch.Tensor:
    """Pad or truncate prompt masks to a shared batch sequence length."""
    if prompt_mask.ndim == 1:
        prompt_mask = prompt_mask.unsqueeze(0)
    _, seq_len = prompt_mask.shape
    if seq_len == target_seq_len:
        return prompt_mask
    if seq_len > target_seq_len:
        return prompt_mask[:, :target_seq_len]
    out = torch.zeros(
        (prompt_mask.shape[0], target_seq_len),
        dtype=prompt_mask.dtype,
        device=prompt_mask.device,
    )
    out[:, :seq_len] = prompt_mask
    return out


def gather_padded_prompt_batch(
    embeds_list: list[torch.Tensor],
    masks_list: list[torch.Tensor],
    *,
    max_sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pad variable-length per-request embeds to one length and concatenate."""
    target_seq_len = min(max(embeds.shape[1] for embeds in embeds_list), max_sequence_length)
    prompt_embeds = torch.cat(
        [pad_prompt_embeds_to_len(embeds, target_seq_len) for embeds in embeds_list],
        dim=0,
    )
    prompt_embeds_mask = torch.cat(
        [pad_prompt_mask_to_len(mask, target_seq_len) for mask in masks_list],
        dim=0,
    )
    return prompt_embeds, prompt_embeds_mask, target_seq_len


def rope_txt_seq_lens(prompt_embeds: torch.Tensor) -> list[int]:
    """Return per-row RoPE text lengths from padded embed width (diffusers semantics)."""
    seq_len = int(prompt_embeds.shape[1])
    return [seq_len] * int(prompt_embeds.shape[0])


def apply_true_cfg(
    noise_pred: torch.Tensor,
    negative_noise_pred: torch.Tensor,
    true_cfg_scale: float,
) -> torch.Tensor:
    comb_pred = negative_noise_pred + true_cfg_scale * (noise_pred - negative_noise_pred)
    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
    return comb_pred * (cond_norm / noise_norm)


class QwenImageTokenIdPromptMixin:
    """Encode pre-tokenized Qwen-Image prompts for rollout adapters."""

    def _get_qwen_prompt_embeds(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
    ):
        dtype = dtype or self.text_encoder.dtype

        if attention_mask is None:
            attention_mask = torch.ones_like(prompt_ids, dtype=torch.long)

        prompt_ids = prompt_ids.unsqueeze(0) if prompt_ids.ndim == 1 else prompt_ids
        attention_mask = attention_mask.unsqueeze(0) if attention_mask.ndim == 1 else attention_mask
        drop_idx = self.prompt_template_encode_start_idx
        encoder_hidden_states = self.text_encoder(
            input_ids=prompt_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            output_hidden_states=True,
        )
        hidden_states = encoder_hidden_states.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype)

        return prompt_embeds, encoder_attention_mask

    def encode_prompt(
        self,
        prompt_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        max_sequence_length: int = 1024,
    ):
        if prompt_embeds is None:
            if prompt_ids is None:
                raise ValueError("`prompt_ids` must be provided when `prompt_embeds` is None.")
            prompt_ids = prompt_ids.unsqueeze(0) if prompt_ids.ndim == 1 else prompt_ids
            attention_mask = (
                attention_mask.unsqueeze(0)
                if attention_mask is not None and attention_mask.ndim == 1
                else attention_mask
            )
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(prompt_ids, attention_mask=attention_mask)

        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]

        if num_images_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            prompt_embeds_mask = prompt_embeds_mask.repeat_interleave(num_images_per_prompt, dim=0)

        return prompt_embeds, prompt_embeds_mask
