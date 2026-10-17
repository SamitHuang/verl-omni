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

"""Instruction-following plus source-consistency reward for image editing RL.

PickScore against the edit instruction alone never sees the source image, so a
policy can raise it by discarding the source and rendering only what the
instruction names. This reward adds a DINOv2 image-to-image similarity between
the source and the edited output and penalizes it only below a threshold:

    score = pickscore_weight * PickScore(instruction, output) / pickscore_divisor
            - consistency_weight * max(0, consistency_threshold - cos(DINO(source), DINO(output)))

The hinge keeps "copy the source unchanged" from being rewarded: above the
threshold, consistency adds nothing and only instruction following counts.
The dataset row must carry the source image as ``extra_info["source_image"]``
(PNG/JPEG bytes, ``{"bytes": ...}``, or a PIL image).
"""

import asyncio
import gc
import io
from concurrent.futures import ThreadPoolExecutor

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from verl_omni.utils.reward_score.clipscore_reward import _decoded_reward_image, _to_rgb_image
from verl_omni.utils.reward_score.pickscore_reward import _MODEL_PATH as PICKSCORE_MODEL_PATH
from verl_omni.utils.reward_score.pickscore_reward import _PROCESSOR_PATH as PICKSCORE_PROCESSOR_PATH
from verl_omni.utils.reward_score.pickscore_reward import _pairwise_pickscore, _PickScoreInferencer
from verl_omni.utils.reward_score.preference_mix_reward import _resolve_dtype

_DINO_MODEL_PATH = "facebook/dinov2-base"


def _source_image(extra_info: dict | None) -> Image.Image:
    source = (extra_info or {}).get("source_image")
    if isinstance(source, dict):
        source = source.get("bytes")
    if isinstance(source, bytes | bytearray):
        with Image.open(io.BytesIO(source)) as image:
            return image.convert("RGB")
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    raise ValueError("Edit consistency reward requires the source image in extra_info['source_image']")


class _DINOv2Inferencer:
    def __init__(self, device, dtype: torch.dtype, model_path: str = _DINO_MODEL_PATH):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype).eval().to(device)

    @torch.no_grad()
    def infer(self, images: list[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        embeddings = self.model(**inputs).pooler_output.float()
        return torch.nn.functional.normalize(embeddings, dim=-1).cpu()


class EditConsistencyNativeModel:
    """One managed native deployment holding PickScore and DINOv2."""

    def __init__(
        self,
        device=None,
        dtype: str | torch.dtype = "bfloat16",
        pickscore_model_path: str = PICKSCORE_MODEL_PATH,
        pickscore_processor_path: str = PICKSCORE_PROCESSOR_PATH,
        dino_model_path: str = _DINO_MODEL_PATH,
        max_batch_size: int = 16,
    ):
        if max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {max_batch_size}")
        self.device = torch.device(device or "cpu")
        model_dtype = _resolve_dtype(dtype)
        self.pickscore = _PickScoreInferencer(
            device=self.device,
            dtype=model_dtype,
            model_path=pickscore_model_path,
            processor_path=pickscore_processor_path,
        )
        self.dino = _DINOv2Inferencer(self.device, model_dtype, dino_model_path)
        self.max_batch_size = max_batch_size
        # Owned so close() can wait for a batch whose awaiting consumer was cancelled.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="edit-consistency")
        self._queue = asyncio.Queue()
        self._consumer_task = None
        self._consumer_lock = asyncio.Lock()
        self._closed = False

    async def _ensure_consumer(self) -> None:
        if self._closed:
            raise RuntimeError("Edit consistency reward model is closed")
        if self._consumer_task is not None and not self._consumer_task.done():
            return
        async with self._consumer_lock:
            if self._closed:
                raise RuntimeError("Edit consistency reward model is closed")
            if self._consumer_task is None or self._consumer_task.done():
                self._consumer_task = asyncio.create_task(self._consumer_loop())

    def _infer_requests(self, requests) -> list[dict[str, torch.Tensor]]:
        prompts = [prompt for prompt, _, _, _ in requests]
        images = [_to_rgb_image(image) for _, image, _, _ in requests]
        sources = [_to_rgb_image(source) for _, _, source, _ in requests]
        pickscore = self.pickscore.infer(prompts, images)
        dino = self.dino.infer(images + sources)
        output_dino, source_dino = dino[: len(requests)], dino[len(requests) :]
        return [
            {
                "pickscore_text_embedding": pickscore["text_embeddings"][index],
                "pickscore_image_embedding": pickscore["image_embeddings"][index],
                "pickscore_logit_scale": pickscore["logit_scale"],
                "dino_output_embedding": output_dino[index],
                "dino_source_embedding": source_dino[index],
            }
            for index in range(len(requests))
        ]

    async def _consumer_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            request = await self._queue.get()
            if request[0] is None:
                return

            requests = [request]
            should_stop = False
            await asyncio.sleep(0)
            while len(requests) < self.max_batch_size:
                try:
                    request = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if request[0] is None:
                    should_stop = True
                    break
                requests.append(request)

            try:
                outputs = await loop.run_in_executor(self._executor, self._infer_requests, requests)
                for (*_, future), output in zip(requests, outputs, strict=True):
                    if not future.done():
                        future.set_result(output)
            except asyncio.CancelledError:
                for *_, future in requests:
                    future.cancel()
                raise
            except Exception as error:
                for *_, future in requests:
                    if not future.done():
                        future.set_exception(error)
            if should_stop:
                return

    async def infer(self, prompts, images, sources):
        prompts, images, sources = list(prompts), list(images), list(sources)
        if not len(prompts) == len(images) == len(sources):
            raise ValueError("Edit consistency reward prompts, images and sources must have the same length")
        await self._ensure_consumer()

        loop = asyncio.get_running_loop()
        futures = []
        for prompt, image, source in zip(prompts, images, sources, strict=True):
            future = loop.create_future()
            futures.append(future)
            await self._queue.put((prompt, image, source, future))
        return await asyncio.gather(*futures)

    async def close(self) -> None:
        self._closed = True
        if self._consumer_task is not None and not self._consumer_task.done():
            await self._queue.put((None, None, None, None))
            await self._consumer_task
        self._consumer_task = None
        await asyncio.to_thread(self._executor.shutdown, wait=True)
        for name in ("pickscore", "dino"):
            if hasattr(self, name):
                delattr(self, name)
        gc.collect()
        accelerator = getattr(torch, self.device.type, None)
        empty_cache = getattr(accelerator, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()


async def compute_score_edit_consistency(
    data_source: str,
    solution_image,
    ground_truth: str,
    extra_info: dict,
    reward_model,
    pickscore_weight: float = 1.0,
    consistency_weight: float = 0.5,
    consistency_threshold: float = 0.4,
    pickscore_divisor: float = 26.0,
) -> dict[str, float]:
    """Score one edited image for instruction following and source consistency."""
    del data_source
    image = _decoded_reward_image(solution_image, extra_info)
    source = _source_image(extra_info)
    outputs = await reward_model.infer(prompts=[ground_truth or ""], images=[image], sources=[source])
    output = outputs[0]

    pickscore = _pairwise_pickscore(
        output["pickscore_text_embedding"].unsqueeze(0),
        output["pickscore_image_embedding"].unsqueeze(0),
        output["pickscore_logit_scale"],
        score_divisor=pickscore_divisor,
    ).item()
    consistency = torch.dot(output["dino_output_embedding"], output["dino_source_embedding"]).item()
    penalty = max(0.0, consistency_threshold - consistency)
    score = pickscore_weight * pickscore - consistency_weight * penalty
    return {
        "score": score,
        "pickscore_raw": pickscore,
        "dino_consistency_raw": consistency,
        "consistency_penalty": penalty,
    }
