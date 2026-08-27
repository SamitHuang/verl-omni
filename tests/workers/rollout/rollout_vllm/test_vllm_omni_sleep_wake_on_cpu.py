# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Hybrid sleep/wake must use AsyncOmni engine APIs, not raw worker RPCs.

``collective_rpc("sleep")`` skips the diffusion worker's pre-offload CUDA
sync and can segfault in ``CuMemAllocator`` ``cudaMemcpy``.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import (
    _LORA_REQUEST_CACHE_MISS,
    vLLMOmniHttpServer,
)


def _server(*, node_rank: int = 0) -> vLLMOmniHttpServer:
    server = object.__new__(vLLMOmniHttpServer)
    server.node_rank = node_rank
    server.engine = AsyncMock()
    server._lora_request_cache = MagicMock(name="cached_lora")
    return server


@pytest.mark.asyncio
async def test_sleep_hybrid_uses_engine_sleep_not_raw_rpc():
    server = _server()

    await server._sleep_hybrid()

    server.engine.sleep.assert_awaited_once_with(level=1)
    server.engine.reset_encoder_cache.assert_awaited_once_with()
    server.engine.collective_rpc.assert_not_called()
    assert server._lora_request_cache is _LORA_REQUEST_CACHE_MISS


@pytest.mark.asyncio
async def test_wake_up_uses_engine_wake_up_not_raw_rpc():
    server = _server()

    await server.wake_up()

    server.engine.wake_up.assert_awaited_once_with(tags=["weights"])
    server.engine.collective_rpc.assert_not_called()
    assert server._lora_request_cache is _LORA_REQUEST_CACHE_MISS


@pytest.mark.asyncio
async def test_wake_up_forwards_explicit_tags():
    server = _server()

    await server.wake_up(tags=["weights", "kv_cache"])

    server.engine.wake_up.assert_awaited_once_with(tags=["weights", "kv_cache"])


@pytest.mark.asyncio
async def test_wake_up_is_a_no_op_on_non_zero_node_rank():
    server = _server(node_rank=1)
    cached = server._lora_request_cache

    await server.wake_up()

    server.engine.wake_up.assert_not_called()
    server.engine.collective_rpc.assert_not_called()
    assert server._lora_request_cache is cached
