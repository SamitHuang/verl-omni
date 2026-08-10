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
"""Compatibility helpers for vllm-omni postprocessors without envelope support."""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from typing import Any

_MEDIA_KEYS = ("image", "video", "output", "audio")


def envelope_aware_postprocessor(postprocess: Callable[..., Any]) -> Callable[..., Any]:
    """Adapt a media-only upstream postprocessor to a payload/metadata envelope."""

    @functools.wraps(postprocess)
    def wrapped(data: Any, **kwargs: Any) -> Any:
        if not _is_envelope(data):
            return postprocess(data, **kwargs)

        payload = data["payload"]
        metadata = dict(data.get("metadata") or {})
        media_key = next((key for key in _MEDIA_KEYS if key in payload), None)
        if media_key is None:
            raise ValueError("Diffusion output envelope has no media payload.")

        processed = postprocess(payload[media_key], **kwargs)
        if _is_envelope(processed):
            return {
                "payload": dict(processed["payload"]),
                "metadata": {**dict(processed.get("metadata") or {}), **metadata},
            }
        return {"payload": {media_key: processed}, "metadata": metadata}

    return wrapped


def _is_envelope(value: Any) -> bool:
    return isinstance(value, Mapping) and isinstance(value.get("payload"), Mapping)
