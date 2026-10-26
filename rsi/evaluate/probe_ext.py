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
"""vLLM worker extension = probe methods + the loop's own worker extension.

vLLM accepts one ``worker_extension_cls``; g1 must keep the one the loop used
(``RSI_BASE_WORKER_EXT``, a qualified class name) so the engine is unchanged.
"""

import importlib
import os

from rsi.evaluate.probe import ProbeWorkerExtension

_base_name = os.environ.get("RSI_BASE_WORKER_EXT", "")
if _base_name:
    _module, _, _cls = _base_name.rpartition(".")
    _Base = getattr(importlib.import_module(_module), _cls)
    RsiProbeExtension = type("RsiProbeExtension", (ProbeWorkerExtension, _Base), {})
else:
    RsiProbeExtension = ProbeWorkerExtension
