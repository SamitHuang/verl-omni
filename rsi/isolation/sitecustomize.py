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
"""Make the Ray job id unique per gate when ``RSI_RAY_JOB_SUFFIX`` is set.

verl names its weight-sync sockets ``/tmp/rl-colocate-zmq-{job_id}-...``.
Every fresh local Ray cluster gives its first driver the same job id, so two
loops on one host bind the same sockets and corrupt each other's weight sync.
This container cannot create mount namespaces, so a private ``/tmp`` is not an
option. Instead ``RuntimeContext.get_job_id`` gets a suffix in every process of
the gate, which keeps both ends of the socket consistent. The patch is applied
when ``ray.runtime_context`` is imported, so processes that never import Ray
are unaffected.

Shadows the system ``sitecustomize``, which only installs the apport hook.
"""

import functools
import importlib.abc
import importlib.util
import os
import sys

_SUFFIX = os.environ.get("RSI_RAY_JOB_SUFFIX")
_TARGET = "ray.runtime_context"


class _PatchJobId(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _TARGET:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        @functools.wraps(exec_module)
        def exec_and_patch(module):
            exec_module(module)
            get_job_id = module.RuntimeContext.get_job_id

            @functools.wraps(get_job_id)
            def suffixed(self):
                return f"{get_job_id(self)}-{_SUFFIX}"

            module.RuntimeContext.get_job_id = suffixed

        spec.loader.exec_module = exec_and_patch
        return spec


if _SUFFIX:
    sys.meta_path.insert(0, _PatchJobId())
