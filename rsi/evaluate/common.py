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
"""Environment for gate subprocesses: candidate worktree first on PYTHONPATH."""

import glob
import os
import uuid


def gate_env(task: dict, worktree: str, harness: str, gpus: str, hooks: bool = False) -> dict:
    python = task["start"]["python"]
    site = os.path.join(os.path.dirname(os.path.dirname(python)), "lib", "python3.12", "site-packages")
    env = dict(os.environ)
    libs = sorted(glob.glob(os.path.join(site, "nvidia", "*", "lib")))
    env["LD_LIBRARY_PATH"] = ":".join([*libs, env.get("LD_LIBRARY_PATH", "")]).rstrip(":")
    repos = ":".join(os.path.join(worktree, name) for name in task["start"]["repos"])
    env["PYTHONPATH"] = f"{repos}:{harness}:{os.path.join(harness, 'rsi', 'isolation')}"
    env["RSI_RAY_JOB_SUFFIX"] = uuid.uuid4().hex[:12]
    env["VERL_USE_EXTERNAL_MODULES"] = "verl_omni,rsi.evaluate.hooks" if hooks else "verl_omni"
    env["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env.pop("RSI_DUMP_DIR", None)
    return env
