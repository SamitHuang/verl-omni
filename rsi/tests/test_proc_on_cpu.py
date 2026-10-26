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

import os
import sys

from rsi.evaluate import proc


def _package(root, mark):
    os.makedirs(root / "verl_omni")
    (root / "verl_omni" / "__init__.py").write_text(f"MARK = {mark!r}\n")


def test_gate_subprocess_imports_the_worktree_package_not_the_callers_cwd(tmp_path, monkeypatch):
    _package(tmp_path / "live", "live")
    _package(tmp_path / "worktree" / "verl-omni", "candidate")
    monkeypatch.chdir(tmp_path / "live")
    env = {**os.environ, "PYTHONPATH": str(tmp_path / "worktree" / "verl-omni")}
    log = tmp_path / "log"
    status, _, _ = proc.run([sys.executable, "-c", "import verl_omni; print(verl_omni.MARK)"], env, str(log), 60)
    assert status == "ok"
    assert log.read_text().strip() == "candidate"


def _fake_ray(root):
    os.makedirs(root / "ray")
    (root / "ray" / "__init__.py").write_text("from ray.runtime_context import RuntimeContext\n")
    (root / "ray" / "runtime_context.py").write_text(
        "class RuntimeContext:\n    def get_job_id(self):\n        return '01000000'\n"
    )


def _job_id(tmp_path, env):
    log = tmp_path / "log"
    cmd = [sys.executable, "-c", "import ray; print(ray.RuntimeContext().get_job_id())"]
    status, _, _ = proc.run(cmd, env, str(log), 60)
    assert status == "ok", log.read_text()
    return log.read_text().strip()


def test_gate_env_gives_each_gate_its_own_ray_job_id(tmp_path):
    from rsi.evaluate.common import gate_env

    harness = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _fake_ray(tmp_path / "fake")
    task = {"start": {"python": sys.executable, "repos": ["fake"]}}
    first, second = (gate_env(task, str(tmp_path), harness, "0") for _ in range(2))
    ids = {_job_id(tmp_path, first), _job_id(tmp_path, second)}
    assert len(ids) == 2
    assert all(i.startswith("01000000-") for i in ids)

    plain = {**first, "RSI_RAY_JOB_SUFFIX": ""}
    assert _job_id(tmp_path, plain) == "01000000"
