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
"""g0 and workspace round trip on a scratch workspace of the T0 repos."""

import os
import subprocess

import pytest
import yaml

from rsi.evaluate.__main__ import env_overlay
from rsi.evaluate.g0 import run_g0
from rsi.orchestrator.workspace import ENV_FILE, Workspace, load_patches, patch_id, read_env, save_patches

TASK = os.path.join(os.path.dirname(__file__), "..", "tasks", "t0_qwen3omni_thinker_zero_diff.yaml")


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    with open(TASK) as f:
        task = yaml.safe_load(f)
    if not all(os.path.isdir(r["path"]) for r in task["start"]["repos"].values()):
        pytest.skip("T0 repos not available")
    w = Workspace(task, str(tmp_path_factory.mktemp("rsi_ws")))
    w.ensure()
    yield task, w
    for name, spec in task["start"]["repos"].items():
        subprocess.run(["git", "-C", spec["path"], "worktree", "remove", "--force", w.path(name)], check=False)


def _write(path, text, mode="a"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode) as f:
        f.write(text)


def test_clean_and_benign_pass(ws):
    task, w = ws
    w.reset()
    assert run_g0(task, w.root)["status"] == "pass"
    _write(w.path("verl-omni") + "/verl_omni/__init__.py", "\n# benign\n")
    _write(w.path("verl") + "/verl/utils/rsi_new_helper.py", "X = 1\n", "w")
    _write(w.path("vllm-omni") + "/vllm_omni/__init__.py", "\nself.top_k = text_config.num_experts_per_tok\n")
    # Prose naming a guarded setting (T2 iterations rejected for these lines).
    _write(
        w.path("vllm-omni") + "/vllm_omni/__init__.py",
        "\n# WHY: with VLLM_BATCH_INVARIANT=1, enforce_eager runs vLLM's custom-op kernels,\n"
        '"""Compiled (CUDA-graph) execution with the ``enforce_eager`` kernels."""\n'
        "transformers bit for bit when they run eagerly. Without ``enforce_eager``\n",
    )
    r = run_g0(task, w.root)
    assert r["status"] == "pass", r


@pytest.mark.parametrize(
    "repo,path,text,reason",
    [
        ("verl-omni", "README.md", "\nextra\n", "outside editable"),
        ("verl-omni", "rsi/evaluate/compare.py", "x = 1\n", "protected"),
        ("verl", "tests/test_x.py", "x = 1\n", "protected"),
        ("verl", "verl/workers/config/rollout.py", "\nbypass_mode = True\n", "bypass_mode"),
        ("verl-omni", "verl_omni/__init__.py", "\nold_log_probs = rollout_log_probs\n", "rollout_logprobs_to_actor"),
        ("vllm-omni", "vllm_omni/__init__.py", "\nos.environ['RSI_DUMP_DIR'] = ''\n", "rsi_env"),
        ("vllm-omni", "vllm_omni/__init__.py", "\nkw['enforce_eager'] = False\n", "eager_or_graph_mode"),
        ("vllm-omni", "vllm_omni/__init__.py", "\nLLM(model, enforce_eager=True)\n", "eager_or_graph_mode"),
        ("vllm-omni", "vllm_omni/__init__.py", "\nif config.enforce_eager:\n", "eager_or_graph_mode"),
        ("verl-omni", "verl_omni/__init__.py", "\ncfg.temperature = 0.7\n", "sampling_params"),
        ("verl-omni", "verl_omni/__init__.py", "\nsampling.top_k = 50\n", "sampling_params"),
    ],
)
def test_violations_fail(ws, repo, path, text, reason):
    task, w = ws
    w.reset()
    _write(os.path.join(w.path(repo), path), text)
    r = run_g0(task, w.root)
    assert r["status"] == "fail"
    reasons = [v["reason"] for v in r["scope_violations"]] + [f["pattern"] for f in r["integrity_flags"]]
    assert reason in reasons, r


def test_removed_line_flagged(ws):
    task, w = ws
    w.reset()
    target = w.path("verl") + "/verl/workers/config/rollout.py"
    with open(target) as f:
        lines = f.readlines()
    idx = next(i for i, line in enumerate(lines) if "temperature" in line and "=" in line)
    del lines[idx]
    with open(target, "w") as f:
        f.writelines(lines)
    r = run_g0(task, w.root)
    assert r["status"] == "fail"
    assert any(f["pattern"] == "sampling_params" and f["line"].startswith("-") for f in r["integrity_flags"])


def test_patch_round_trip(ws):
    task, w = ws
    w.reset()
    _write(w.path("verl-omni") + "/verl_omni/__init__.py", "\n# change\n")
    _write(w.path("vllm-omni") + "/vllm_omni/rsi_new_file.py", "Y = 2\n", "w")
    patches = w.diff()
    assert set(patches) == {"verl-omni", "vllm-omni"}
    pid = patch_id(patches)
    w.reset()
    assert w.diff() == {}
    w.reset(patches)
    assert patch_id(w.diff()) == pid
    assert os.path.exists(w.path("vllm-omni") + "/vllm_omni/rsi_new_file.py")


def test_candidate_env_is_part_of_the_candidate(ws, tmp_path):
    task, w = ws
    w.reset()
    _write(os.path.join(w.root, ENV_FILE), "# opt-in\nVLLM_BATCH_INVARIANT=1\n", "w")
    patches = w.diff()
    assert patches == {ENV_FILE: "# opt-in\nVLLM_BATCH_INVARIANT=1\n"}
    assert patch_id(patches) != "base"
    save_patches(patches, str(tmp_path / "p"))
    assert load_patches(str(tmp_path / "p")) == patches
    w.reset()
    assert not os.path.exists(os.path.join(w.root, ENV_FILE))
    w.reset(patches)
    assert read_env(w.root) == {"VLLM_BATCH_INVARIANT": "1"}
    assert run_g0(task, w.root)["status"] == "pass"
    w.reset()


@pytest.mark.parametrize(
    "text", ["RSI_DUMP_DIR=/tmp\n", "VERL_USE_EXTERNAL_MODULES=x\n", "PYTHONPATH=/x\n", "not a line\n"]
)
def test_candidate_env_violations(ws, text):
    task, w = ws
    w.reset()
    _write(os.path.join(w.root, ENV_FILE), text, "w")
    r = run_g0(task, w.root)
    w.reset()
    assert r["status"] == "fail"
    assert {v["reason"] for v in r["scope_violations"]} & {"env key not allowed", "malformed env line"}, r


def test_env_overlay_restores(monkeypatch):
    monkeypatch.setenv("VLLM_KEEP", "a")
    monkeypatch.delenv("VLLM_NEW", raising=False)
    with env_overlay({"VLLM_KEEP": "b", "VLLM_NEW": "1"}):
        assert os.environ["VLLM_KEEP"] == "b" and os.environ["VLLM_NEW"] == "1"
    assert os.environ["VLLM_KEEP"] == "a" and "VLLM_NEW" not in os.environ
