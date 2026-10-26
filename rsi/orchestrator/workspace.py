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
"""Candidate workspaces: one detached git worktree per repo, reset per iteration.

A candidate is a set of patches against the task's start commits plus an
optional ``CANDIDATE_ENV`` file at the workspace root (opt-in switches such
as ``VLLM_BATCH_INVARIANT=1``, one ``KEY=VALUE`` per line). Nothing is ever
committed: a workspace is reset to the start commit, the parent candidate is
applied, the agent edits, and the result is saved back as patches.
"""

import hashlib
import os
import re
import subprocess

ENV_FILE = "CANDIDATE_ENV"
_ENV_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def parse_env(text: str) -> tuple[dict, list]:
    """``KEY=VALUE`` lines (blank lines and ``#`` comments ignored) -> (env, malformed lines)."""
    env, bad = {}, []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _ENV_LINE.match(line)
        if m:
            env[m.group(1)] = m.group(2)
        else:
            bad.append(line)
    return env, bad


def read_env(root: str) -> dict:
    path = os.path.join(root, ENV_FILE)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return parse_env(f.read())[0]


def _git(repo, *args, input_text=None):
    # The agent user owns the workspace files while the orchestrator (root) runs git on them.
    return subprocess.run(
        ["git", "-c", "safe.directory=*", "-C", repo, *args],
        check=True,
        capture_output=True,
        text=True,
        input=input_text,
    ).stdout


class Workspace:
    def __init__(self, task: dict, root: str):
        self.task = task
        self.root = root
        self.repos = task["start"]["repos"]

    def path(self, name):
        return os.path.join(self.root, name)

    def ensure(self):
        os.makedirs(self.root, exist_ok=True)
        for name, spec in self.repos.items():
            if not os.path.isdir(self.path(name)):
                _git(spec["path"], "worktree", "add", "--detach", self.path(name), spec["sha"])

    def reset(self, patches: dict | None = None):
        """Reset every repo to its start commit, then apply ``patches`` ({repo: diff text, ENV_FILE: text})."""
        self.ensure()
        for name, spec in self.repos.items():
            repo = self.path(name)
            _git(repo, "checkout", "--detach", "--force", spec["sha"])
            _git(repo, "clean", "-fdq")
            text = (patches or {}).get(name)
            if text:
                _git(repo, "apply", "--whitespace=nowarn", "-", input_text=text)
        env_path = os.path.join(self.root, ENV_FILE)
        env_text = (patches or {}).get(ENV_FILE)
        if env_text:
            with open(env_path, "w") as f:
                f.write(env_text)
        elif os.path.exists(env_path):
            os.remove(env_path)

    def diff(self) -> dict:
        """Current changes per repo as a patch against the start commit, untracked files included, plus ENV_FILE."""
        patches = {}
        for name, spec in self.repos.items():
            repo = self.path(name)
            _git(repo, "add", "--intent-to-add", "--all")
            text = _git(repo, "diff", "--binary", spec["sha"])
            _git(repo, "reset", "-q")
            if text:
                patches[name] = text
        env_path = os.path.join(self.root, ENV_FILE)
        if os.path.exists(env_path):
            with open(env_path) as f:
                text = f.read()
            if any(parse_env(text)):
                patches[ENV_FILE] = text
        return patches


def patch_id(patches: dict) -> str:
    h = hashlib.sha256()
    for name in sorted(patches):
        h.update(name.encode())
        h.update(patches[name].encode())
    return h.hexdigest()[:16] if patches else "base"


def save_patches(patches: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for name, text in patches.items():
        with open(os.path.join(out_dir, name if name == ENV_FILE else f"{name}.diff"), "w") as f:
            f.write(text)


def load_patches(in_dir: str) -> dict:
    out = {}
    if not in_dir or not os.path.isdir(in_dir):
        return out
    for fname in os.listdir(in_dir):
        if fname.endswith(".diff") or fname == ENV_FILE:
            with open(os.path.join(in_dir, fname)) as f:
                out[fname if fname == ENV_FILE else fname[: -len(".diff")]] = f.read()
    return out
