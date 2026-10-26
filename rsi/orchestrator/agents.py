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
"""Agent adapters. An agent edits the workspace and writes ``HYPOTHESIS.md`` at its root.

``run(root, log_path, timeout_s, env)`` gets the GPUs leased for the agent's
turn in ``env["CUDA_VISIBLE_DEVICES"]``.
"""

import os
import pwd
import shutil
import subprocess

from rsi.evaluate import proc

BRIEF_NAME = "RSI_BRIEF.md"
HYPOTHESIS_NAME = "HYPOTHESIS.md"


class NoopAgent:
    """Dry-run agent. Optionally applies fixed patches ({repo: path to diff}) to exercise the gates.

    A ``CANDIDATE_ENV`` entry is copied to the workspace root instead of applied with git.
    """

    name = "noop"

    def __init__(self, scripted_patches: dict | None = None):
        self.scripted_patches = scripted_patches or {}

    def run(self, root, log_path, timeout_s, env=None):
        from rsi.orchestrator.workspace import ENV_FILE

        for repo, diff_path in self.scripted_patches.items():
            if repo == ENV_FILE:
                with open(diff_path) as src, open(os.path.join(root, ENV_FILE), "w") as dst:
                    dst.write(src.read())
            else:
                subprocess.run(["git", "-C", os.path.join(root, repo), "apply", diff_path], check=True)
        with open(os.path.join(root, HYPOTHESIS_NAME), "w") as f:
            f.write("no-op agent: " + (", ".join(self.scripted_patches) or "no change") + "\n")
        with open(log_path, "w") as f:
            f.write("noop\n")
        return "ok"

    def unreadable(self, paths):
        return []

    def readable(self, paths):
        return []

    def lock(self, root):
        pass

    def reset_home(self):
        pass


class CursorCliAgent:
    """Cursor CLI in print mode.

    The CLI sandbox needs Linux >= 6.2. Where it is unavailable, isolation comes
    from ``run_as``: the agent runs as an unprivileged user that can write only
    its workspace and cannot read the evaluator, other runs or reference
    solutions (directory permissions, see DESIGN.md section 8). Every process of
    that user is killed after the turn so nothing keeps running into the gates.
    """

    name = "cursor-cli"

    def __init__(
        self,
        model: str | None = None,
        binary: str = "rsi-agent-cli",
        run_as: str | None = "rsi-agent",
        sandbox: bool = False,
    ):
        self.model = model
        self.binary = binary
        self.run_as = run_as
        self.sandbox = sandbox

    def reset_home(self):
        """Start a run without CLI state (chats, transcripts) or scratch files from earlier runs of this user."""
        if not self.run_as:
            return
        home = pwd.getpwnam(self.run_as).pw_dir
        for name in (".cursor", "tmp"):
            shutil.rmtree(os.path.join(home, name), ignore_errors=True)

    def unreadable(self, paths):
        """Files (or errors) under ``paths`` the agent user cannot read."""
        if not self.run_as:
            return []
        out = subprocess.run(
            ["runuser", "-u", self.run_as, "--", "find", *paths, "!", "-readable"],
            capture_output=True,
            text=True,
            cwd="/",
        )
        return out.stdout.split() + ([out.stderr.strip()] if out.returncode else [])

    def readable(self, paths):
        """Those of ``paths`` the agent user can read (files) or list (directories)."""
        if not self.run_as or not paths:
            return []
        script = 'for p; do [ -r "$p" ] && echo "$p"; done; true'
        out = subprocess.run(
            ["runuser", "-u", self.run_as, "--", "sh", "-c", script, "sh", *paths],
            capture_output=True,
            text=True,
            cwd="/",
        )
        return out.stdout.split()

    def lock(self, root):
        """Take a finished workspace away from the agent user; later runs by the same user must not read it."""
        if self.run_as and os.path.isdir(root):
            subprocess.run(["chown", "-R", "root:root", root], check=True)
            os.chmod(root, 0o700)

    def run(self, root, log_path, timeout_s, env=None):
        cmd = [self.binary, "-p", "--output-format", "stream-json", "--force", "--trust", "--workspace", root]
        cmd += ["--sandbox", "enabled" if self.sandbox else "disabled"]
        if self.model:
            cmd += ["--model", self.model]
        cmd.append(f"Read {BRIEF_NAME} in the workspace root and follow it exactly.")
        run_env = {**os.environ, **(env or {})}
        if self.run_as:
            home = pwd.getpwnam(self.run_as).pw_dir
            subprocess.run(["chown", "-R", f"{self.run_as}:{self.run_as}", root], check=True)
            os.chmod(root, 0o700)  # other agent users (concurrent runs) must not read this run's work
            tmp = os.path.join(home, "tmp")
            subprocess.run(["install", "-d", "-o", self.run_as, "-g", self.run_as, tmp], check=True)
            # Private temp dirs: Ray and Triton state in /tmp created by root's gates is not writable by the agent.
            # umask 077: files the agent leaves in shared places like /tmp stay unreadable to other agent users.
            base = [
                "runuser",
                "-u",
                self.run_as,
                "--",
                "sh",
                "-c",
                'umask 077 && exec "$@"',
                "sh",
                "env",
                f"HOME={home}",
                f"USER={self.run_as}",
                "PATH=/usr/local/bin:/usr/bin:/bin",
                f"TMPDIR={tmp}",
                f"RAY_TMPDIR={tmp}",
            ]
            cmd = base + [f"{k}={v}" for k, v in (env or {}).items()] + cmd
        try:
            status, _, _ = proc.run(cmd, run_env, log_path, timeout_s, cwd=root)
        finally:
            if self.run_as:
                subprocess.run(["pkill", "-KILL", "-u", self.run_as], check=False)
        return status


def make_agent(spec: str):
    kind, _, arg = spec.partition(":")
    if kind == "noop":
        scripted = dict(item.split("=", 1) for item in arg.split(",") if item) if arg else {}
        return NoopAgent(scripted)
    if kind == "cursor":
        # cursor[:model][@user]: concurrent runs need distinct users, since each turn ends with pkill -u.
        model, _, user = arg.partition("@")
        return CursorCliAgent(model=model or None, run_as=user or "rsi-agent")
    raise ValueError(f"unknown agent {spec!r}; use noop[:repo=diff,...] or cursor[:model][@user]")
