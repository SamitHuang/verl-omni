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
"""Run a gate subprocess with a hard timeout and full process-tree cleanup.

Ray, vLLM and dataloader workers are descendants of the gate process, so the
tree is collected before killing; nothing outside it (for example another
user's Ray cluster) is touched.
"""

import os
import subprocess
import time

import psutil


def _tree(pid):
    try:
        root = psutil.Process(pid)
        return [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        return []


def _alive(procs):
    alive = []
    for p in procs:
        try:
            if p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
                alive.append(p)
        except psutil.NoSuchProcess:
            pass
    return alive


def kill_tree(procs, grace_s=10):
    # Polls instead of psutil.wait_procs: its pidfd_open path fails with
    # EINVAL for some descendants on this kernel.
    for p in procs:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    deadline = time.time() + grace_s
    alive = _alive(procs)
    while alive and time.time() < deadline:
        time.sleep(0.5)
        alive = _alive(alive)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass


def run(cmd, env, log_path, timeout_s, cwd="/"):
    """Run ``cmd``; return ``(status, returncode, wall_clock_s)``.

    ``status`` is ``ok``, ``error`` or ``timeout``. All descendants are killed
    when the command exits or times out. ``cwd`` defaults to ``/`` because
    ``python -m`` puts the cwd first on ``sys.path``: an inherited cwd holding
    another ``verl_omni`` (e.g. the live repo) would shadow the candidate's.
    """
    start = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=cwd, start_new_session=True)
        seen = {}
        status = None
        while status is None:
            for p in _tree(proc.pid):
                seen[p.pid] = p
            try:
                proc.wait(timeout=5)
                status = "ok" if proc.returncode == 0 else "error"
            except subprocess.TimeoutExpired:
                if time.time() - start > timeout_s:
                    status = "timeout"
        for p in _tree(proc.pid):
            seen[p.pid] = p
        kill_tree([p for p in seen.values() if p.pid != os.getpid()])
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    return status, proc.returncode, time.time() - start
