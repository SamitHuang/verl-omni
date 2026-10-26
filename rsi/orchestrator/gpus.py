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
"""GPU leases: an exclusive file lock per GPU plus an idle check.

A GPU is leased only if its lock is free and it has less than
``max_used_mib`` allocated, so jobs started outside the harness are left alone.
"""

import contextlib
import fcntl
import os
import subprocess
import time


def _used_mib():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {int(i): int(m) for i, m in (line.split(",") for line in out.strip().splitlines())}


@contextlib.contextmanager
def lease(candidates: list[int], count: int, lock_dir: str, timeout_s: float = 3600, max_used_mib: int = 1024):
    """Yield a comma-separated list of ``count`` leased GPU indices."""
    os.makedirs(lock_dir, exist_ok=True)
    deadline = time.time() + timeout_s
    while True:
        held = []
        used = _used_mib()
        for gpu in candidates:
            if used.get(gpu, 1 << 30) >= max_used_mib:
                continue
            f = open(os.path.join(lock_dir, f"gpu{gpu}.lock"), "w")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                f.close()
                continue
            held.append((gpu, f))
            if len(held) == count:
                break
        if len(held) == count:
            try:
                yield ",".join(str(g) for g, _ in held)
            finally:
                for _, f in held:
                    fcntl.flock(f, fcntl.LOCK_UN)
                    f.close()
            return
        for _, f in held:
            fcntl.flock(f, fcntl.LOCK_UN)
            f.close()
        if time.time() > deadline:
            raise TimeoutError(f"could not lease {count} GPUs from {candidates}")
        time.sleep(15)
