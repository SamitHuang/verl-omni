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

"""CPU regression test: a fork during a background video export must not hang the export."""

import multiprocessing
import threading
import time

import pytest
import torch

from verl_omni.utils.tracking import _export_video, media_export_lock

if "fork" not in multiprocessing.get_all_start_methods():
    pytest.skip("requires the fork start method", allow_module_level=True)


def test_fork_under_media_export_lock_waits_for_running_export(tmp_path):
    started = tmp_path / "ffmpeg_started"
    # Stands in for ffmpeg: it finishes only once every holder of its stdin pipe closes it,
    # and it reads late so the frames (larger than a pipe buffer) are still being written.
    fake_ffmpeg = tmp_path / "fake_ffmpeg"
    fake_ffmpeg.write_text(f'#!/bin/sh\ntouch "{started}"\nsleep 1\ncat > /dev/null\n')
    fake_ffmpeg.chmod(0o755)

    errors = []

    def export():
        try:
            _export_video(
                torch.zeros(4, 3, 128, 128, dtype=torch.uint8),
                str(tmp_path / "sample.mp4"),
                fps=24,
                ffmpeg_exe=str(fake_ffmpeg),
            )
        except Exception as error:
            errors.append(error)

    exporter = threading.Thread(target=export, daemon=True)
    exporter.start()
    deadline = time.monotonic() + 10
    while not started.exists():
        assert time.monotonic() < deadline, "fake ffmpeg did not start"
        time.sleep(0.01)

    # A long-lived fork, like a DataLoader worker; forked mid-export it would hold the pipe open.
    child = multiprocessing.get_context("fork").Process(target=time.sleep, args=(60,), daemon=True)
    try:
        with media_export_lock:
            child.start()
        exporter.join(timeout=10)
        assert not exporter.is_alive(), "video export hung after a concurrent fork"
        assert errors == []
    finally:
        child.terminate()
        child.join()
        exporter.join(timeout=10)
