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
"""Gate g0: static scope and integrity checks on a candidate worktree.

A candidate fails if any changed or new file is outside ``editable``, inside
``protected``, or if an added/removed line matches an integrity pattern
(measurement tampering such as feeding rollout log-probs to the actor).
"""

import fnmatch
import os
import re
import subprocess

# (name, regex on added or removed diff lines). Changes matching these alter
# what is measured rather than the system under test.
INTEGRITY_PATTERNS = [
    ("bypass_mode", re.compile(r"bypass_mode")),
    ("rollout_logprobs_to_actor", re.compile(r"old_log_probs?\W.*rollout_log_probs|rollout_log_probs.*old_log_probs?")),
    # MoE routers also call their expert count top_k; those assignments name the experts.
    ("sampling_params", re.compile(r"\b(temperature|top_k|top_p|min_p|repetition_penalty)\b\s*[=:](?!.*expert)")),
    ("layer_count", re.compile(r"num_hidden_layers")),
    # Code uses only (attribute, keyword or assignment, quoted key); prose naming the setting is allowed.
    (
        "eager_or_graph_mode",
        re.compile(
            r"\.(enforce_eager|cudagraph_mode)\b|\b(enforce_eager|cudagraph_mode)\s*[=:]|[\"'](enforce_eager|cudagraph_mode)[\"']"
        ),
    ),
    ("rsi_env", re.compile(r"RSI_[A-Z_]+|VERL_USE_EXTERNAL_MODULES")),
]


def _git(repo, *args):
    return subprocess.run(
        ["git", "-c", "safe.directory=*", "-C", repo, *args], check=True, capture_output=True, text=True
    ).stdout


def changed_files(repo, base_sha):
    tracked = _git(repo, "diff", "--name-only", base_sha).split()
    untracked = _git(repo, "ls-files", "--others", "--exclude-standard").split()
    return sorted(set(tracked) | set(untracked))


def diff_lines(repo, base_sha):
    text = _git(repo, "diff", "--unified=0", base_sha)
    current = None
    for line in text.splitlines():
        if line.startswith("+++ "):
            current = line[6:] if line.startswith("+++ b/") else None
        elif line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            yield current, line
    for path in _git(repo, "ls-files", "--others", "--exclude-standard").split():
        try:
            with open(f"{repo}/{path}", errors="replace") as f:
                for line in f:
                    yield path, "+" + line.rstrip("\n")
        except OSError:
            continue


def _matches(path, globs):
    return any(fnmatch.fnmatch(path, g) for g in globs)


def check_env(task: dict, worktree: str) -> list:
    """Violations in the candidate's ``CANDIDATE_ENV``: malformed lines, keys not allowed by the task."""
    from rsi.orchestrator.workspace import ENV_FILE, parse_env

    path = os.path.join(worktree, ENV_FILE)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        env, bad = parse_env(f.read())
    rules = task.get("candidate_env_rules") or {}
    allow = [re.compile(p) for p in rules.get("allow", [])]
    deny = [re.compile(p) for p in rules.get("deny", [])]
    out = [{"file": ENV_FILE, "reason": "malformed env line", "line": line[:200]} for line in bad]
    for key in env:
        if any(p.fullmatch(key) for p in deny) or not any(p.fullmatch(key) for p in allow):
            out.append({"file": ENV_FILE, "reason": "env key not allowed", "line": key})
    return out


def run_g0(task: dict, worktree: str) -> dict:
    editable = task["editable"]
    protected = task.get("protected", [])
    violations, flags, files = check_env(task, worktree), [], []
    for name, repo in task["start"]["repos"].items():
        repo_path = f"{worktree}/{name}"
        for path in changed_files(repo_path, repo["sha"]):
            full = f"{name}/{path}"
            files.append(full)
            if _matches(full, protected):
                violations.append({"file": full, "reason": "protected"})
            elif not _matches(full, editable):
                violations.append({"file": full, "reason": "outside editable"})
        for path, line in diff_lines(repo_path, repo["sha"]):
            if line[1:].lstrip().startswith("#"):
                continue  # whole-line comment
            for pname, pattern in INTEGRITY_PATTERNS:
                if pattern.search(line):
                    flags.append({"file": f"{name}/{path}", "pattern": pname, "line": line[:200]})
    status = "pass" if not violations and not flags else "fail"
    return {
        "gate": "g0_static",
        "status": status,
        "metrics": {"files_changed": len(files)},
        "checks": [{"name": "changed_files", "files": files}],
        "scope_violations": violations,
        "integrity_flags": flags,
    }
