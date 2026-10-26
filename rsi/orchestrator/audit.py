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
"""Post-turn audit of the agent log.

Directory permissions stop the agent from reading hidden paths; the network
stays open because the agent needs its model API. This scan flags what
permissions cannot stop or what would show an attempt: web tools, references
to hidden paths, the human solution, or the evaluator. Hits are recorded in
the ledger for human review; they do not change the verdict.
"""

import json
import re

DEFAULT_RULES = [
    ("web_tool", r"\"(web_?search|web_?fetch|WebSearch|WebFetch)\""),
    ("url", r"https?://(?!localhost|127\.0\.0\.1)[\w.-]+"),
    ("git_history", r"git (log|show|rev-list|reflog)\b[^\"]*(--all|origin/|main\b)"),
]


def _call_text(kind, args):
    """What the agent asked for. Results and edit bodies are file contents (license URLs, the brief) and are skipped."""
    if kind == "shellToolCall":
        return str(args.get("command", ""))
    if kind in ("readToolCall", "editToolCall", "deleteToolCall"):
        return str(args.get("path", ""))
    return json.dumps(args)


def _tool_texts(log_path):
    """Requests from tool-call events of a stream-json log; plain text logs are scanned whole."""
    with open(log_path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                yield line
                continue
            if event.get("type") != "tool_call" or event.get("subtype") != "started":
                continue
            for kind, call in (event.get("tool_call") or {}).items():
                if kind.endswith("ToolCall") and isinstance(call, dict):
                    yield f'"{kind}" ' + _call_text(kind, call.get("args") or {})


def audit_log(log_path, extra_rules=()):
    """[(rule, excerpt)] for every match; ``extra_rules`` are (name, regex) pairs from the task spec."""
    rules = [(name, re.compile(pattern)) for name, pattern in [*DEFAULT_RULES, *extra_rules]]
    hits, seen = [], set()
    for text in _tool_texts(log_path):
        for name, pattern in rules:
            for m in pattern.finditer(text):
                excerpt = text[max(0, m.start() - 80) : m.end() + 80]
                key = (name, m.group(0))
                if key not in seen:
                    seen.add(key)
                    hits.append({"rule": name, "match": m.group(0), "excerpt": excerpt})
    return hits
