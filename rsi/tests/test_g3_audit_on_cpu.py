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
"""g3 gradient comparison and the agent-log audit."""

import json

import torch

from rsi.evaluate.g3 import DEFAULTS, compare
from rsi.evaluate.grad_common import canonical, expected_lora_grads, grad_stats, lora_base_name, packed_inputs
from rsi.orchestrator.audit import audit_log


def _grads(seed, noise=0.0, base=None):
    g = torch.Generator().manual_seed(seed)
    base = base or {n: torch.randn(64, 32, generator=g) for n in ("a", "b", "c")}
    return {n: t + noise * torch.randn(t.shape, generator=g) * t.std() for n, t in base.items()}


def _ok(cand, stock=None):
    ref = _grads(0)
    stock = stock or _grads(1, 0.1, ref)
    return compare(grad_stats(ref, stock, cand), DEFAULTS)


def test_close_candidate_passes():
    ref = _grads(0)
    ok, checks, _ = _ok(_grads(2, 0.1, ref))
    assert ok, checks


def test_missing_gradient_fails():
    cand = _grads(2, 0.1, _grads(0))
    cand["b"] = torch.zeros_like(cand["b"])
    ok, checks, _ = _ok(cand)
    assert not ok and checks[0]["missing"] == ["b"]


def test_rescaled_gradient_fails_norm():
    cand = {n: 3 * t for n, t in _grads(2, 0.1, _grads(0)).items()}
    ok, checks, _ = _ok(cand)
    assert not ok and not checks[2]["ok"] and checks[1]["ok"]


def test_much_noisier_than_stock_fails_cosine():
    ref = _grads(0)
    cand = _grads(2, 0.1, ref)
    cand["c"] = _grads(3, 3.0, ref)["c"]
    ok, checks, _ = _ok(cand)
    assert not ok and checks[1]["worse"] == ["c"]


def test_canonical_names():
    assert (
        canonical("thinker._fsdp_wrapped_module.model.layers.0._checkpoint_wrapped_module.mlp.gate.weight")
        == "model.layers.0.mlp.gate.weight"
    )
    assert canonical("model.norm.weight") == "model.norm.weight"
    assert canonical("base_model.model.llm.model.layers.0.self_attn.q_proj.lora_B.default.weight") == (
        "model.layers.0.self_attn.q_proj.lora_B.default.weight"
    )


def test_packed_positions_mrope_and_1d():
    def fields(pos0, pos1):
        return {
            "input_ids": [torch.arange(5), torch.arange(4)],
            "position_ids": [pos0, pos1],
            "response_mask": [torch.ones(2), torch.ones(1)],
        }

    ids, pos, weights = packed_inputs(fields(torch.arange(5), torch.arange(4)), device="cpu")
    assert ids.shape == (1, 9) and pos.shape == (1, 9) and pos[0, 5] == 0
    _, pos3, weights3 = packed_inputs(fields(torch.arange(5).expand(4, 5), torch.arange(4).expand(4, 4)), device="cpu")
    assert pos3.shape == (4, 1, 9) and torch.equal(weights, weights3)
    # Row 0: prompt 3, response tokens 3..4 -> labels 2..3 (+1); row 1: prompt 3, response token 8 -> label 7 (-1).
    assert weights.tolist() == [0, 0, 1, 1, 0, 0, 0, -1]


def test_lora_projection_matches_autograd():
    torch.manual_seed(0)
    w = torch.randn(16, 8, dtype=torch.float64)
    a = torch.randn(4, 8, dtype=torch.float64)
    b = torch.zeros(16, 4, dtype=torch.float64, requires_grad=True)
    x = torch.randn(5, 8, dtype=torch.float64)
    target = torch.randn(5, 16, dtype=torch.float64)
    s = 2.0
    loss = ((x @ (w + s * b @ a).T - target) ** 2).sum()
    loss.backward()
    w_full = w.clone().requires_grad_()
    ((x @ w_full.T - target) ** 2).sum().backward()
    name = "model.layers.0.self_attn.q_proj.lora_B.default.weight"
    assert lora_base_name(name) == "model.layers.0.self_attn.q_proj.weight"
    meta = {name: {"base": lora_base_name(name), "A": a, "scaling": s}}
    expected = expected_lora_grads({"model.layers.0.self_attn.q_proj.weight": w_full.grad}, meta)[name]
    assert torch.allclose(expected, b.grad)
    assert canonical("base_model.model.thinker.model.layers.0.self_attn.q_proj.lora_B.default.weight") == name


def test_audit_flags_tool_calls_only(tmp_path):
    log = tmp_path / "agent.log"

    def call(kind, args, subtype="started", **extra):
        return {"type": "tool_call", "subtype": subtype, "tool_call": {kind: {"args": args, **extra}}}

    events = [
        call("shellToolCall", {"command": "cat /home/public/yx/rsi_work/dev/x.py"}),
        call("webSearchToolCall", {"query": "verl-omni#637"}),
        call(
            "readToolCall",
            {"path": "/ws/a.py"},
            subtype="completed",
            result={"content": "https://example.com rsi_work/runs"},
        ),
        call("editToolCall", {"path": "/ws/b.py", "streamContent": "# see https://example.com"}),
        {"type": "assistant", "message": "see https://example.com and rsi_work/dev"},
    ]
    log.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    rules = [("human_solution", r"verl-omni#637"), ("hidden_path", r"rsi_work/(dev|scratch|runs)")]
    hits = {(h["rule"], h["match"]) for h in audit_log(str(log), rules)}
    assert ("hidden_path", "rsi_work/dev") in hits
    assert ("human_solution", "verl-omni#637") in hits
    assert not any(m == "https://example.com" for _, m in hits)
    assert ("hidden_path", "rsi_work/runs") not in hits
