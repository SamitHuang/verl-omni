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
"""Per-op backward check of verl_omni.utils.vllm_parity against fp32 torch references.

For each op: rel error of the custom bf16 backward vs fp32 autograd, next to the
rel error of plain bf16 torch autograd (calibration). A custom backward is fine
when its error is about the same as the calibration. Rope is checked with an
adjoint (dot-product) test and R^T R q == q since its backward is the inverse rotation.

Run from the workspace root on one GPU:
    PYTHONPATH=verl-omni:verl:vllm-omni python rsi_tools/backward_check.py
Adapt the op list when you add or change autograd wrappers.
"""

import os

os.environ["VLLM_BATCH_INVARIANT"] = "1"

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from vllm.model_executor.determinism.batch_invariant import init_batch_invariance  # noqa: E402

from verl_omni.utils import vllm_parity as ops  # noqa: E402

init_batch_invariance()
torch.manual_seed(0)
dev = "cuda"


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def check(name, custom, ref, inputs, out_grad_shape=None):
    def grads(fn, dtype):
        xs = [t.detach().to(dtype).requires_grad_() if t.is_floating_point() else t for t in inputs]
        out = fn(*xs)
        g = torch.ones_like(out) if out_grad_shape is None else out_grad_shape
        gen = torch.Generator(device=dev).manual_seed(1)
        g = torch.randn(out.shape, generator=gen, device=dev, dtype=torch.float32).to(out.dtype)
        gs = torch.autograd.grad(out, [x for x in xs if x.is_floating_point()], g)
        return out, gs

    o32, g32 = grads(ref, torch.float32)
    oc, gc = grads(custom, torch.bfloat16)
    ob, gb = grads(ref, torch.bfloat16)
    print(
        f"{name:22s} fwd custom={rel(oc, o32):.1e} bf16ref={rel(ob, o32):.1e} | grads custom="
        + ",".join(f"{rel(a, b):.1e}" for a, b in zip(gc, g32, strict=True))
        + " bf16ref="
        + ",".join(f"{rel(a, b):.1e}" for a, b in zip(gb, g32, strict=True))
    )


T, H, N = 1100, 2048, 4096
x = torch.randn(T, H, device=dev)
w = torch.randn(N, H, device=dev) * 0.02
check("linear", lambda x, w: ops.linear(x, w), lambda x, w: F.linear(x, w), [x, w])

nw = 1 + 0.1 * torch.randn(H, device=dev)


def rms_ref(x, w, eps=1e-6):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


for fused in (False, True):
    check(f"rms_norm fused={fused}", lambda x, w, f=fused: ops.rms_norm(x, w, 1e-6, f), rms_ref, [x * 3, nw])

# attention: two packed causal sequences, GQA 32/4, head 128
lens = [700, 400]
cu = torch.tensor([0, 700, 1100], device=dev, dtype=torch.int32)
q = torch.randn(T, 32, 128, device=dev)
k = torch.randn(T, 4, 128, device=dev)
v = torch.randn(T, 4, 128, device=dev)
scale = 128**-0.5


def attn_ref(q, k, v):
    outs, s = [], 0
    for n in lens:
        qq = q[s : s + n].transpose(0, 1)
        kk = k[s : s + n].repeat_interleave(8, dim=1).transpose(0, 1)
        vv = v[s : s + n].repeat_interleave(8, dim=1).transpose(0, 1)
        outs.append(F.scaled_dot_product_attention(qq, kk, vv, is_causal=True, scale=scale).transpose(0, 1))
        s += n
    return torch.cat(outs)


check("flash_attn", lambda q, k, v: ops.flash_attn(q, k, v, cu, 700, scale), attn_ref, [q, k, v])

# router top-k
E, K = 128, 8
logits = torch.randn(T, E, device=dev).bfloat16().float()
hid = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
_, ids_fixed = torch.softmax(logits, -1).topk(K, -1)


def topk_ref(lg):
    wts = torch.softmax(lg.float(), -1).gather(-1, ids_fixed)
    return wts / wts.sum(-1, keepdim=True)


def topk_custom(lg):
    wts, ids = ops.moe_topk(hid, lg, K, True)
    assert torch.equal(ids.long().sort(-1).values, ids_fixed.sort(-1).values), "top-k ids differ"
    order = ids.long().argsort(-1)
    back = ids_fixed.sort(-1).indices.argsort(-1)
    return wts.gather(-1, order).gather(-1, back)


check("moe_topk", topk_custom, topk_ref, [logits])

# experts
Ie = 768
xe = torch.randn(T, H, device=dev)
w13 = torch.randn(E, 2 * Ie, H, device=dev) * 0.02
w2 = torch.randn(E, H, Ie, device=dev) * 0.02
wts = torch.softmax(torch.randn(T, K, device=dev), -1)
ids = torch.stack([torch.randperm(E, device=dev)[:K] for _ in range(T)]).to(torch.int32)


def experts_loop(x, w13, w2, wt):
    out = torch.zeros_like(x)
    for e in range(E):
        tok, pos = (ids == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        gate, up = F.linear(x[tok], w13[e]).chunk(2, dim=-1)
        out.index_add_(0, tok, F.linear(F.silu(gate) * up, w2[e]) * wt[tok, pos, None].to(x.dtype))
    return out


check("moe_experts", lambda x, a, b, wt: ops.moe_experts(x, a, b, wt.float(), ids), experts_loop, [xe, w13, w2, wts])

# log-probs
V = 152064
lg = torch.randn(2048, V, device=dev) * 3
labels = torch.randint(0, V, (2048,), device=dev)
check(
    "logprobs",
    lambda lg: ops.logprobs_from_logits(lg, labels),
    lambda lg: torch.log_softmax(lg.float(), -1).gather(-1, labels[:, None]).squeeze(-1),
    [lg],
)

# mrope adjoint test: <R q, gq> + <R k, gk> == <q, R^T gq> + <k, R^T gk>
from transformers import AutoConfig  # noqa: E402

from verl_omni.pipelines.qwen3_omni.vllm_parity import _vllm_rope_parameters  # noqa: E402

cfg = AutoConfig.from_pretrained("/home/public/yx/rsi_ckpts/qwen3omni_thinker_4l").thinker_config.text_config
rope = ops.make_mrope(cfg.head_dim, cfg.max_position_embeddings, _vllm_rope_parameters(cfg), torch.bfloat16, dev)
pos = torch.stack([torch.arange(T), torch.arange(T) // 2, torch.arange(T) // 3]).to(dev)
qr = torch.randn(T, 32 * 128, device=dev, dtype=torch.bfloat16, requires_grad=True)
kr = torch.randn(T, 4 * 128, device=dev, dtype=torch.bfloat16, requires_grad=True)
q2, k2 = ops.mrope(rope, pos, qr, kr)
gq, gk = torch.randn_like(q2), torch.randn_like(k2)
dq, dk = torch.autograd.grad((q2, k2), (qr, kr), (gq, gk))
lhs = (q2.double() * gq.double()).sum() + (k2.double() * gk.double()).sum()
rhs = (qr.double() * dq.double()).sum() + (kr.double() * dk.double()).sum()
print(
    f"{'mrope adjoint':22s} <Rq,g>={lhs.item():.3f} <q,R^T g>={rhs.item():.3f} "
    f"rel={abs(lhs - rhs).item() / abs(lhs).item():.1e}"
)
norm_ok = rel(q2.float().norm(dim=-1), qr.float().norm(dim=-1))
print(f"{'mrope norm-preserving':22s} rel={norm_ok:.1e}")

# R^T R q == q (rotation): backward fed with the forward output must return the input.
dq2, dk2 = torch.autograd.grad(ops.mrope(rope, pos, qr, kr), (qr, kr), (q2.detach(), k2.detach()))
print(f"{'mrope R^T R q == q':22s} rel_q={rel(dq2, qr):.1e} rel_k={rel(dk2, kr):.1e}")
