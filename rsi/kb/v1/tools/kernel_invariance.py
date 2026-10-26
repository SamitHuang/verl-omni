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
# ruff: noqa: E402, E741
# A script: VLLM_BATCH_INVARIANT must be set before vLLM is imported; E and I are the MoE shapes.
"""Microtests: are the vLLM batch-invariant kernels invariant across the shapes actor and rollout use?

Covers FA3 packed causal prefill vs paged one-query decode vs chunked prefill,
the batch-invariant matmul over M, fused vs separate QKV, both RMSNorm paths,
fused top-k and fused MoE experts over M, and log-softmax. Every line should
print 0 mismatches. Extend it with any new op before relying on it.

    PYTHONPATH=verl-omni:verl:vllm-omni python rsi_tools/kernel_invariance.py
"""

import os

os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
import torch
from vllm.model_executor.determinism import batch_invariant as bi

bi.init_batch_invariance()
from vllm.vllm_flash_attn import flash_attn_varlen_func

torch.manual_seed(0)
dev = "cuda"
bf = torch.bfloat16


def neq(a, b):
    return int((a != b).sum().item())


# ---- attention: packed causal prefill vs paged decode, one query at a time
H, KVH, D = 32, 4, 128
lens = [300, 517, 129]
total = sum(lens)
q = torch.randn(total, H, D, device=dev, dtype=bf)
k = torch.randn(total, KVH, D, device=dev, dtype=bf)
v = torch.randn(total, KVH, D, device=dev, dtype=bf)
cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), device=dev, dtype=torch.int32)
scale = D**-0.5
for ver in (3,):
    pre = flash_attn_varlen_func(
        q,
        k,
        v,
        max(lens),
        cu,
        max(lens),
        cu_seqlens_k=cu,
        softmax_scale=scale,
        causal=True,
        num_splits=1,
        fa_version=ver,
    )
    # paged cache, block 16
    B = 16
    nblocks = sum((L + B - 1) // B for L in lens) + 4
    kc = torch.zeros(nblocks, B, KVH, D, device=dev, dtype=bf)
    vc = torch.zeros_like(kc)
    tables, nb = [], 0
    for i, L in enumerate(lens):
        n = (L + B - 1) // B
        blocks = list(range(nb, nb + n))[::-1]  # non-contiguous order
        nb += n
        tables.append(blocks)
        s = int(cu[i])
        for j in range(L):
            kc[blocks[j // B], j % B] = k[s + j]
            vc[blocks[j // B], j % B] = v[s + j]
    maxb = max(len(t) for t in tables)
    bt = torch.tensor([t + [0] * (maxb - len(t)) for t in tables], device=dev, dtype=torch.int32)
    # decode: for each position t of every seq, query row t with seqused_k = t+1, batched across seqs
    dec = torch.empty_like(pre)
    for t in range(max(lens)):
        rows = [i for i, L in enumerate(lens) if t < L]
        qq = torch.stack([q[int(cu[i]) + t] for i in rows])
        cuq = torch.arange(len(rows) + 1, device=dev, dtype=torch.int32)
        used = torch.tensor([t + 1] * len(rows), device=dev, dtype=torch.int32)
        out = flash_attn_varlen_func(
            qq,
            kc,
            vc,
            1,
            cuq,
            t + 1,
            seqused_k=used,
            block_table=bt[rows],
            softmax_scale=scale,
            causal=True,
            num_splits=1,
            fa_version=ver,
        )
        for r, i in enumerate(rows):
            dec[int(cu[i]) + t] = out[r]
    # chunked prefill: prompt processed in two chunks through the paged cache
    ch = torch.empty_like(pre)
    for i, L in enumerate(lens):
        s = int(cu[i])
        cut = L // 2 + 7
        for a, b in ((0, cut), (cut, L)):
            qq = q[s + a : s + b]
            cuq = torch.tensor([0, b - a], device=dev, dtype=torch.int32)
            used = torch.tensor([b], device=dev, dtype=torch.int32)
            ch[s + a : s + b] = flash_attn_varlen_func(
                qq,
                kc,
                vc,
                b - a,
                cuq,
                b,
                seqused_k=used,
                block_table=bt[i : i + 1],
                softmax_scale=scale,
                causal=True,
                num_splits=1,
                fa_version=ver,
            )
    print(
        f"FA{ver}: prefill vs paged-decode mismatches {neq(pre, dec)} / {pre.numel()}, "
        f"prefill vs chunked {neq(pre, ch)}"
    )

# ---- linear: batch invariance and fused qkv vs separate
K = 2048
W = torch.randn(5120, K, device=dev, dtype=bf) * 0.02
x = torch.randn(1500, K, device=dev, dtype=bf)
full = bi.linear_batch_invariant(x, W)
for m in (1, 3, 8, 64, 700):
    print(f"linear M={m}: {neq(bi.linear_batch_invariant(x[:m], W), full[:m])}")
sep = torch.cat(
    [
        bi.linear_batch_invariant(x, W[:4096]),
        bi.linear_batch_invariant(x, W[4096:4608]),
        bi.linear_batch_invariant(x, W[4608:]),
    ],
    -1,
)
print("fused qkv vs separate:", neq(full, sep))
import torch.nn.functional as F

print("cuBLAS F.linear M=8 vs M=1500 rows:", neq(F.linear(x[:8], W), F.linear(x, W)[:8]))

# ---- rmsnorm: triton (no residual) and fused add
import vllm._custom_ops as ops

w = torch.randn(K, device=dev, dtype=bf)
h = torch.randn(1500, K, device=dev, dtype=bf)
r = torch.randn(1500, K, device=dev, dtype=bf)
ref = bi.rms_norm_batch_invariant(h, w, 1e-6)
print("triton rmsnorm M=8 vs full:", neq(bi.rms_norm_batch_invariant(h[:8], w, 1e-6), ref[:8]))
a1, b1 = h.clone(), r.clone()
ops.fused_add_rms_norm(a1, b1, w, 1e-6)
for m in (1, 8, 255, 256, 700):
    a2, b2 = h[:m].clone(), r[:m].clone()
    ops.fused_add_rms_norm(a2, b2, w, 1e-6)
    print(f"fused_add_rms_norm M={m}: out {neq(a2, a1[:m])} resid {neq(b2, b1[:m])}")
print("fused residual == bf16 add:", neq(b1, h + r))

# ---- moe
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

E, I = 128, 768
w13 = torch.randn(E, 2 * I, K, device=dev, dtype=bf) * 0.02
w2 = torch.randn(E, K, I, device=dev, dtype=bf) * 0.02
gw = torch.randn(E, K, device=dev, dtype=bf) * 0.02
logits = bi.linear_batch_invariant(h, gw)
tw, ti, _ = fused_topk(h, logits, 8, True)
ref = fused_experts(h, w13, w2, tw, ti)
for m in (1, 8, 64, 700):
    tw2, ti2, _ = fused_topk(h[:m], logits[:m], 8, True)
    out = fused_experts(h[:m], w13, w2, tw2, ti2)
    print(f"moe M={m}: topk w {neq(tw2, tw[:m])} ids {neq(ti2, ti[:m])} out {neq(out, ref[:m])}")

# ---- log_softmax
lg = torch.randn(1500, 151936, device=dev, dtype=torch.float32)
ls = torch.log_softmax(lg, -1)
print(
    "log_softmax M=8 vs full:",
    neq(torch.log_softmax(lg[:8], -1), ls[:8]),
    "| bi.log_softmax == torch (overridden):",
    neq(bi.log_softmax(lg[:8], -1), ls[:8]),
)
