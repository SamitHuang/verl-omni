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
"""Raw log-prob comparison used by every gate.

A token matches only if both log-probs are finite, share a dtype, and are
equal element-wise. No ``exp``, no tolerance.
"""

import torch


def _rows(value):
    if isinstance(value, list):
        return value
    return list(value.unbind(0))


def compare_logprobs(rollout, actor, mask) -> dict:
    """Compare per-row log-prob tensors on positions where ``mask`` is true.

    Args:
        rollout: tensor ``[B, T]`` or list of 1-D tensors.
        actor: same layout as ``rollout``.
        mask: same layout; truthy marks a valid response token.

    Returns:
        dict with ``mismatch_total``, ``tokens_compared``, ``abs_diff_max``,
        ``abs_diff_mean``, ``nonfinite``, ``dtype_rollout``, ``dtype_actor``
        and ``per_row`` mismatch counts.
    """
    rollout_rows, actor_rows, mask_rows = _rows(rollout), _rows(actor), _rows(mask)
    if not (len(rollout_rows) == len(actor_rows) == len(mask_rows)):
        raise ValueError(f"row count differs: {len(rollout_rows)} {len(actor_rows)} {len(mask_rows)}")

    dtype_rollout = {str(r.dtype) for r in rollout_rows}
    dtype_actor = {str(a.dtype) for a in actor_rows}
    mismatch = tokens = nonfinite = 0
    diff_max, diff_sum = 0.0, 0.0
    per_row = []
    for r, a, m in zip(rollout_rows, actor_rows, mask_rows, strict=True):
        n = min(r.shape[-1], a.shape[-1], m.shape[-1])
        if not (r.shape[-1] == a.shape[-1] == m.shape[-1]):
            valid_tail = m[n:].bool().any().item() if m.shape[-1] > n else False
            if valid_tail:
                raise ValueError(f"row lengths differ with valid tokens beyond {n}")
        m_ = m[:n].bool()
        r_, a_ = r[:n][m_], a[:n][m_]
        finite = torch.isfinite(r_) & torch.isfinite(a_)
        nonfinite += int((~finite).sum())
        neq = (r_ != a_) | ~finite
        if r_.dtype != a_.dtype:
            neq = torch.ones_like(neq)
        row_mismatch = int(neq.sum())
        per_row.append(row_mismatch)
        mismatch += row_mismatch
        tokens += int(m_.sum())
        if finite.any():
            d = (r_[finite].double() - a_[finite].double()).abs()
            diff_max = max(diff_max, float(d.max()))
            diff_sum += float(d.sum())
    return {
        "mismatch_total": mismatch,
        "tokens_compared": tokens,
        "abs_diff_max": diff_max,
        "abs_diff_mean": diff_sum / tokens if tokens else 0.0,
        "nonfinite": nonfinite,
        "dtype_rollout": sorted(dtype_rollout),
        "dtype_actor": sorted(dtype_actor),
        "per_row": per_row,
    }
