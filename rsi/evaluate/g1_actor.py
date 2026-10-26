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
"""Actor side of gate g1: replay a captured loop batch through the real actor.

Builds ``ActorRolloutRefWorker(role="actor")`` in-process (world size 1) from
the same Hydra config as g2 and calls ``compute_log_prob`` on the fixture rows
in several groupings. Optionally records per-op outputs for one row.

Usage:
    python -m rsi.evaluate.g1_actor --worktree W --harness H --fixture F --out O.pt [--probe-row 0] [--state S.pt]
"""

import argparse
import json
import os
import socket


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


_OUTPUT_FIELDS = {"entropy", "log_probs", "old_log_probs", "rollout_log_probs", "rm_scores", "dummy_tensor"}


def build_td(fields, rows):
    import torch
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu

    data = {}
    for key, value in fields.items():
        if key in _OUTPUT_FIELDS:
            continue
        items = [value[i] for i in rows]
        if items[0].dim() == 2:
            # [C, L] rows (mrope position ids): ragged on the sequence axis, as
            # the engine assumes (``maybe_fix_3d_position_ids``) even when all
            # rows have equal length.
            lengths = torch.tensor([0] + [t.shape[-1] for t in items])
            data[key] = torch.nested.nested_tensor_from_jagged(
                torch.cat(items, dim=-1), offsets=lengths.cumsum(0), jagged_dim=2
            )
        else:
            data[key] = torch.nested.as_nested_tensor(items, layout=torch.jagged)
    td = TensorDict(data, batch_size=[len(rows)])
    seq_lens = [int(fields["input_ids"][i].shape[-1]) for i in rows]
    tu.assign_non_tensor(
        td,
        calculate_entropy=True,
        compute_loss=False,
        temperature=1.0,
        global_token_num=seq_lens,
    )
    return td


def replay(worker, fields, groups):
    """Return per-row response log-probs, computing each group as one batch."""
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.padding import response_from_nested

    out = {}
    for rows in groups:
        td = build_td(fields, rows)
        result = worker.compute_log_prob(td)
        log_probs = tu.get(result, "log_probs")
        resp = response_from_nested(log_probs, td["response_mask"])
        resp_rows = list(resp.unbind(0)) if not resp.is_nested else list(resp.unbind())
        for i, r in zip(rows, resp_rows, strict=True):
            out[i] = r.detach().cpu()
    return [out[i] for i in sorted(out)]


def build_worker(worktree, harness, ckpt, lora=None, family="qwen3_omni"):
    """The candidate's ``ActorRolloutRefWorker(role="actor")`` from the g2 Hydra config.

    World size 1 unless launched by torchrun (``bench_actor`` on several GPUs).
    """
    if "TORCHELASTIC_RUN_ID" not in os.environ:
        os.environ.update(
            RANK="0", WORLD_SIZE="1", LOCAL_RANK="0", MASTER_ADDR="127.0.0.1", MASTER_PORT=str(_free_port())
        )
    else:
        import torch

        # Ray gives each worker one visible GPU; torchrun ranks share them.
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    n_gpus = int(os.environ["WORLD_SIZE"])
    from hydra import compose, initialize_config_dir

    from rsi.evaluate.loop_config import overrides

    config_dir = os.path.join(worktree, "verl-omni", "verl_omni", "trainer", "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            "omni_trainer",
            overrides=overrides(
                ckpt,
                harness,
                "/tmp/rsi_g1_unused",
                "/home/public/yx/data/gsm8k",
                1,
                n_gpus=n_gpus,
                lora=lora,
                family=family,
            ),
        )

    from verl_omni.workers.engine_workers import ActorRolloutRefWorker

    worker = ActorRolloutRefWorker(config=cfg.actor_rollout_ref, role="actor")
    worker.init_model()
    return worker


def load_state(worker, path):
    """Overwrite trainable parameters with ``actor_state_sync{k}.pt`` from g2 (full tensors, world size 1)."""
    import torch
    from torch.distributed.tensor import DTensor, distribute_tensor

    state = torch.load(path, weights_only=False)
    params = dict(worker.actor.engine.module.named_parameters())
    missing = sorted(set(state) - set(params))
    if missing:
        raise KeyError(f"{len(missing)} captured parameters not in the actor, e.g. {missing[:3]}")
    with torch.no_grad():
        for name, full in state.items():
            p = params[name]
            if isinstance(p, DTensor):
                p.copy_(distribute_tensor(full.to(p.device, p.dtype), p.device_mesh, p.placements))
            else:
                p.copy_(full.to(p.device, p.dtype))
    return len(state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--harness", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--loop-chunk", type=int, default=4, help="rows per actor GPU in the loop")
    parser.add_argument("--probe-row", type=int, default=None)
    parser.add_argument("--lora", default="false", help="scope.lora as JSON")
    parser.add_argument("--family", default="qwen3_omni")
    parser.add_argument("--state", default=None, help="actor_state_sync{k}.pt from g2, loaded after init")
    args = parser.parse_args()

    import torch

    worker = build_worker(args.worktree, args.harness, args.ckpt, lora=json.loads(args.lora), family=args.family)
    if args.state:
        print(f"loaded {load_state(worker, args.state)} parameters from {args.state}")
    fields = torch.load(args.fixture, weights_only=False)["fields"]
    n = len(fields["input_ids"])
    chunk = args.loop_chunk
    result = {
        "loop_grouping": replay(worker, fields, [list(range(i, min(i + chunk, n))) for i in range(0, n, chunk)]),
        "alone": replay(worker, fields, [[i] for i in range(n)]),
        "all_packed": replay(worker, fields, [list(range(n))]),
    }

    if args.probe_row is not None:
        from rsi.evaluate.probe import ActorProbe

        module = worker.actor.engine.module
        with ActorProbe(module) as probe:
            replay(worker, fields, [[args.probe_row]])
        result["probe"] = probe.records

    torch.save(result, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
