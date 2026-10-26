#!/usr/bin/env bash
# Gate g2: real verl training steps (FSDP2 actor + vLLM-Omni rollout) on the
# truncated Qwen3-Omni thinker, with evaluator capture hooks enabled.
# Overrides come from rsi/evaluate/loop_config.py.
#
# Required env:
#   RSI_WORKTREE   root holding verl-omni/, verl/, vllm-omni/ checkouts
#   RSI_HARNESS    directory containing the trusted `rsi` package
#   RSI_DUMP_DIR   where hooks write batch_step{N}.pt and weight digests
#   RSI_CKPT       truncated checkpoint
#   RSI_PYTHON     interpreter of the shared venv
#   CUDA_VISIBLE_DEVICES  two GPUs
# Optional env:
#   RSI_DATA_DIR   directory with gsm8k train.parquet / test.parquet
#   RSI_STEPS      training steps (default 2)
#   RSI_HOOKS      0 to run without the capture hooks (bench_loop; default 1)
# Extra arguments are appended as Hydra overrides.
set -euo pipefail

: "${RSI_WORKTREE:?}" "${RSI_HARNESS:?}" "${RSI_DUMP_DIR:?}" "${RSI_CKPT:?}" "${RSI_PYTHON:?}" "${CUDA_VISIBLE_DEVICES:?}"

VENV_SITE="$(dirname "$(dirname "${RSI_PYTHON}")")/lib/python3.12/site-packages"
for lib_dir in "${VENV_SITE}/nvidia"/*/lib; do
    [[ -d "${lib_dir}" ]] && export LD_LIBRARY_PATH="${lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
done

export PYTHONPATH="${RSI_WORKTREE}/verl-omni:${RSI_WORKTREE}/verl:${RSI_WORKTREE}/vllm-omni:${RSI_HARNESS}:${RSI_HARNESS}/rsi/isolation"
# Unique weight-sync socket names per loop; see rsi/isolation/sitecustomize.py.
export RSI_RAY_JOB_SUFFIX="${RSI_RAY_JOB_SUFFIX:-$(cut -c1-12 /proc/sys/kernel/random/uuid)}"
if [[ "${RSI_HOOKS:-1}" == 1 ]]; then
    export VERL_USE_EXTERNAL_MODULES=verl_omni,rsi.evaluate.hooks
else
    export VERL_USE_EXTERNAL_MODULES=verl_omni
fi
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
mkdir -p "${RSI_DUMP_DIR}"

mapfile -t OVERRIDES < <(PYTHONPATH="${RSI_HARNESS}" "${RSI_PYTHON}" -m rsi.evaluate.loop_config)

cd "${RSI_WORKTREE}/verl-omni"
exec "${RSI_PYTHON}" -m verl_omni.trainer.main_omni "${OVERRIDES[@]}" "$@"
