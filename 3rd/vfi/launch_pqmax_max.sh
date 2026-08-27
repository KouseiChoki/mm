#!/usr/bin/env bash
set -euo pipefail

VFI_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VFI_PYTHON="/home/zhenying/qhong/envs/anaconda3/envs/vfi/bin/python"
VFI_CONFIG="train_config_0825_pqmax_amt_mamba_max.yaml"
VFI_BASE_CKPT="ckpt/0729_lc_v3s2/0729_lc_v3s2_800.pkl"
VFI_CORR_CKPT="ckpt/0821_flyingthings_correspondence_pretrain/best.pkl"
VFI_EXP="0825_pqmax_amt_mamba_max_b2_balanced"
VFI_RECORD_DIR="${VFI_PROJECT_DIR}/record/${VFI_EXP}"
VFI_LOG="${VFI_RECORD_DIR}/launcher.log"
VFI_PID_FILE="${VFI_RECORD_DIR}/launcher.pid"

cd "${VFI_PROJECT_DIR}"
mkdir -p "${VFI_RECORD_DIR}"

if [[ ! -x "${VFI_PYTHON}" ]]; then
  echo "Python不存在或不可执行: ${VFI_PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${VFI_CONFIG}" ]]; then
  echo "配置不存在: ${VFI_CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${VFI_BASE_CKPT}" ]]; then
  echo "0729初始化checkpoint不存在: ${VFI_BASE_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${VFI_CORR_CKPT}" ]]; then
  echo "correspondence初始化checkpoint不存在: ${VFI_CORR_CKPT}" >&2
  exit 1
fi
if [[ -f "${VFI_PID_FILE}" ]]; then
  VFI_OLD_PID="$(<"${VFI_PID_FILE}")"
  if [[ -n "${VFI_OLD_PID}" ]] && kill -0 "${VFI_OLD_PID}" 2>/dev/null; then
    echo "PQMax训练已经在运行，PID=${VFI_OLD_PID}"
    exit 1
  fi
fi

# expandable_segments avoids fragmentation when the phase switches from
# 320x576x2 to 336x608x2.  The script intentionally uses finetune semantics:
# 0729 initializes compatible tensors while all PQMax weights start new.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
echo "执行PQMax完整训练图显存preflight（两个phase尺寸）"
"${VFI_PYTHON}" smoke_pqmax.py \
  --config "${VFI_CONFIG}" \
  --restore_ckpt "${VFI_BASE_CKPT}" \
  --shape 320x576x2 \
  --shape 336x608x2 \
  --max_reserved_gib 29.5

nohup "${VFI_PYTHON}" -u train.py \
  --config "${VFI_CONFIG}" \
  --restore_ckpt "${VFI_BASE_CKPT}" \
  >"${VFI_LOG}" 2>&1 &
VFI_PID=$!
echo "${VFI_PID}" >"${VFI_PID_FILE}"

echo "PQMax最大质量长训练已启动，PID=${VFI_PID}"
echo "日志: ${VFI_LOG}"
echo "查看: tail -f ${VFI_LOG}"
