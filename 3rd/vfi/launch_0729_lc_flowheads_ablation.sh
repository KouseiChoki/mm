#!/usr/bin/env bash
set -euo pipefail

VFI_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VFI_PYTHON="/home/zhenying/qhong/envs/anaconda3/envs/vfi/bin/python"
VFI_CHECKPOINT="ckpt/0729_lc_v3s2/0729_lc_v3s2_800.pkl"
VFI_MATRIX_DIR="${VFI_PROJECT_DIR}/record/0817_0729_lc_flowheads_ablation"
VFI_MATRIX_LOG="${VFI_MATRIX_DIR}/matrix.log"
VFI_PID_FILE="${VFI_MATRIX_DIR}/matrix.pid"

cd "${VFI_PROJECT_DIR}"
mkdir -p "${VFI_MATRIX_DIR}"

if [[ "${1:-}" != "--worker" ]]; then
  if [[ -f "${VFI_PID_FILE}" ]]; then
    VFI_OLD_PID="$(<"${VFI_PID_FILE}")"
    if [[ -n "${VFI_OLD_PID}" ]] && kill -0 "${VFI_OLD_PID}" 2>/dev/null; then
      echo "0729 LC flow-head A/B 已在运行，PID=${VFI_OLD_PID}"
      exit 1
    fi
  fi
  if pgrep -f 'train.py --config train_config_0729_lc_flowheads_' >/dev/null; then
    echo "检测到单独启动的 0729 flow-head 实验，请勿重复运行队列。"
    exit 1
  fi
  nohup "${BASH_SOURCE[0]}" --worker >"${VFI_MATRIX_LOG}" 2>&1 &
  VFI_PID=$!
  echo "${VFI_PID}" >"${VFI_PID_FILE}"
  echo "0729 LC flow-head A/B 已串行启动，PID=${VFI_PID}"
  echo "查看: tail -f ${VFI_MATRIX_LOG}"
  exit 0
fi

# Run the primary cycle experiment first. Both variants restore the same 0729
# checkpoint independently; this is a fine-tune comparison, not a resume chain.
VFI_CONFIGS=(
  train_config_0729_lc_flowheads_cycle.yaml
  train_config_0729_lc_flowheads_noflow.yaml
)
VFI_EXPS=(
  0817_0729_lc_flowheads_cycle
  0817_0729_lc_flowheads_noflow
)
VFI_FINAL_EPOCHS=(
  150
  30
)

echo "[$(date '+%F %T %Z')] 开始 0729 LC flow-head A/B"
for VFI_INDEX in "${!VFI_CONFIGS[@]}"; do
  VFI_CONFIG="${VFI_CONFIGS[${VFI_INDEX}]}"
  VFI_EXP="${VFI_EXPS[${VFI_INDEX}]}"
  VFI_FINAL_EPOCH="${VFI_FINAL_EPOCHS[${VFI_INDEX}]}"
  VFI_FINAL_CKPT="ckpt/${VFI_EXP}/${VFI_EXP}_${VFI_FINAL_EPOCH}.pkl"
  VFI_LOG_DIR="record/${VFI_EXP}"
  VFI_LOG="${VFI_LOG_DIR}/train.log"

  if [[ -f "${VFI_FINAL_CKPT}" ]]; then
    echo "[$(date '+%F %T %Z')] 跳过已完成实验 ${VFI_EXP}"
    continue
  fi
  mkdir -p "${VFI_LOG_DIR}"
  echo "[$(date '+%F %T %Z')] 启动 ${VFI_EXP}"
  "${VFI_PYTHON}" -u train.py \
    --config "${VFI_CONFIG}" \
    --restore_ckpt "${VFI_CHECKPOINT}" \
    2>&1 | tee "${VFI_LOG}"
  echo "[$(date '+%F %T %Z')] 完成 ${VFI_EXP}"
done
echo "[$(date '+%F %T %Z')] 0729 LC flow-head A/B 全部完成"
