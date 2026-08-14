#!/usr/bin/env bash
set -euo pipefail

VFI_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VFI_PYTHON="/home/zhenying/qhong/envs/anaconda3/envs/vfi/bin/python"
VFI_CHECKPOINT="ckpt/0807_s2v3_official_tuesday/0807_s2v3_official_tuesday_320.pkl"
VFI_MATRIX_DIR="${VFI_PROJECT_DIR}/record/0814_teacher_flyingthings_ablation"
VFI_MATRIX_LOG="${VFI_MATRIX_DIR}/matrix.log"
VFI_MATRIX_PID_FILE="${VFI_MATRIX_DIR}/matrix.pid"

cd "${VFI_PROJECT_DIR}"
mkdir -p "${VFI_MATRIX_DIR}"

if [[ "${1:-}" != "--worker" ]]; then
  if [[ -f "${VFI_MATRIX_PID_FILE}" ]]; then
    VFI_OLD_PID="$(<"${VFI_MATRIX_PID_FILE}")"
    if [[ -n "${VFI_OLD_PID}" ]] && kill -0 "${VFI_OLD_PID}" 2>/dev/null; then
      echo "FlyingThings 四组实验已经在运行，PID=${VFI_OLD_PID}"
      exit 1
    fi
  fi
  if pgrep -f 'train.py --config train_config_teacher_flyingthings_' >/dev/null; then
    echo "检测到单独启动的 FlyingThings 训练；队列会等待它完成后接着运行。"
  fi
  nohup "${BASH_SOURCE[0]}" --worker >"${VFI_MATRIX_LOG}" 2>&1 &
  VFI_MATRIX_PID=$!
  echo "${VFI_MATRIX_PID}" >"${VFI_MATRIX_PID_FILE}"
  echo "四组 FlyingThings 对照已串行启动，PID=${VFI_MATRIX_PID}"
  echo "总日志: ${VFI_MATRIX_LOG}"
  echo "查看: tail -f ${VFI_MATRIX_LOG}"
  exit 0
fi

# A variant may have been started manually before this matrix launcher.  Wait
# for its parent process and DataLoader workers to leave, then the checkpoint
# check below skips the completed variant instead of training it twice.
if pgrep -f 'train.py --config train_config_teacher_flyingthings_' >/dev/null; then
  echo "[$(date '+%F %T %Z')] 等待当前单独启动的 FlyingThings 训练完成"
  while pgrep -f 'train.py --config train_config_teacher_flyingthings_' \
      >/dev/null; do
    sleep 30
  done
  echo "[$(date '+%F %T %Z')] 当前训练已结束，继续四组队列"
fi

VFI_CONFIGS=(
  train_config_teacher_flyingthings_cycle.yaml
  train_config_teacher_flyingthings_noflow.yaml
  train_config_teacher_flyingthings_raw.yaml
  train_config_teacher_flyingthings_cycle_ifblock.yaml
)
VFI_EXPS=(
  0814_teacher_flyingthings_cycle
  0814_teacher_flyingthings_noflow
  0814_teacher_flyingthings_raw
  0814_teacher_flyingthings_cycle_ifblock
)

echo "[$(date '+%F %T %Z')] 开始四组 FlyingThings 对照"
for VFI_INDEX in "${!VFI_CONFIGS[@]}"; do
  VFI_NUMBER=$((VFI_INDEX + 1))
  VFI_CONFIG="${VFI_CONFIGS[${VFI_INDEX}]}"
  VFI_EXP="${VFI_EXPS[${VFI_INDEX}]}"
  VFI_FINAL_CKPT="ckpt/${VFI_EXP}/${VFI_EXP}_150.pkl"
  VFI_LOG_DIR="record/${VFI_EXP}"
  VFI_LOG="${VFI_LOG_DIR}/train.log"

  if [[ -f "${VFI_FINAL_CKPT}" ]]; then
    echo "[$(date '+%F %T %Z')] 跳过已完成实验 ${VFI_EXP}"
    continue
  fi

  mkdir -p "${VFI_LOG_DIR}"
  echo "[$(date '+%F %T %Z')] [${VFI_NUMBER}/4] 启动 ${VFI_EXP}"
  "${VFI_PYTHON}" -u train.py \
    --config "${VFI_CONFIG}" \
    --restore_ckpt "${VFI_CHECKPOINT}" \
    2>&1 | tee "${VFI_LOG}"
  echo "[$(date '+%F %T %Z')] 完成 ${VFI_EXP}"
done
echo "[$(date '+%F %T %Z')] 四组 FlyingThings 对照全部完成"
