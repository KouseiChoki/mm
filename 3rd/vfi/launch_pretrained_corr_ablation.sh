#!/usr/bin/env bash
set -euo pipefail

VFI_DIR="/home/zhenying/qhong/repo/mm/3rd/vfi"
PYTHON_BIN="/home/zhenying/qhong/envs/anaconda3/envs/vfi/bin/python"
CONFIG="train_config_0821_lc_pretrained_corr_weekend.yaml"
BASE_CKPT="ckpt/0729_lc_v3s2/0729_lc_v3s2_800.pkl"

cd "$VFI_DIR"

run_arm() {
    local stage="$1"
    local tag="$2"
    echo "════════ correspondence ablation: $stage ($tag) ════════"
    "$PYTHON_BIN" train.py \
        --config "$CONFIG" \
        --restore_ckpt "$BASE_CKPT" \
        --smoke_epochs 4 \
        --smoke_steps 401 \
        --smoke_tag "$tag" \
        --corr_stages "$stage"
}

# Same seed, data order, LR curve, initialization and validation set.  Only
# the cost-volume injection level changes between these three arms.
run_arm "1/16" "ablate_corr_16"
run_arm "1/8"  "ablate_corr_8"
run_arm "both" "ablate_corr_both"

echo "════════ all three correspondence ablations completed ════════"
