#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 CHECKPOINT [RESULT_NAME]" >&2
  exit 2
fi

ckpt=$1
name=${2:-$(basename "${ckpt%.*}")}
here=$(cd "$(dirname "$0")" && pwd)

cd "$here"

python benchmark_vimeo_x4k.py \
  --ckpt "$ckpt" \
  --model-name "$name" \
  --output-dir benchmark_results/explicit_matching \
  --no-tta

python eval.py \
  --ckpt "$ckpt" \
  --scene-list eval_lists/explicit_matching_real_scenes.txt \
  --max-pairs-per-scene 1 \
  --no-tta \
  --min-disk \
  --output "/home/zhenying/qhong/sync/result/${name}_explicit_matching_gate"
