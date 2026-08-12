#!/usr/bin/env bash
# DROID training launcher for cml18 (2x RTX 4090, 8 CPUs, GCS-streamed data).
#
# One-time setup (see experiments/droid/README.md):
#   assets under $DROID_ROOT: dinov3-vitb16/, bert-base-uncased/,
#   groundingdino_swint_ogc.pth, droid_sample_ranges_v1_0_1.json
#   stats: python scripts/droid/compute_droid_stats.py --filter_ranges_path ...
#
# Usage:
#   scripts/droid/train_cml18.sh smoke   # 2k-step smoke run
#   scripts/droid/train_cml18.sh full    # 100k-step run

set -euo pipefail

MODE="${1:-smoke}"
DROID_ROOT="${DROID_ROOT:-/tmp2/chungyili/droid}"
VENV="${VENV:-/tmp2/chungyili/turbovla-droid-venv}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HF_HOME="${HF_HOME:-/tmp2/chungyili/hf_home}"
export WANDB_PROJECT="${WANDB_PROJECT:-turbovla-droid}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Keep TF quiet and CPU-only; thread caps are set inside the dataset.
export TF_CPP_MIN_LOG_LEVEL=1
# NCCL 2.28's RoCE/IB path segfaults on cml18 (driver 535); single-node
# training only needs SHM/P2P anyway.
export NCCL_IB_DISABLE=1
# GCS reads default to 64MB blocks with no request timeout; campus network
# stalls mid-transfer, so keep requests small and time them out fast. The
# loader's own watchdog rebuilds the pipeline if reads stall anyway.
export GCS_READ_CACHE_BLOCK_SIZE_MB=16
export GCS_READ_CACHE_MAX_SIZE_MB=128
export GCS_READ_REQUEST_TIMEOUT_SECS=120
export GCS_REQUEST_CONNECTION_TIMEOUT_SECS=30
export GCS_METADATA_REQUEST_TIMEOUT_SECS=60

case "$MODE" in
  smoke)
    MAX_STEPS=2000
    WARMUP_STEPS=200
    RUN_NAME="droid-smoke-$(date +%Y%m%d-%H%M)"
    CKPT_DIR="$DROID_ROOT/outputs/smoke"
    ;;
  full)
    MAX_STEPS=100000
    WARMUP_STEPS=1000
    RUN_NAME="${RUN_NAME:-droid-full}"
    CKPT_DIR="$DROID_ROOT/outputs/full"
    # Stable id so supervisor relaunches resume the same wandb run.
    export WANDB_RUN_ID="${WANDB_RUN_ID:-droid-full-cml18}"
    export WANDB_RESUME=allow
    ;;
  *)
    echo "unknown mode: $MODE (expected smoke|full)" >&2
    exit 1
    ;;
esac

mkdir -p "$CKPT_DIR"

exec "$VENV/bin/torchrun" --nproc_per_node=2 --master_port="${MASTER_PORT:-29517}" \
  "$REPO_ROOT/experiments/droid/train.py" \
  --dataset_dir "gs://gresearch/robotics/droid/1.0.1" \
  --stats_path "$REPO_ROOT/experiments/droid/configs/droid_stats.json" \
  --stats_key droid \
  --filter_ranges_path "$DROID_ROOT/droid_sample_ranges_v1_0_1.json" \
  --dinov3_path "$DROID_ROOT/dinov3-vitb16" \
  --bert_path "$DROID_ROOT/bert-base-uncased" \
  --pretrained_init_ckpt "$DROID_ROOT/groundingdino_swint_ogc.pth" \
  --checkpoint_dir "$CKPT_DIR" \
  --checkpoint_prefix turbovla_droid \
  --resume_mode all \
  --max_steps "$MAX_STEPS" \
  --warmup_steps "$WARMUP_STEPS" \
  --wandb_run_name "$RUN_NAME" \
  "${@:2}"
