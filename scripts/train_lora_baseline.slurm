#!/usr/bin/env bash
# Submit from the project root with:
#   mkdir -p logs && sbatch scripts/train_lora_baseline.slurm
#
# This is the non-private baseline only. Do not use it for DP experiments.
#SBATCH --job-name=distilgpt2-lora
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/lora-baseline-%j.out
#SBATCH --error=logs/lora-baseline-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("SLURM allocated no CUDA-visible GPU; do not run the baseline.")
print("GPU:", torch.cuda.get_device_name(0))
PY

python -m src.train \
  --max-train-samples 0 \
  --max-train-steps 0 \
  --epochs 3 \
  --batch-size 4 \
  --gradient-accumulation-steps 1 \
  --max-seq-len 128 \
  --output-dir "artifacts/lora_baseline_seed42_job${SLURM_JOB_ID}" \
  --result-file "results/lora_baseline_seed42_job${SLURM_JOB_ID}.json"
