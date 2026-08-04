#!/usr/bin/env bash
# Submit one private LoRA run from the project root:
#   mkdir -p logs
#   sbatch scripts/train_dp_lora_epsilon.slurm 4

#SBATCH --job-name=distilgpt2-dp-lora
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dp-lora-%j.out
#SBATCH --error=logs/dp-lora-%j.err

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: sbatch scripts/train_dp_lora_epsilon.slurm <epsilon>" >&2
  exit 2
fi

epsilon="$1"
cd "${SLURM_SUBMIT_DIR}"
source scripts/activate_snorlax_env.sh

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("SLURM allocated no CUDA-visible GPU; cannot run DP-LoRA.")
print("GPU:", torch.cuda.get_device_name(0))
PY

python -m src.train \
  --dp \
  --target-epsilon "$epsilon" \
  --target-delta 1e-5 \
  --max-grad-norm 1.0 \
  --accountant prv \
  --max-train-samples 0 \
  --max-train-steps 0 \
  --epochs 3 \
  --batch-size 4 \
  --gradient-accumulation-steps 1 \
  --max-seq-len 128 \
  --learning-rate 5e-4 \
  --seed 42 \
  --output-dir "artifacts/dp_lora_eps${epsilon}_seed42_job${SLURM_JOB_ID}" \
  --result-file "results/dp_lora_eps${epsilon}_seed42_job${SLURM_JOB_ID}.json"
