#!/usr/bin/env bash
# Submit from the project root:
#   mkdir -p logs
#   sbatch scripts/train_dp_lora_subset.slurm last-block 4
#   sbatch scripts/train_dp_lora_subset.slurm lm-head 4

#SBATCH --job-name=distilgpt2-dp-subset
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dp-subset-%j.out
#SBATCH --error=logs/dp-subset-%j.err

set -euo pipefail

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  echo "Usage: sbatch scripts/train_dp_lora_subset.slurm <last-block|lm-head> [epsilon=4]" >&2
  exit 2
fi

scope="$1"
epsilon="${2:-4}"
if [[ "$scope" != "last-block" && "$scope" != "lm-head" ]]; then
  echo "Subset must be last-block or lm-head." >&2
  exit 2
fi

cd "${SLURM_SUBMIT_DIR}"
source scripts/activate_snorlax_env.sh

python -m src.train \
  --dp --target-epsilon "$epsilon" --target-delta 1e-5 --max-grad-norm 1.0 --accountant prv \
  --lora-scope "$scope" \
  --max-train-samples 0 --max-train-steps 0 --epochs 3 --batch-size 4 \
  --gradient-accumulation-steps 1 --max-seq-len 128 --learning-rate 5e-4 --seed 42 \
  --output-dir "artifacts/dp_lora_${scope}_eps${epsilon}_seed42_job${SLURM_JOB_ID}" \
  --result-file "results/dp_lora_${scope}_eps${epsilon}_seed42_job${SLURM_JOB_ID}.json"
