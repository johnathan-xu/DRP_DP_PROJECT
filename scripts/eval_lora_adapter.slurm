#!/usr/bin/env bash
# Submit from the project root with:
#   mkdir -p logs
#   sbatch scripts/eval_lora_adapter.slurm artifacts/<adapter-directory> [method] [epsilon] [training-json]

#SBATCH --job-name=distilgpt2-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/lora-eval-%j.out
#SBATCH --error=logs/lora-eval-%j.err

set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 4 ]; then
  echo "Usage: sbatch scripts/eval_lora_adapter.slurm artifacts/<adapter-directory> [method] [epsilon] [training-json]" >&2
  exit 2
fi

cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
adapter_dir="$1"
method="${2:-lora}"
epsilon="${3:-}"
training_result="${4:-}"
private_args=()
if [[ "$method" == dp-* ]]; then
  private_args+=(--private)
fi
if [[ -n "$epsilon" ]]; then
  private_args+=(--target-epsilon "$epsilon")
fi
if [[ -n "$training_result" ]]; then
  private_args+=(--training-result "$training_result")
fi

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("SLURM allocated no CUDA-visible GPU; cannot run evaluation.")
print("GPU:", torch.cuda.get_device_name(0))
PY

python -m src.eval \
  --adapter-dir "$adapter_dir" \
  --method "$method" \
  "${private_args[@]}" \
  --max-examples 0 \
  --max-new-tokens 64 \
  --output-file "results/predictions_job${SLURM_JOB_ID}.jsonl" \
  --result-file "results/evaluation_job${SLURM_JOB_ID}.json"
