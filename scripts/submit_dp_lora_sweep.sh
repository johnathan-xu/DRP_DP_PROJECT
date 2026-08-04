#!/usr/bin/env bash
# Submit the six Experiment-1 DP-LoRA runs from the project root.
set -euo pipefail

mkdir -p logs
for epsilon in 0.5 1 2 4 8 16; do
  sbatch scripts/train_dp_lora_epsilon.slurm "$epsilon"
done
