#!/usr/bin/env bash
# Source this *inside a Slurm job* after cd "$SLURM_SUBMIT_DIR".
# The archive is made once from a working compute-node Conda environment by
# scripts/pack_snorlax_env.sh, then unpacked to this job's local disk.

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "activate_snorlax_env.sh must run inside a Slurm allocation/job." >&2
  return 2
fi

archive="${SNORLAX_ENV_ARCHIVE:-$HOME/dp-lora-env.tar.gz}"
job_tmp="${SLURM_TMPDIR:-/tmp/${USER}-slurm-${SLURM_JOB_ID}}"
env_dir="${job_tmp}/dp-lora-env"

if [[ ! -f "$archive" ]]; then
  echo "Missing $archive. In your working compute-node environment, run:" >&2
  echo "  bash scripts/pack_snorlax_env.sh" >&2
  return 2
fi

if [[ ! -x "${env_dir}/bin/python" ]]; then
  mkdir -p "$env_dir"
  tar -xzf "$archive" -C "$env_dir"
  if [[ -x "${env_dir}/bin/conda-unpack" ]]; then
    "${env_dir}/bin/conda-unpack"
  fi
fi

source "${env_dir}/bin/activate"
export DP_ENV_DIR="$env_dir"
python -c "import numpy, torch; print('Python environment:', '$env_dir'); print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
