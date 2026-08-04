#!/usr/bin/env bash
# Run this ONCE from a compute node while its known-working Conda environment
# is activated. The resulting archive is stored in $HOME for future Slurm jobs.

set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate the working Conda environment first (for example /tmp/.../dp-env)." >&2
  exit 2
fi
if [[ "$CONDA_PREFIX" != /tmp/* ]]; then
  echo "Refusing to pack $CONDA_PREFIX: use the working local compute-node environment under /tmp." >&2
  exit 2
fi

archive="${SNORLAX_ENV_ARCHIVE:-$HOME/dp-lora-env.tar.gz}"
python -c "import dp_transformers, numpy, opacus, torch; assert torch.cuda.is_available(); print('Environment checks passed')"
python -m pip install conda-pack
conda-pack -p "$CONDA_PREFIX" -o "$archive" --force
echo "Packed environment: $archive"
