#!/usr/bin/env bash
# Bootstrap a GCP VM for chemprop HPO and CheMeleon fine-tuning.
# Use a Deep Learning VM image with a GPU (e.g. `common-cu121`, T4/L4/A100)
# so torch/CUDA are already wired up.
#
# Usage: scp this file up, or paste it after SSH-ing in, then:
#   bash gcp_setup.sh <git-remote-url>
set -euo pipefail

REPO_URL="${1:?usage: gcp_setup.sh <git-remote-url>}"

git clone "$REPO_URL" egfr-pic50-prediction
cd egfr-pic50-prediction

conda env create -f environment.yml
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate egfr-env

# Only needed for hpopt_chemprop.py — left out of environment.yml since the
# Mac track never uses it. optuna is separate from ray[tune] and is required
# by hpopt_chemprop.py's --raytune-search-algorithm optuna.
pip install -U "ray[tune]" optuna

cat <<'EOF'

Set this before running anything (wandb.ai/authorize for the key):
  export WANDB_API_KEY="<your-wandb-key>"

Then:
  # 1. hyperparameter search for the D-MPNN
  python scripts/hpopt_chemprop.py --num-samples 30 --num-gpus 1
  # 2. train the D-MPNN on the best config it found
  python scripts/train_chemprop.py --accelerator gpu \
      --config-path models/chemprop/hpopt/scaffold/best_config.toml
  # 3. fine-tune the CheMeleon foundation model (separate track)
  python scripts/finetune_chemeleon.py --accelerator gpu --epochs 30
EOF
