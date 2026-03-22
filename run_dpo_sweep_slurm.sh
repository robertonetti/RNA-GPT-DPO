#!/usr/bin/env bash
set -euo pipefail

# One-file Slurm sweep launcher for DPO training.
# Usage:
#   1) Add your Slurm options in SBATCH_ARGS below.
#   2) Run: bash run_dpo_sweep_slurm.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR_REL="data/clustering_methods/train_split_85"
TEST_DIR_REL="data/clustering_methods/test_split_15"
TRAIN_DIR="${ROOT_DIR}/${TRAIN_DIR_REL}"
TEST_DIR="${ROOT_DIR}/${TEST_DIR_REL}"

PYTHON_BIN="${PYTHON_BIN:-python}"

# Add your Slurm specs here, for example:
#   --partition=gpu
#   --gres=gpu:1
#   --cpus-per-task=8
#   --mem=32G
#   --time=24:00:00
SBATCH_ARGS=(
)

if [[ "${1:-}" == "--worker" ]]; then
    : "${TRAIN_CSV:?TRAIN_CSV not set}"
    : "${TEST_CSV:?TEST_CSV not set}"
    : "${FREEZE_LAYERS:?FREEZE_LAYERS not set}"
    : "${RUN_NAME:?RUN_NAME not set}"

    cd "${ROOT_DIR}"

    "${PYTHON_BIN}" - <<'PY'
import os
from dataclasses import replace

from src.dpo_config import CONFIG
from src.dpo_train import main

train_csv = os.environ["TRAIN_CSV"]
test_csv = os.environ["TEST_CSV"]
freeze_layers = int(os.environ["FREEZE_LAYERS"])
run_name = os.environ["RUN_NAME"]

cfg = replace(
    CONFIG,
    train_csv_mapping_path=train_csv,
    val_csv_mapping_path=test_csv,
    layers_to_freeze=freeze_layers,
    checkpoint_dir=f"checkpoints/checkpoints_roberto/sweep/{run_name}",
    image_dir=f"images_roberto/sweep/{run_name}",
    history_json_path=f"images_roberto/sweep/{run_name}/history.json",
)

print("Starting run with config:")
print(f"  train_csv_mapping_path={cfg.train_csv_mapping_path}")
print(f"  val_csv_mapping_path={cfg.val_csv_mapping_path}")
print(f"  layers_to_freeze={cfg.layers_to_freeze}")
print(f"  checkpoint_dir={cfg.checkpoint_dir}")
print(f"  image_dir={cfg.image_dir}")

main(cfg)
PY

    exit 0
fi

if [[ ! -d "${TRAIN_DIR}" ]]; then
    echo "Train directory not found: ${TRAIN_DIR}" >&2
    exit 1
fi

if [[ ! -d "${TEST_DIR}" ]]; then
    echo "Test directory not found: ${TEST_DIR}" >&2
    exit 1
fi

shopt -s nullglob

submitted=0
skipped=0

for train_csv_abs in "${TRAIN_DIR}"/*.csv; do
    base_name="$(basename "${train_csv_abs}")"
    test_csv_abs="${TEST_DIR}/${base_name}"

    if [[ ! -f "${test_csv_abs}" ]]; then
        echo "Skipping ${base_name}: missing counterpart in test_split_15"
        skipped=$((skipped + 1))
        continue
    fi

    train_csv_rel="${TRAIN_DIR_REL}/${base_name}"
    test_csv_rel="${TEST_DIR_REL}/${base_name}"

    for freeze in 0 1 2; do
        run_name="${base_name%.csv}_freeze${freeze}"

        sbatch \
            "${SBATCH_ARGS[@]}" \
            --job-name="dpo_${run_name}" \
            --chdir="${ROOT_DIR}" \
            --export="ALL,TRAIN_CSV=${train_csv_rel},TEST_CSV=${test_csv_rel},FREEZE_LAYERS=${freeze},RUN_NAME=${run_name},PYTHON_BIN=${PYTHON_BIN}" \
            "$0" --worker

        submitted=$((submitted + 1))
    done
done

echo "Submitted jobs: ${submitted}"
echo "Skipped CSV files (missing train/test pair): ${skipped}"
