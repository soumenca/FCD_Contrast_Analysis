#!/bin/bash
# Submit 5-fold nnU-Net v2 CV training on M3.
#
# Usage:
#   sbatch code/submit_nnunet_cv_m3.sh 108
#   sbatch --export=ALL,DATASET_ID=105,CONFIG=3d_fullres code/submit_nnunet_cv_m3.sh

#SBATCH --job-name=nnUNet_CV
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=./logs/%x_fold%a_%j.out
#SBATCH --error=./logs/%x_fold%a_%j.err
#SBATCH --array=0-4

set -euo pipefail

mkdir -p ./logs

source ~/fj32_scratch2/soumen/fcd/venvs/activate_nnunet.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ID="${1:-${DATASET_ID:-108}}"
CONFIG="${CONFIG:-3d_fullres}"
FOLD="${SLURM_ARRAY_TASK_ID}"
PLANS="${PLANS:-nnUNetPlans}"
TRAINER="${TRAINER:-nnUNetTrainer}"
AUTO_PREP="${AUTO_PREP:-1}"
SPLITS_FILE="${SPLITS_FILE:-${REPO_ROOT}/data/nnunet_data/nnUNet_splits/splits_final.json}"
VERIFY_DATASET_INTEGRITY="${VERIFY_DATASET_INTEGRITY:-0}"
PREPROCESS_NP="${PREPROCESS_NP:-4}"

echo "Python diagnostics:"
python --version
which python

echo "nnUNet paths:"
echo "nnUNet_raw=${nnUNet_raw:-}"
echo "nnUNet_preprocessed=${nnUNet_preprocessed:-}"
echo "nnUNet_results=${nnUNet_results:-}"

echo "Job diagnostics:"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "DATASET_ID=${DATASET_ID}"
echo "CONFIG=${CONFIG}"
echo "PLANS=${PLANS}"
echo "TRAINER=${TRAINER}"
echo "FOLD=${FOLD}"
echo "AUTO_PREP=${AUTO_PREP}"
echo "SPLITS_FILE=${SPLITS_FILE}"

nvidia-smi

if [[ -z "${nnUNet_preprocessed:-}" ]]; then
  echo "nnUNet_preprocessed is not set"
  exit 1
fi

PREP_DONE_FILE="${nnUNet_preprocessed}/.dataset${DATASET_ID}_${CONFIG}_${PLANS}_ready"
PREP_LOCK_DIR="${nnUNet_preprocessed}/.dataset${DATASET_ID}_${CONFIG}_${PLANS}_lock"

prepare_dataset() {
  local verify_flag=""
  if [[ "${VERIFY_DATASET_INTEGRITY}" == "1" ]]; then
    verify_flag="--verify_dataset_integrity"
  fi

  echo "Running nnU-Net planning/preprocessing for Dataset${DATASET_ID}"
  nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" -c "${CONFIG}" -np "${PREPROCESS_NP}" ${verify_flag}

  echo "Installing shared splits_final.json for Dataset${DATASET_ID}"
  python3 "${REPO_ROOT}/code/install_nnunet_splits.py" \
    --splits-file "${SPLITS_FILE}" \
    --preprocessed-root "${nnUNet_preprocessed}" \
    --dataset-ids "${DATASET_ID}" \
    --overwrite
}

if [[ "${AUTO_PREP}" == "1" ]]; then
  if [[ -f "${PREP_DONE_FILE}" ]]; then
    echo "Dataset${DATASET_ID} preprocessing and split installation already marked done"
  elif mkdir "${PREP_LOCK_DIR}" 2>/dev/null; then
    trap 'rmdir "${PREP_LOCK_DIR}" 2>/dev/null || true' EXIT
    prepare_dataset
    touch "${PREP_DONE_FILE}"
    rmdir "${PREP_LOCK_DIR}" 2>/dev/null || true
    trap - EXIT
  else
    echo "Another array task is preparing Dataset${DATASET_ID}. Waiting for completion..."
    while [[ ! -f "${PREP_DONE_FILE}" ]]; do
      sleep 30
    done
    echo "Preparation marker detected. Continuing to training."
  fi
fi

nnUNetv2_train "${DATASET_ID}" "${CONFIG}" "${FOLD}" \
  -tr "${TRAINER}" \
  -p "${PLANS}" \
  --npz
