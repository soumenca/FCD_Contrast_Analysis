#!/bin/bash
# Submit 5-fold nnU-Net v2 CV training on M3.
#
# Usage:
#   sbatch code/submit_nnunet_cv_m3.sh 108
#   sbatch --export=ALL,DATASET_ID=105,CONFIG=3d_fullres code/submit_nnunet_cv_m3.sh

#SBATCH --job-name=nnUNet_CV
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=./logs/%x_fold%a_%j.out
#SBATCH --error=./logs/%x_fold%a_%j.err
#SBATCH --array=0-4

set -euo pipefail

mkdir -p ./logs

source ./venvs/activate_nnunet.sh

DATASET_ID="${1:-${DATASET_ID:-108}}"
CONFIG="${CONFIG:-3d_fullres}"
FOLD="${SLURM_ARRAY_TASK_ID}"
PLANS="${PLANS:-nnUNetPlans}"
TRAINER="${TRAINER:-nnUNetTrainer}"

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

nvidia-smi

nnUNetv2_train "${DATASET_ID}" "${CONFIG}" "${FOLD}" \
  -tr "${TRAINER}" \
  -p "${PLANS}" \
  --npz
