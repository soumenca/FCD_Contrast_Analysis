#!/bin/bash
set -euo pipefail

VENV_ROOT="${HOME}/fj32_scratch2/soumen/fcd/venvs"
VENV_DIR="${VENV_ROOT}/nnunet"
REPO_ROOT="${1:-${HOME}/fj32_scratch2/soumen/fcd/FCD_Contrast_Analysis}"

mkdir -p "${VENV_ROOT}"

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

pip install --upgrade pip setuptools wheel
pip install --no-cache-dir -r "${REPO_ROOT}/code/requirements_m3_nnunet.txt"

cat > "${VENV_ROOT}/activate_nnunet.sh" <<EOF
source ${VENV_DIR}/bin/activate
export nnUNet_raw=${REPO_ROOT}/data/nnunet_data/nnUNet_raw
export nnUNet_preprocessed=${REPO_ROOT}/data/nnunet_data/nnUNet_preprocessed
export nnUNet_results=${REPO_ROOT}/data/nnunet_data/nnUNet_results
EOF

chmod +x "${VENV_ROOT}/activate_nnunet.sh"

echo "Created nnU-Net environment at: ${VENV_DIR}"
echo "Activation helper: ${VENV_ROOT}/activate_nnunet.sh"
echo "Test with:"
echo "  source ${VENV_ROOT}/activate_nnunet.sh"
echo "  nnUNetv2_plan_and_preprocess -h"
