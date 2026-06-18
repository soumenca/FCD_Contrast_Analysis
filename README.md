# FCD Contrast Analysis

This repository contains scripts to:

- organize the local FCD dataset
- generate derived contrast images
- convert the dataset to nnU-Net v2 format
- run 5-fold nnU-Net training on M3
- extract encoder bottleneck features
- prepare subject-level vectors for PCA, UMAP, or t-SNE

## Repository layout

- [code/](/Users/soumen/wkdir/CodexApp/FCD_Contrast_Analysis/code): scripts
- [data/](/Users/soumen/wkdir/CodexApp/FCD_Contrast_Analysis/data): local dataset and nnU-Net outputs

Important:

- `data/` is ignored by Git
- `__pycache__/` is ignored by Git

## Current dataset layout

The local dataset is expected to look like:

- `data/dataset/subjects/<subject_id>`
- `data/dataset/controls/<subject_id>`

Each subject folder should contain:

- `T1w.nii.gz`
- `FLAIR.nii.gz`
- `brain_mask.nii.gz`
- `label.nii.gz`
- `subject_manifest.json`

## Scripts

### 1. Generate ratio images

Script:
- [code/generate_t1w_flair_ratio.py](/Users/soumen/wkdir/CodexApp/FCD_Contrast_Analysis/code/generate_t1w_flair_ratio.py)

What it does:

- creates `T1w_div_FLAIR.nii.gz`
- creates `FLAIR_div_T1w.nii.gz`
- handles divide-by-zero safely
- can update each `subject_manifest.json`

Run:

```bash
python3 code/generate_t1w_flair_ratio.py --dataset-root data/dataset --update-manifest
```

### 2. Convert to nnU-Net v2 datasets

Script:
- [code/convert_to_nnunetv2.py](/Users/soumen/wkdir/CodexApp/FCD_Contrast_Analysis/code/convert_to_nnunetv2.py)

This creates:

- `Dataset101`: E1 = `T1w`
- `Dataset102`: E2 = `FLAIR`
- `Dataset103`: E3 = `T1w/FLAIR`
- `Dataset104`: E4 = `FLAIR/T1w`
- `Dataset105`: E5 = `T1w + FLAIR`
- `Dataset106`: E6 = `T1w + FLAIR + T1w/FLAIR`
- `Dataset107`: E7 = `T1w + FLAIR + FLAIR/T1w`
- `Dataset108`: E8 = `T1w + FLAIR + T1w/FLAIR + FLAIR/T1w`

It also creates one shared 5-fold split file:

- `data/nnunet_data/nnUNet_splits/splits_final.json`

Run:

```bash
python3 code/convert_to_nnunetv2.py \
  --dataset-root data/dataset \
  --output-root data/nnunet_data/nnUNet_raw \
  --splits-output data/nnunet_data/nnUNet_splits/splits_final.json \
  --mode symlink \
  --overwrite
```

### 3. Install the shared CV splits into nnU-Net preprocessed folders

Script:
- [code/install_nnunet_splits.py](/Users/soumen/wkdir/CodexApp/FCD_Contrast_Analysis/code/install_nnunet_splits.py)

nnU-Net training reads `splits_final.json` from the matching
`nnUNet_preprocessed/DatasetXXX...` folder. Use this helper to copy the same
split file into all experiment datasets so E1 to E8 use identical folds.

Run:

```bash
python3 code/install_nnunet_splits.py \
  --splits-file data/nnunet_data/nnUNet_splits/splits_final.json \
  --preprocessed-root /path/to/nnUNet_preprocessed \
  --overwrite
```

## nnU-Net training on M3

Submission script:
- [code/submit_nnunet_cv_m3.sh](/Users/soumen/wkdir/CodexApp/FCD_Contrast_Analysis/code/submit_nnunet_cv_m3.sh)

This script:

- can automatically run nnU-Net planning and preprocessing
- can automatically install the shared `splits_final.json`
- runs 5-fold training with `#SBATCH --array=0-4`
- activates `./venvs/activate_nnunet.sh`
- trains one fold per array task

Default settings:

- `DATASET_ID=108`
- `CONFIG=3d_fullres`
- `TRAINER=nnUNetTrainer`
- `PLANS=nnUNetPlans`
- SLURM defaults: `1 GPU`, `8 CPUs`, `96G RAM`, `2 days`

Submit example:

```bash
cd FCD_Contrast_Analysis
sbatch code/submit_nnunet_cv_m3.sh 108
```

That single command can now do the full training setup automatically for the
chosen dataset:

1. run `nnUNetv2_plan_and_preprocess`
2. copy the shared `splits_final.json` into the matching preprocessed folder
3. start 5-fold CV training

Override example:

```bash
sbatch --export=ALL,DATASET_ID=105,CONFIG=3d_fullres,TRAINER=nnUNetTrainer,PLANS=nnUNetPlans code/submit_nnunet_cv_m3.sh
```

Logs:

- `./logs/*.out`
- `./logs/*.err`

Useful cluster commands:

```bash
squeue -u $USER
sacct -j <jobid>
```

### M3 requirements before training

Make sure these environment variables are set:

- `nnUNet_raw`
- `nnUNet_preprocessed`
- `nnUNet_results`

The script can do preprocessing automatically, so you do not need to run that
manually unless you want to.

Optional controls:

- `AUTO_PREP=1`: default, auto-run preprocessing and install splits
- `VERIFY_DATASET_INTEGRITY=1`: run dataset integrity check during preprocessing
- `PREPROCESS_NP=4`: number of preprocessing workers

If your dataset or preprocessing step needs more memory or more time, increase
the SLURM requests in `code/submit_nnunet_cv_m3.sh`. The current defaults are
intentionally leaner to improve scheduling on the cluster.

### Rebuild the nnU-Net venv on M3

Files:

- [code/requirements_m3_nnunet.txt](/Users/soumen/wkdir/CodexApp/FCD_Contrast_Analysis/code/requirements_m3_nnunet.txt)
- [code/setup_m3_nnunet_env.sh](/Users/soumen/wkdir/CodexApp/FCD_Contrast_Analysis/code/setup_m3_nnunet_env.sh)

These pin a safer M3 environment for this project, including:

- `numpy<2`
- `torch==2.1.2`
- `torchvision==0.16.2`
- `torchaudio==2.1.2`
- `nnunetv2` installed separately with `--no-deps`

This avoids the broken `python-gdcm` build path, which is not needed for this
project because the workflow uses `.nii.gz` data rather than DICOM input.

Run on M3:

```bash
cd ~/fj32_scratch2/soumen/fcd/FCD_Contrast_Analysis
bash code/setup_m3_nnunet_env.sh
```

Then activate:

```bash
source ~/fj32_scratch2/soumen/fcd/venvs/activate_nnunet.sh
```

## Bottleneck feature extraction

Main script:
- [code/extract_nnunet_bottleneck_features.py](/Users/soumen/wkdir/CodexApp/FCD_Contrast_Analysis/code/extract_nnunet_bottleneck_features.py)

Detailed notes:
- [code/README_bottleneck_features.md](/Users/soumen/wkdir/CodexApp/FCD_Contrast_Analysis/code/README_bottleneck_features.md)

What it does:

1. loads a trained nnU-Net fold
2. preprocesses each subject with nnU-Net preprocessing
3. extracts the deepest encoder bottleneck feature map
4. applies global average pooling
5. produces one vector per subject
6. saves subject-level features to CSV and optionally `.npy`

Example:

```bash
python3 code/extract_nnunet_bottleneck_features.py \
  --model-folder data/nnUNet_results/Dataset108_E8_T1w_FLAIR_T1wDivFLAIR_FLAIRDivT1w/nnUNetTrainer__nnUNetPlans__3d_fullres \
  --raw-dataset-dir data/nnunet_data/nnUNet_raw/Dataset108_E8_T1w_FLAIR_T1wDivFLAIR_FLAIRDivT1w \
  --fold 0 \
  --subset val \
  --splits-file data/nnunet_data/nnUNet_splits/splits_final.json \
  --output-csv results/fold0_val_bottleneck_features.csv \
  --output-npy results/fold0_val_bottleneck_features.npy \
  --metadata-json results/fold0_val_bottleneck_features.json
```

## Suggested experiment workflow

1. Prepare local dataset under `data/dataset`
2. Generate ratio images
3. Convert to nnU-Net v2 format
4. Submit one M3 batch job for training
5. Extract bottleneck feature vectors from trained folds
6. Run PCA / UMAP / t-SNE on the extracted subject vectors
7. Compare `FCD` vs `control` separability

## Recommended training length

For final nnU-Net experiments:

- preferred final run: `1000` epochs
- quick pilot run: `200-300` epochs
- reasonable compromise: `500` epochs

For fair comparison across E1 to E8, keep the training policy the same for all experiments.

## Clone on M3

SSH:

```bash
git clone git@github.com:soumenca/FCD_Contrast_Analysis.git
cd FCD_Contrast_Analysis
```

HTTPS:

```bash
git clone https://github.com/soumenca/FCD_Contrast_Analysis.git
cd FCD_Contrast_Analysis
```

## Notes

- The repo tracks code and docs only. The dataset and nnU-Net outputs are local.
- The bottleneck extraction script expects a trained nnU-Net model folder to already exist.
- For downstream visualization, the next natural step is a separate PCA / UMAP / t-SNE analysis script.
