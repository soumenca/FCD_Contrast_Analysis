# Bottleneck Feature Extraction

This workflow extracts one subject-level feature vector from a trained nnU-Net v2
model for downstream separability analysis such as PCA, UMAP, or t-SNE.

## What the script does

The script [extract_nnunet_bottleneck_features.py](/Users/soumen/wkdir/CodexApp/FCD_Contrast_Analysis/code/extract_nnunet_bottleneck_features.py):

1. Loads a trained nnU-Net v2 fold.
2. Preprocesses each subject with nnU-Net's own preprocessing pipeline.
3. Runs the encoder only and takes the deepest encoder feature map.
4. Applies global average pooling to each bottleneck patch.
5. Averages pooled patch vectors into one final vector per subject.
6. Saves one row per subject to CSV.

This means the script already performs:

`encoder bottleneck feature map -> global average pooling -> one vector per subject`

## Required inputs

- A trained nnU-Net model folder, for example:
  `data/nnUNet_results/Dataset108_E8_T1w_FLAIR_T1wDivFLAIR_FLAIRDivT1w/nnUNetTrainer__nnUNetPlans__3d_fullres`
- The matching nnU-Net raw dataset folder, for example:
  `data/nnUNet_raw/Dataset108_E8_T1w_FLAIR_T1wDivFLAIR_FLAIRDivT1w`
- The shared split file if you want only train or validation subjects:
  `data/nnUNet_splits/splits_final.json`

## Example runs

### Extract validation vectors for fold 0

```bash
python3 code/extract_nnunet_bottleneck_features.py \
  --model-folder data/nnUNet_results/Dataset108_E8_T1w_FLAIR_T1wDivFLAIR_FLAIRDivT1w/nnUNetTrainer__nnUNetPlans__3d_fullres \
  --raw-dataset-dir data/nnUNet_raw/Dataset108_E8_T1w_FLAIR_T1wDivFLAIR_FLAIRDivT1w \
  --fold 0 \
  --subset val \
  --splits-file data/nnUNet_splits/splits_final.json \
  --output-csv results/fold0_val_bottleneck_features.csv \
  --output-npy results/fold0_val_bottleneck_features.npy \
  --metadata-json results/fold0_val_bottleneck_features.json
```

### Extract training vectors for fold 0

```bash
python3 code/extract_nnunet_bottleneck_features.py \
  --model-folder data/nnUNet_results/Dataset108_E8_T1w_FLAIR_T1wDivFLAIR_FLAIRDivT1w/nnUNetTrainer__nnUNetPlans__3d_fullres \
  --raw-dataset-dir data/nnUNet_raw/Dataset108_E8_T1w_FLAIR_T1wDivFLAIR_FLAIRDivT1w \
  --fold 0 \
  --subset train \
  --splits-file data/nnUNet_splits/splits_final.json \
  --output-csv results/fold0_train_bottleneck_features.csv
```

### Extract all subjects for fold 0

```bash
python3 code/extract_nnunet_bottleneck_features.py \
  --model-folder data/nnUNet_results/Dataset108_E8_T1w_FLAIR_T1wDivFLAIR_FLAIRDivT1w/nnUNetTrainer__nnUNetPlans__3d_fullres \
  --raw-dataset-dir data/nnUNet_raw/Dataset108_E8_T1w_FLAIR_T1wDivFLAIR_FLAIRDivT1w \
  --fold 0 \
  --subset all \
  --output-csv results/fold0_all_bottleneck_features.csv
```

### Run on CPU if needed

```bash
python3 code/extract_nnunet_bottleneck_features.py \
  --model-folder data/nnUNet_results/Dataset108_E8_T1w_FLAIR_T1wDivFLAIR_FLAIRDivT1w/nnUNetTrainer__nnUNetPlans__3d_fullres \
  --raw-dataset-dir data/nnUNet_raw/Dataset108_E8_T1w_FLAIR_T1wDivFLAIR_FLAIRDivT1w \
  --fold 0 \
  --subset val \
  --splits-file data/nnUNet_splits/splits_final.json \
  --output-csv results/fold0_val_bottleneck_features_cpu.csv \
  --device cpu
```

## Output format

The CSV contains one row per subject with fields such as:

- `case_id`
- `group`
- `subset`
- `fold`
- `num_tiles`
- `bottleneck_channels`
- `bottleneck_feature_map_shape`
- `preprocessed_shape`
- `feat_0000`, `feat_0001`, ...

`feat_*` columns are the final pooled bottleneck vector for each subject.

## Typical next step

After extraction, use the CSV or `.npy` file for:

- PCA
- UMAP
- t-SNE

Then compare clustering or separability between:

- `FCD`
- `control`
