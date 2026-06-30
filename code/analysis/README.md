# Analysis Scripts

This folder contains analysis utilities for evaluating trained nnU-Net models.

## Segmentation Metrics

Script:
- [compute_segmentation_metrics.py](/Users/soumen/wkdir/CodexApp/FCD_Contrast_Analysis/code/analysis/compute_segmentation_metrics.py)

What it does:

- reads validation predictions from each `fold_*` directory
- matches predictions to ground-truth masks in `labelsTr`
- computes per-case segmentation metrics
- aggregates results across folds

Metrics included:

- Dice
- IoU
- Precision
- Recall
- Specificity
- Accuracy
- False positive rate
- False negative rate
- predicted volume
- ground-truth volume
- volume difference

Expected prediction layout:

- `fold_0/validation/*.nii.gz`
- `fold_1/validation/*.nii.gz`
- `fold_2/validation/*.nii.gz`
- `fold_3/validation/*.nii.gz`
- `fold_4/validation/*.nii.gz`

Example:

```bash
python3 code/analysis/compute_segmentation_metrics.py \
  --model-dir /path/to/nnUNet_results/Dataset108_E8_T1w_FLAIR_T1wDivFLAIR_FLAIRDivT1w/nnUNetTrainer__nnUNetPlans__3d_fullres \
  --labels-dir /path/to/nnUNet_raw/Dataset108_E8_T1w_FLAIR_T1wDivFLAIR_FLAIRDivT1w/labelsTr \
  --output-dir /path/to/results/segmentation_metrics
```

Outputs:

- `segmentation_metrics_per_case.csv`
- `segmentation_metrics_per_fold.csv`
- `segmentation_metrics_summary.json`
