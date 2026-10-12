# E1–E8 subject-level and fold-wise results

These files summarize existing held-out predictions for 85 FCD subjects. No retraining or new inference was performed. Coded case IDs match the shared [splits_final.json](../../data/nnUNet_splits/splits_final.json). Controls are excluded from this analysis.

## Files

- `subject_level_dice_E1_E8.csv`: one row per FCD subject, with case ID, zero-based fold, eight Dice values and `detected_all_eight` (1 when all eight Dice values are strictly greater than zero).
- `subject_level_metrics_E1_E8.csv`: 680 rows, one per subject/configuration, containing Dice, sensitivity and precision for further analysis.
- `common_detected_dice_summary.csv`: mean Dice and between-subject SD for each configuration on the identical 16-subject intersection.
- `fold_wise_metrics.csv`: subject counts, means and between-subject SD within each fold, separately for all FCD subjects and the common detected subset. Includes Dice, sensitivity and precision.
- `variability_across_fold_means.csv`: unweighted means and SDs of the nonempty fold means. This measures between-fold variability, unlike the between-subject SD in the preceding files.
- `provenance.json`: source paths and SHA-256 hashes, split-file hash, validation checks and output hashes.

Dice, sensitivity and precision are on the 0–1 scale. `sd_population` uses divisor N (`ddof=0`), matching the existing paper-summary code. `sd_sample` uses divisor N−1 (`ddof=1`). A sample SD is blank when fewer than two observations are available. In the across-fold file, N is the number of nonempty fold means, explicitly recorded as `fold_means_n`. These are SDs, not standard errors or confidence intervals.

## Configurations

| Configuration | Input channels |
|---|---|
| E1 | T1w |
| E2 | FLAIR |
| E3 | T1w/FLAIR |
| E4 | FLAIR/T1w |
| E5 | T1w + FLAIR |
| E6 | T1w + FLAIR + T1w/FLAIR |
| E7 | T1w + FLAIR + FLAIR/T1w |
| E8 | T1w + FLAIR + T1w/FLAIR + FLAIR/T1w |

## Common detected subjects

Sixteen of 85 FCD subjects (18.82%) have Dice > 0 in all eight configurations. Their distribution across folds 0–4 is 3, 1, 5, 3 and 4. The full FCD cohort has 17 held-out subjects in each fold.

| Configuration | Mean Dice on the same 16 subjects |
|---|---:|
| E1 | 0.5449 |
| E2 | 0.5175 |
| E3 | 0.5005 |
| E4 | 0.6039 |
| E5 | 0.5468 |
| E6 | 0.5518 |
| E7 | 0.5319 |
| E8 | 0.5430 |

The common subset is selected using observed predictions from all eight configurations. It is a supplementary conditional comparison and does not replace the full-cohort results. Dice > 0 indicates some voxel overlap, not a clinical detection threshold. No significance testing is performed here. Because common-subset fold sizes differ, its pooled subject mean generally differs from its unweighted mean of fold means.

## Reproduction and checks

The export script is `code/analysis/compare_configurations.py`. With the original local analysis exports available, run from the repository root:

```bash
python3 code/analysis/compare_configurations.py
```

The original `doc/analysis/` exports are local inputs and are not included in this directory. Their hashes are recorded in `provenance.json`. All reported comparisons can also be recalculated directly from the published subject-level CSVs and split file.

The script checks five disjoint train/validation splits, one validation assignment per case, complete and unique FCD coverage in every configuration, exact agreement with the original per-case exports, matching fold assignments, metric ranges, Dice against confusion counts, consistent ground-truth sizes and agreement with existing paper-summary means and population SDs. Existing prediction checkpoint provenance is not established by these metric files.
