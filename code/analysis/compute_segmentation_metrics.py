#!/usr/bin/env python3
"""Compute nnU-Net segmentation metrics across validation folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute voxel-wise segmentation metrics for all nnU-Net validation "
            "folds and save per-case and per-fold summaries."
        )
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help=(
            "nnU-Net training output directory containing fold_* folders. "
            "Example: .../Dataset108.../nnUNetTrainer__nnUNetPlans__3d_fullres"
        ),
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        required=True,
        help="Ground-truth labelsTr directory containing <case_id>.nii.gz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where analysis CSV/JSON outputs will be written.",
    )
    parser.add_argument(
        "--prediction-subdir",
        default="validation",
        help="Subdirectory inside each fold_* folder containing predicted segmentations.",
    )
    parser.add_argument(
        "--suffix",
        default=".nii.gz",
        help="Segmentation filename suffix for both predictions and labels.",
    )
    return parser.parse_args()


def safe_divide(numerator: float, denominator: float, both_empty_value: float = 1.0) -> float:
    if denominator == 0:
        return both_empty_value
    return numerator / denominator


def compute_case_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, voxel_volume_mm3: float) -> dict[str, float]:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = float(np.logical_and(np.logical_not(pred), gt).sum())
    tn = float(np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum())

    pred_voxels = float(pred.sum())
    gt_voxels = float(gt.sum())

    dice = safe_divide(2.0 * tp, 2.0 * tp + fp + fn)
    iou = safe_divide(tp, tp + fp + fn)
    precision = safe_divide(tp, tp + fp, both_empty_value=1.0 if gt_voxels == 0 else 0.0)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    accuracy = safe_divide(tp + tn, tp + tn + fp + fn)
    fpr = safe_divide(fp, fp + tn, both_empty_value=0.0)
    fnr = safe_divide(fn, fn + tp, both_empty_value=0.0)
    volume_diff_voxels = pred_voxels - gt_voxels
    abs_volume_diff_voxels = abs(volume_diff_voxels)
    volume_diff_mm3 = volume_diff_voxels * voxel_volume_mm3
    abs_volume_diff_mm3 = abs(volume_diff_mm3)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "pred_voxels": pred_voxels,
        "gt_voxels": gt_voxels,
        "pred_volume_mm3": pred_voxels * voxel_volume_mm3,
        "gt_volume_mm3": gt_voxels * voxel_volume_mm3,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "volume_diff_voxels": volume_diff_voxels,
        "abs_volume_diff_voxels": abs_volume_diff_voxels,
        "volume_diff_mm3": volume_diff_mm3,
        "abs_volume_diff_mm3": abs_volume_diff_mm3,
        "gt_nonempty": float(gt_voxels > 0),
        "pred_nonempty": float(pred_voxels > 0),
    }


def load_binary_mask(path: Path) -> tuple[np.ndarray, float]:
    image = nib.load(str(path))
    data = np.asarray(image.get_fdata(), dtype=np.float32)
    spacing = image.header.get_zooms()[:3]
    voxel_volume_mm3 = float(np.prod(spacing))
    return data > 0, voxel_volume_mm3


def aggregate_metrics(df: pd.DataFrame, group_name: str) -> dict:
    metric_cols = [
        "dice",
        "iou",
        "precision",
        "recall",
        "specificity",
        "accuracy",
        "false_positive_rate",
        "false_negative_rate",
        "pred_volume_mm3",
        "gt_volume_mm3",
        "volume_diff_mm3",
        "abs_volume_diff_mm3",
    ]
    summary = {
        "group": group_name,
        "num_cases": int(len(df)),
        "num_gt_positive_cases": int(df["gt_nonempty"].sum()),
        "num_pred_positive_cases": int(df["pred_nonempty"].sum()),
    }
    for col in metric_cols:
        summary[f"{col}_mean"] = float(df[col].mean())
        summary[f"{col}_std"] = float(df[col].std(ddof=0))
        summary[f"{col}_median"] = float(df[col].median())
    return summary


def find_fold_dirs(model_dir: Path) -> list[Path]:
    return sorted(path for path in model_dir.glob("fold_*") if path.is_dir())


def main() -> int:
    args = parse_args()

    model_dir = args.model_dir.resolve()
    labels_dir = args.labels_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory does not exist: {labels_dir}")

    fold_dirs = find_fold_dirs(model_dir)
    if not fold_dirs:
        raise FileNotFoundError(f"No fold_* directories found under {model_dir}")

    rows: list[dict] = []
    fold_summaries: list[dict] = []

    for fold_dir in fold_dirs:
        prediction_dir = fold_dir / args.prediction_subdir
        if not prediction_dir.exists():
            print(f"Skipping {fold_dir.name}: missing prediction directory {prediction_dir}")
            continue

        prediction_files = sorted(prediction_dir.glob(f"*{args.suffix}"))
        prediction_files = [p for p in prediction_files if p.name != "summary.json"]
        if not prediction_files:
            print(f"Skipping {fold_dir.name}: no prediction files found in {prediction_dir}")
            continue

        fold_name = fold_dir.name
        for pred_path in prediction_files:
            case_id = pred_path.name.removesuffix(args.suffix)
            gt_path = labels_dir / pred_path.name
            if not gt_path.exists():
                raise FileNotFoundError(f"Missing ground-truth label for {case_id}: {gt_path}")

            pred_mask, pred_voxel_volume = load_binary_mask(pred_path)
            gt_mask, gt_voxel_volume = load_binary_mask(gt_path)
            if pred_mask.shape != gt_mask.shape:
                raise ValueError(
                    f"Shape mismatch for {case_id}: pred {pred_mask.shape} vs gt {gt_mask.shape}"
                )

            voxel_volume_mm3 = gt_voxel_volume
            if not np.isclose(pred_voxel_volume, gt_voxel_volume):
                print(
                    f"Warning: voxel volume mismatch for {case_id}; "
                    f"using ground-truth volume {gt_voxel_volume:.6f} mm^3"
                )

            metrics = compute_case_metrics(pred_mask, gt_mask, voxel_volume_mm3)
            row = {
                "fold": fold_name,
                "case_id": case_id,
                **metrics,
            }
            rows.append(row)

        fold_df = pd.DataFrame([row for row in rows if row["fold"] == fold_name])
        fold_summaries.append(aggregate_metrics(fold_df, fold_name))

    if not rows:
        raise RuntimeError(
            f"No prediction/label pairs were processed from {model_dir} and {labels_dir}"
        )

    per_case_df = pd.DataFrame(rows).sort_values(["fold", "case_id"]).reset_index(drop=True)
    overall_summary = aggregate_metrics(per_case_df, "overall")
    fold_summary_df = pd.DataFrame(fold_summaries + [overall_summary])

    output_dir.mkdir(parents=True, exist_ok=True)
    per_case_csv = output_dir / "segmentation_metrics_per_case.csv"
    per_fold_csv = output_dir / "segmentation_metrics_per_fold.csv"
    summary_json = output_dir / "segmentation_metrics_summary.json"

    per_case_df.to_csv(per_case_csv, index=False)
    fold_summary_df.to_csv(per_fold_csv, index=False)
    summary_json.write_text(
        json.dumps(
            {
                "model_dir": str(model_dir),
                "labels_dir": str(labels_dir),
                "prediction_subdir": args.prediction_subdir,
                "folds": fold_summaries,
                "overall": overall_summary,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"Wrote per-case metrics to {per_case_csv}")
    print(f"Wrote per-fold metrics to {per_fold_csv}")
    print(f"Wrote summary JSON to {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
