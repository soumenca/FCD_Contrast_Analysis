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
    sensitivity = recall
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
        "sensitivity": sensitivity,
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
        "sensitivity",
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
        "num_zero_dice_cases": int((df["dice"] == 0).sum()),
        "zero_dice_rate": float((df["dice"] == 0).mean()) if len(df) > 0 else 0.0,
        "num_zero_dice_and_pred_positive_cases": int(
            ((df["dice"] == 0) & (df["pred_nonempty"] == 1)).sum()
        ),
        "num_zero_dice_and_pred_empty_cases": int(
            ((df["dice"] == 0) & (df["pred_nonempty"] == 0)).sum()
        ),
        "num_overlap_detected_cases": int((df["dice"] > 0).sum()),
    }
    for col in metric_cols:
        summary[f"{col}_mean"] = float(df[col].mean())
        summary[f"{col}_std"] = float(df[col].std(ddof=0))
        summary[f"{col}_median"] = float(df[col].median())
    return summary


def aggregate_positive_dice_metrics(df: pd.DataFrame, group_name: str) -> dict:
    filtered = df[df["dice"] > 0].copy()
    summary = {
        "group": f"{group_name}_dice_gt_0",
        "num_cases": int(len(filtered)),
        "detected_only_rate": float(len(filtered) / len(df)) if len(df) > 0 else 0.0,
    }
    if len(filtered) == 0:
        return summary

    metric_cols = [
        "dice",
        "iou",
        "precision",
        "recall",
        "sensitivity",
        "specificity",
        "accuracy",
        "false_positive_rate",
        "false_negative_rate",
        "pred_volume_mm3",
        "gt_volume_mm3",
        "volume_diff_mm3",
        "abs_volume_diff_mm3",
    ]
    for col in metric_cols:
        summary[f"{col}_mean"] = float(filtered[col].mean())
        summary[f"{col}_std"] = float(filtered[col].std(ddof=0))
        summary[f"{col}_median"] = float(filtered[col].median())
    return summary


def case_group(case_id: str) -> str:
    if case_id.startswith("FCD_"):
        return "FCD"
    if case_id.startswith("CON_"):
        return "control"
    return "unknown"


def build_paper_summary_rows(
    fcd_summary: dict,
    control_summary: dict,
    fcd_positive_dice_summary: dict,
    control_positive_dice_summary: dict,
) -> tuple[list[dict], list[dict]]:
    main_rows = [
        {
            "group": "FCD",
            "num_cases": fcd_summary["num_cases"],
            "dice_mean": fcd_summary["dice_mean"],
            "dice_std": fcd_summary["dice_std"],
            "sensitivity_mean": fcd_summary["sensitivity_mean"],
            "sensitivity_std": fcd_summary["sensitivity_std"],
            "precision_mean": fcd_summary["precision_mean"],
            "precision_std": fcd_summary["precision_std"],
            "zero_dice_cases": fcd_summary["fcd_zero_dice_cases"],
            "zero_dice_rate": fcd_summary["fcd_zero_dice_rate"],
            "control_cases_with_predicted_lesion": np.nan,
            "control_predicted_lesion_rate": np.nan,
        },
        {
            "group": "control",
            "num_cases": control_summary["num_cases"],
            "dice_mean": np.nan,
            "dice_std": np.nan,
            "sensitivity_mean": np.nan,
            "sensitivity_std": np.nan,
            "precision_mean": np.nan,
            "precision_std": np.nan,
            "zero_dice_cases": np.nan,
            "zero_dice_rate": np.nan,
            "control_cases_with_predicted_lesion": control_summary[
                "control_cases_with_predicted_lesion"
            ],
            "control_predicted_lesion_rate": control_summary[
                "control_predicted_lesion_rate"
            ],
        },
    ]

    detected_only_rows = [
        {
            "group": "FCD_detected_only",
            "num_cases_dice_gt_0": fcd_positive_dice_summary["num_cases"],
            "detected_only_rate": fcd_positive_dice_summary["detected_only_rate"],
            "dice_mean": fcd_positive_dice_summary.get("dice_mean", np.nan),
            "dice_std": fcd_positive_dice_summary.get("dice_std", np.nan),
            "sensitivity_mean": fcd_positive_dice_summary.get("sensitivity_mean", np.nan),
            "sensitivity_std": fcd_positive_dice_summary.get("sensitivity_std", np.nan),
            "precision_mean": fcd_positive_dice_summary.get("precision_mean", np.nan),
            "precision_std": fcd_positive_dice_summary.get("precision_std", np.nan),
        },
        {
            "group": "control_detected_only",
            "num_cases_dice_gt_0": control_positive_dice_summary["num_cases"],
            "detected_only_rate": control_positive_dice_summary["detected_only_rate"],
            "dice_mean": control_positive_dice_summary.get("dice_mean", np.nan),
            "dice_std": control_positive_dice_summary.get("dice_std", np.nan),
            "sensitivity_mean": control_positive_dice_summary.get("sensitivity_mean", np.nan),
            "sensitivity_std": control_positive_dice_summary.get("sensitivity_std", np.nan),
            "precision_mean": control_positive_dice_summary.get("precision_mean", np.nan),
            "precision_std": control_positive_dice_summary.get("precision_std", np.nan),
        },
    ]
    return main_rows, detected_only_rows


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
    fold_positive_dice_summaries: list[dict] = []

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
                "group": case_group(case_id),
                **metrics,
            }
            rows.append(row)

        fold_df = pd.DataFrame([row for row in rows if row["fold"] == fold_name])
        fold_summaries.append(aggregate_metrics(fold_df, fold_name))
        fold_positive_dice_summaries.append(
            aggregate_positive_dice_metrics(fold_df, fold_name)
        )

    if not rows:
        raise RuntimeError(
            f"No prediction/label pairs were processed from {model_dir} and {labels_dir}"
        )

    per_case_df = pd.DataFrame(rows).sort_values(["fold", "case_id"]).reset_index(drop=True)
    overall_summary = aggregate_metrics(per_case_df, "overall")

    fcd_df = per_case_df[per_case_df["group"] == "FCD"].copy()
    control_df = per_case_df[per_case_df["group"] == "control"].copy()
    fcd_summary = aggregate_metrics(fcd_df, "FCD")
    control_summary = aggregate_metrics(control_df, "control")
    overall_positive_dice_summary = aggregate_positive_dice_metrics(per_case_df, "overall")
    fcd_positive_dice_summary = aggregate_positive_dice_metrics(fcd_df, "FCD")
    control_positive_dice_summary = aggregate_positive_dice_metrics(control_df, "control")

    fcd_summary["fcd_zero_dice_cases"] = fcd_summary["num_zero_dice_cases"]
    fcd_summary["fcd_zero_dice_rate"] = fcd_summary["zero_dice_rate"]
    control_summary["control_cases_with_predicted_lesion"] = control_summary["num_pred_positive_cases"]
    control_summary["control_predicted_lesion_rate"] = safe_divide(
        control_summary["num_pred_positive_cases"],
        control_summary["num_cases"],
        both_empty_value=0.0,
    )

    fold_summary_df = pd.DataFrame(
        fold_summaries
        + [
            fcd_summary,
            control_summary,
            overall_summary,
        ]
    )
    positive_dice_summary_df = pd.DataFrame(
        fold_positive_dice_summaries
        + [
            fcd_positive_dice_summary,
            control_positive_dice_summary,
            overall_positive_dice_summary,
        ]
    )
    paper_rows, paper_detected_only_rows = build_paper_summary_rows(
        fcd_summary,
        control_summary,
        fcd_positive_dice_summary,
        control_positive_dice_summary,
    )
    paper_summary_df = pd.DataFrame(paper_rows)
    paper_detected_only_df = pd.DataFrame(paper_detected_only_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    per_case_csv = output_dir / "segmentation_metrics_per_case.csv"
    fcd_per_case_csv = output_dir / "segmentation_metrics_per_case_fcd.csv"
    control_per_case_csv = output_dir / "segmentation_metrics_per_case_control.csv"
    per_fold_csv = output_dir / "segmentation_metrics_per_fold.csv"
    positive_dice_csv = output_dir / "segmentation_metrics_dice_gt_0_summary.csv"
    paper_summary_csv = output_dir / "segmentation_metrics_paper_summary.csv"
    paper_detected_only_csv = (
        output_dir / "segmentation_metrics_paper_summary_detected_only.csv"
    )
    summary_json = output_dir / "segmentation_metrics_summary.json"

    per_case_df.to_csv(per_case_csv, index=False)
    fcd_df.to_csv(fcd_per_case_csv, index=False)
    control_df.to_csv(control_per_case_csv, index=False)
    fold_summary_df.to_csv(per_fold_csv, index=False)
    positive_dice_summary_df.to_csv(positive_dice_csv, index=False)
    paper_summary_df.to_csv(paper_summary_csv, index=False)
    paper_detected_only_df.to_csv(paper_detected_only_csv, index=False)
    summary_json.write_text(
        json.dumps(
            {
                "model_dir": str(model_dir),
                "labels_dir": str(labels_dir),
                "prediction_subdir": args.prediction_subdir,
                "folds": fold_summaries,
                "folds_dice_gt_0": fold_positive_dice_summaries,
                "fcd_summary": fcd_summary,
                "control_summary": control_summary,
                "fcd_summary_dice_gt_0": fcd_positive_dice_summary,
                "control_summary_dice_gt_0": control_positive_dice_summary,
                "overall_dice_gt_0": overall_positive_dice_summary,
                "overall": overall_summary,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"Wrote per-case metrics to {per_case_csv}")
    print(f"Wrote FCD per-case metrics to {fcd_per_case_csv}")
    print(f"Wrote control per-case metrics to {control_per_case_csv}")
    print(f"Wrote per-fold metrics to {per_fold_csv}")
    print(f"Wrote Dice>0 summary metrics to {positive_dice_csv}")
    print(f"Wrote paper summary metrics to {paper_summary_csv}")
    print(f"Wrote paper detected-only metrics to {paper_detected_only_csv}")
    print(f"Wrote summary JSON to {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
