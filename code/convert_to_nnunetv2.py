#!/usr/bin/env python3
"""Convert the organized dataset into nnU-Net v2 raw datasets."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from random import Random


DATASET_SPECS = [
    (101, "E1_T1w", [("T1w", "0000")]),
    (102, "E2_FLAIR", [("FLAIR", "0000")]),
    (103, "E3_T1wDivFLAIR", [("T1w_div_FLAIR", "0000")]),
    (104, "E4_FLAIRDivT1w", [("FLAIR_div_T1w", "0000")]),
    (105, "E5_T1w_FLAIR", [("T1w", "0000"), ("FLAIR", "0001")]),
    (
        106,
        "E6_T1w_FLAIR_T1wDivFLAIR",
        [("T1w", "0000"), ("FLAIR", "0001"), ("T1w_div_FLAIR", "0002")],
    ),
    (
        107,
        "E7_T1w_FLAIR_FLAIRDivT1w",
        [("T1w", "0000"), ("FLAIR", "0001"), ("FLAIR_div_T1w", "0002")],
    ),
    (
        108,
        "E8_T1w_FLAIR_T1wDivFLAIR_FLAIRDivT1w",
        [
            ("T1w", "0000"),
            ("FLAIR", "0001"),
            ("T1w_div_FLAIR", "0002"),
            ("FLAIR_div_T1w", "0003"),
        ],
    ),
]


@dataclass(frozen=True)
class Case:
    case_id: str
    cohort: str
    folder: Path
    label_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the local dataset into nnU-Net v2 raw datasets."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/dataset"),
        help="Source dataset root containing subjects/ and controls/ folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/nnUNet_raw"),
        help="Destination root for nnU-Net v2 raw datasets.",
    )
    parser.add_argument(
        "--splits-output",
        type=Path,
        default=Path("data/nnUNet_splits/splits_final.json"),
        help="Path for the shared 5-fold CV splits_final.json.",
    )
    parser.add_argument(
        "--mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="Whether to symlink or copy source NIfTI files into nnU-Net folders.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to generate deterministic 5-fold splits.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of cross-validation folds to generate.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing dataset directories and split file output.",
    )
    return parser.parse_args()


def safe_mkdir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if overwrite:
            dst.unlink()
        else:
            return

    if mode == "copy":
        shutil.copy2(src, dst)
        return

    os.symlink(src.resolve(), dst)


def collect_cases(dataset_root: Path) -> list[Case]:
    cases: list[Case] = []
    for cohort_name, prefix in (("subjects", "FCD"), ("controls", "CON")):
        cohort_root = dataset_root / cohort_name
        folders = sorted(path for path in cohort_root.iterdir() if path.is_dir())
        for index, folder in enumerate(folders, start=1):
            label_path = folder / "label.nii.gz"
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label file: {label_path}")
            cases.append(
                Case(
                    case_id=f"{prefix}_{index:03d}",
                    cohort=cohort_name,
                    folder=folder,
                    label_path=label_path,
                )
            )
    return cases


def make_folds(case_ids: list[str], folds: int, rng: Random) -> list[list[str]]:
    shuffled = case_ids[:]
    rng.shuffle(shuffled)
    return [shuffled[index::folds] for index in range(folds)]


def build_splits(cases: list[Case], folds: int, seed: int) -> list[dict[str, list[str]]]:
    rng = Random(seed)
    subject_ids = [case.case_id for case in cases if case.cohort == "subjects"]
    control_ids = [case.case_id for case in cases if case.cohort == "controls"]

    subject_folds = make_folds(subject_ids, folds, rng)
    control_folds = make_folds(control_ids, folds, rng)

    splits: list[dict[str, list[str]]] = []
    all_case_ids = sorted(case.case_id for case in cases)
    for fold_index in range(folds):
        val_ids = sorted(subject_folds[fold_index] + control_folds[fold_index])
        val_set = set(val_ids)
        train_ids = [case_id for case_id in all_case_ids if case_id not in val_set]
        splits.append({"train": train_ids, "val": val_ids})
    return splits


def build_dataset_json(dataset_name: str, channels: list[tuple[str, str]], num_cases: int) -> dict:
    channel_names = {
        str(index): channel_name for index, (channel_name, _channel_suffix) in enumerate(channels)
    }
    labels = {
        "background": 0,
        "lesion": 1,
    }
    return {
        "channel_names": channel_names,
        "labels": labels,
        "numTraining": num_cases,
        "file_ending": ".nii.gz",
        "name": dataset_name,
        "overwrite_image_reader_writer": "SimpleITKIO",
    }


def write_dataset(
    dataset_root: Path,
    dataset_name: str,
    channels: list[tuple[str, str]],
    cases: list[Case],
    mode: str,
    overwrite: bool,
) -> None:
    images_tr = dataset_root / "imagesTr"
    labels_tr = dataset_root / "labelsTr"
    images_ts = dataset_root / "imagesTs"

    safe_mkdir(dataset_root, overwrite)
    images_tr.mkdir(exist_ok=True)
    labels_tr.mkdir(exist_ok=True)
    images_ts.mkdir(exist_ok=True)

    for case in cases:
        for channel_name, channel_index in channels:
            src = case.folder / f"{channel_name}.nii.gz"
            if not src.exists():
                raise FileNotFoundError(f"Missing input file: {src}")
            dst = images_tr / f"{case.case_id}_{channel_index}.nii.gz"
            link_or_copy(src, dst, mode, overwrite)

        label_dst = labels_tr / f"{case.case_id}.nii.gz"
        link_or_copy(case.label_path, label_dst, mode, overwrite)

    dataset_json = build_dataset_json(dataset_name, channels, len(cases))
    (dataset_root / "dataset.json").write_text(json.dumps(dataset_json, indent=2) + "\n")


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    splits_output = args.splits_output.resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    cases = collect_cases(dataset_root)
    if not cases:
        raise FileNotFoundError(f"No cases found under {dataset_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    for dataset_id, dataset_suffix, channels in DATASET_SPECS:
        dataset_name = f"Dataset{dataset_id}_{dataset_suffix}"
        dataset_dir = output_root / dataset_name
        write_dataset(
            dataset_root=dataset_dir,
            dataset_name=dataset_name,
            channels=channels,
            cases=cases,
            mode=args.mode,
            overwrite=args.overwrite,
        )
        print(f"Wrote {dataset_name}")

    splits = build_splits(cases, args.folds, args.seed)
    splits_output.parent.mkdir(parents=True, exist_ok=True)
    if splits_output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Splits file already exists: {splits_output}. Use --overwrite to replace it."
        )
    splits_output.write_text(json.dumps(splits, indent=2) + "\n")
    print(f"Wrote shared splits file to {splits_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
