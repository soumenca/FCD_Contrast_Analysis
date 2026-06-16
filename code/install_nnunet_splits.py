#!/usr/bin/env python3
"""Copy a shared nnU-Net splits_final.json into one or more preprocessed dataset folders."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Install a shared splits_final.json into nnUNet_preprocessed dataset "
            "folders so all experiments use the same cross-validation splits."
        )
    )
    parser.add_argument(
        "--splits-file",
        type=Path,
        default=Path("data/nnUNet_splits/splits_final.json"),
        help="Path to the shared splits_final.json file.",
    )
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        required=True,
        help="Path to the nnUNet_preprocessed root directory.",
    )
    parser.add_argument(
        "--dataset-ids",
        type=int,
        nargs="*",
        default=list(range(101, 109)),
        help="Dataset IDs to update. Defaults to 101 102 103 104 105 106 107 108.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing splits_final.json in target dataset folders.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    splits_file = args.splits_file.resolve()
    preprocessed_root = args.preprocessed_root.resolve()

    if not splits_file.exists():
        raise FileNotFoundError(f"Shared splits file does not exist: {splits_file}")
    if not preprocessed_root.exists():
        raise FileNotFoundError(
            f"nnUNet_preprocessed root does not exist: {preprocessed_root}"
        )

    for dataset_id in args.dataset_ids:
        matches = sorted(preprocessed_root.glob(f"Dataset{dataset_id}_*"))
        if not matches:
            print(f"Skipping Dataset{dataset_id}: no matching folder under {preprocessed_root}")
            continue

        for dataset_dir in matches:
            target = dataset_dir / "splits_final.json"
            if target.exists() and not args.overwrite:
                print(f"Skipping {target}: already exists")
                continue

            shutil.copy2(splits_file, target)
            print(f"Copied splits to {target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
