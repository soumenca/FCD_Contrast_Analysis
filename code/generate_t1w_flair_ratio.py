#!/usr/bin/env python3
"""Generate safe T1w/FLAIR and FLAIR/T1w ratio images for the dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate ratio images for every subject folder under a dataset "
            "root and save the results next to the source images."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/dataset"),
        help="Dataset root containing subjects/ and controls/ directories.",
    )
    parser.add_argument(
        "--output-name",
        default="T1w_div_FLAIR.nii.gz",
        help="Filename to use for the generated T1w/FLAIR ratio image.",
    )
    parser.add_argument(
        "--inverse-output-name",
        default="FLAIR_div_T1w.nii.gz",
        help="Filename to use for the generated FLAIR/T1w ratio image.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
        help="Minimum absolute denominator value allowed during division.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing generated ratio image.",
    )
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="Add generated ratio filenames to each subject_manifest.json files map.",
    )
    return parser.parse_args()


def subject_dirs(dataset_root: Path) -> list[Path]:
    dirs: list[Path] = []
    for cohort in ("subjects", "controls"):
        cohort_root = dataset_root / cohort
        if not cohort_root.exists():
            continue
        dirs.extend(sorted(path for path in cohort_root.iterdir() if path.is_dir()))
    return dirs


def update_manifest(manifest_path: Path, mapping: dict[str, str]) -> None:
    manifest = json.loads(manifest_path.read_text())
    files = manifest.setdefault("files", {})
    files.update(mapping)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def build_ratio_image(
    numerator_path: Path, denominator_path: Path, epsilon: float
) -> nib.Nifti1Image:
    numerator_img = nib.load(str(numerator_path))
    denominator_img = nib.load(str(denominator_path))

    numerator = np.asarray(numerator_img.get_fdata(dtype=np.float32), dtype=np.float32)
    denominator = np.asarray(
        denominator_img.get_fdata(dtype=np.float32), dtype=np.float32
    )

    if numerator.shape != denominator.shape:
        raise ValueError(
            f"Shape mismatch for {numerator_path.parent}: "
            f"{numerator_path.name} {numerator.shape} vs "
            f"{denominator_path.name} {denominator.shape}"
        )

    ratio = np.zeros_like(numerator, dtype=np.float32)
    valid = np.abs(denominator) > epsilon

    # Keep undefined voxels at zero so zero-valued denominators do not create inf/nan artifacts.
    np.divide(numerator, denominator, out=ratio, where=valid)
    ratio[~np.isfinite(ratio)] = 0.0

    header = numerator_img.header.copy()
    header.set_data_dtype(np.float32)
    return nib.Nifti1Image(ratio, affine=numerator_img.affine, header=header)


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    all_subject_dirs = subject_dirs(dataset_root)
    if not all_subject_dirs:
        raise FileNotFoundError(f"No subject folders found under {dataset_root}")

    generated = 0
    skipped = 0

    for folder in all_subject_dirs:
        t1w_path = folder / "T1w.nii.gz"
        flair_path = folder / "FLAIR.nii.gz"
        output_path = folder / args.output_name
        inverse_output_path = folder / args.inverse_output_name
        manifest_path = folder / "subject_manifest.json"

        if not t1w_path.exists() or not flair_path.exists():
            print(f"Skipping {folder}: missing T1w.nii.gz or FLAIR.nii.gz")
            skipped += 1
            continue

        if (
            output_path.exists()
            and inverse_output_path.exists()
            and not args.overwrite
        ):
            print(
                f"Skipping {folder}: both {args.output_name} and "
                f"{args.inverse_output_name} already exist"
            )
            skipped += 1
            continue

        if args.overwrite or not output_path.exists():
            ratio_img = build_ratio_image(t1w_path, flair_path, args.epsilon)
            nib.save(ratio_img, str(output_path))

        if args.overwrite or not inverse_output_path.exists():
            inverse_ratio_img = build_ratio_image(flair_path, t1w_path, args.epsilon)
            nib.save(inverse_ratio_img, str(inverse_output_path))

        if args.update_manifest and manifest_path.exists():
            update_manifest(
                manifest_path,
                {
                    "T1w_div_FLAIR": args.output_name,
                    "FLAIR_div_T1w": args.inverse_output_name,
                },
            )

        print(f"Generated ratio images in {folder}")
        generated += 1

    print(f"Done. Processed {generated} folders; skipped {skipped}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
