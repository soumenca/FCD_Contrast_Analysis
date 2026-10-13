#!/usr/bin/env python3
"""Measure saved pre-nnU-Net ratio intensities within each subject's brain mask."""
import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/nnUNet_raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/ratio_ranges"))
    args = parser.parse_args()
    definitions = [(103, "T1w_div_FLAIR"), (104, "FLAIR_div_T1w")]
    rows, sources, case_sets = [], [], []
    for dataset_id, channel in definitions:
        candidates = sorted(args.raw_root.glob(f"Dataset{dataset_id}_*"))
        if len(candidates) != 1:
            raise ValueError(f"Expected one Dataset{dataset_id} directory")
        dataset = candidates[0]
        meta = json.loads((dataset / "dataset.json").read_text())
        assert meta["channel_names"] == {"0": channel}
        paths = sorted((dataset / "imagesTr").glob("*_0000.nii.gz"))
        assert len(paths) == meta["numTraining"]
        ids = set()
        for path in paths:
            case = path.name.removesuffix("_0000.nii.gz")
            assert case not in ids
            ids.add(case)
            resolved = path.resolve(strict=True)
            mask_path = resolved.parent / "brain_mask.nii.gz"
            # The raw datasets link to per-subject ratio images beside their masks.
            # Never substitute the lesion label for the brain mask.
            img, mask_img = nib.load(resolved), nib.load(mask_path)
            assert img.shape == mask_img.shape, f"Shape mismatch: {case}"
            assert np.allclose(img.affine, mask_img.affine, rtol=0, atol=1e-5), f"Affine mismatch: {case}"
            mask_values = np.asarray(mask_img.dataobj)
            assert np.isfinite(mask_values).all(), f"Nonfinite mask: {case}"
            mask = mask_values > 0
            assert mask.any(), f"Empty brain mask: {case}"
            values = np.asarray(img.dataobj, dtype=np.float64)[mask]
            finite = values[np.isfinite(values)]
            assert len(finite) > 0, f"No finite brain intensities: {case}"
            group = "FCD" if case.startswith("FCD_") else "control" if case.startswith("CON_") else None
            assert group is not None
            rows.append({"case_id": case, "group": group, "ratio": channel,
                         "brain_voxels": len(values), "nonfinite_voxels": len(values) - len(finite),
                         "zero_voxels": int(np.count_nonzero(finite == 0)),
                         "negative_voxels": int(np.count_nonzero(finite < 0)),
                         "minimum": float(finite.min()), "maximum": float(finite.max())})
            sources.append({"case_id": case, "ratio": channel, "raw_image": str(path),
                            "brain_mask": str(mask_path.relative_to(Path.cwd())) if mask_path.is_relative_to(Path.cwd()) else str(mask_path)})
        case_sets.append(ids)
        print(f"Measured {channel}: {len(paths)} images", flush=True)
    assert case_sets[0] == case_sets[1]
    summaries = []
    for _, channel in definitions:
        for cohort in ("all", "FCD", "control"):
            selected = [r for r in rows if r["ratio"] == channel and (cohort == "all" or r["group"] == cohort)]
            low, high = min(r["minimum"] for r in selected), max(r["maximum"] for r in selected)
            summaries.append({"ratio": channel, "group": cohort, "subjects": len(selected),
                              "minimum": low, "maximum": high,
                              "minimum_case_ids": ";".join(r["case_id"] for r in selected if r["minimum"] == low),
                              "maximum_case_ids": ";".join(r["case_id"] for r in selected if r["maximum"] == high),
                              **{key: sum(r[key] for r in selected) for key in ("brain_voxels", "nonfinite_voxels", "zero_voxels", "negative_voxels")}})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("ratio_ranges_per_subject.csv", rows), ("ratio_ranges_summary.csv", summaries)):
        with (args.output_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(data)
    report = {"method": "Minimum and maximum over finite saved ratio intensities inside brain_mask > 0. Zeros and negative values retained. No normalization, clipping, resampling or inference performed.",
              "software": {"numpy": np.__version__, "nibabel": nib.__version__},
              "summaries": summaries, "sources": sources}
    (args.output_dir / "ratio_ranges_provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
