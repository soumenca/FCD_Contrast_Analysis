#!/usr/bin/env python3
"""Compare existing E1-E8 FCD results; no training or inference is performed."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, pstdev, stdev


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(values):
    return {"n": len(values), "mean": mean(values) if values else "",
            "sd_population": pstdev(values) if values else "",
            "sd_sample": stdev(values) if len(values) > 1 else ""}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("doc/analysis"))
    parser.add_argument("--splits", type=Path, default=Path("data/nnUNet_splits/splits_final.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/reviewer_analysis"))
    args = parser.parse_args()
    splits = json.loads(args.splits.read_text())
    assert len(splits) == 5, "Expected five folds"
    fold_map = {}
    universe = set(splits[0]["train"]) | set(splits[0]["val"])
    for i, fold in enumerate(splits):
        assert len(fold["train"]) == len(set(fold["train"]))
        assert len(fold["val"]) == len(set(fold["val"]))
        assert not set(fold["train"]) & set(fold["val"])
        assert set(fold["train"]) | set(fold["val"]) == universe
        for case in fold["val"]:
            assert case not in fold_map, f"Repeated validation case: {case}"
            fold_map[case] = f"fold_{i}"
    assert set(fold_map) == universe
    expected = {case for case in fold_map if case.startswith("FCD_")}
    datasets, sources = {}, []
    metrics = ("dice", "sensitivity", "precision")
    for e in range(1, 9):
        path = args.source_root / f"Dataset{100 + e}" / f"E{e}_segmentation_metrics_per_case_fcd.csv"
        rows = read_csv(path)
        indexed = {r["case_id"]: r for r in rows}
        assert len(indexed) == len(rows), f"Duplicate IDs: {path}"
        assert set(indexed) == expected, f"Unexpected case coverage: {path}"
        original = {r["case_id"]: r for r in read_csv(path.parent / "segmentation_metrics_per_case.csv") if r["group"] == "FCD"}
        assert indexed == original, f"FCD export differs from original per-case results: {path}"
        for case, row in indexed.items():
            assert row["group"] == "FCD" and row["fold"] == fold_map[case]
            for metric in metrics:
                value = float(row[metric])
                assert math.isfinite(value) and 0 <= value <= 1
            tp, fp, fn = (float(row[k]) for k in ("tp", "fp", "fn"))
            assert math.isclose(float(row["dice"]), 2 * tp / (2 * tp + fp + fn), abs_tol=1e-12)
        published = next(r for r in read_csv(path.parent / "segmentation_metrics_paper_summary.csv") if r["group"] == "FCD")
        for metric in metrics:
            stats = summarize([float(r[metric]) for r in rows])
            assert math.isclose(stats["mean"], float(published[f"{metric}_mean"]), abs_tol=1e-12)
            assert math.isclose(stats["sd_population"], float(published[f"{metric}_std"]), abs_tol=1e-12)
        datasets[f"E{e}"] = indexed
        sources.append({"configuration": f"E{e}", "source": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    for case in expected:
        assert len({float(data[case]["gt_voxels"]) for data in datasets.values()}) == 1, f"Ground-truth size differs: {case}"
    common = {case for case in expected if all(float(data[case]["dice"]) > 0 for data in datasets.values())}
    wide, long = [], []
    for case in sorted(expected):
        row = {"case_id": case, "fold": fold_map[case]}
        row.update({f"{e}_dice": data[case]["dice"] for e, data in datasets.items()})
        row["detected_all_eight"] = int(case in common)
        wide.append(row)
        for e, data in datasets.items():
            long.append({"case_id": case, "fold": fold_map[case], "configuration": e,
                         **{m: data[case][m] for m in metrics}, "detected_all_eight": int(case in common)})
    common_summary, fold_results, fold_summary = [], [], []
    for e, data in datasets.items():
        common_summary.append({"configuration": e, **summarize([float(data[c]["dice"]) for c in sorted(common)])})
        for subset, cases in (("all_fcd", expected), ("detected_all_eight", common)):
            for metric in metrics:
                means = []
                for i in range(5):
                    selected = [c for c in sorted(cases) if fold_map[c] == f"fold_{i}"]
                    stats = summarize([float(data[c][metric]) for c in selected])
                    fold_results.append({"configuration": e, "subset": subset, "metric": metric,
                                         "fold": f"fold_{i}", **stats})
                    if selected:
                        means.append(stats["mean"])
                fold_summary.append({"configuration": e, "subset": subset, "metric": metric,
                                     "n_subjects": len(cases),
                                     **{f"fold_means_{k}": v for k, v in summarize(means).items()}})
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    outputs = {"subject_level_dice_E1_E8.csv": wide,
               "subject_level_metrics_E1_E8.csv": long,
               "common_detected_dice_summary.csv": common_summary,
               "fold_wise_metrics.csv": fold_results,
               "variability_across_fold_means.csv": fold_summary}
    for name, rows in outputs.items():
        write_csv(out / name, rows)
        assert len(read_csv(out / name)) == len(rows)
    manifest = {"fcd_subjects": len(expected), "common_detected_subjects": len(common),
                "common_detected_by_fold": {f"fold_{i}": sum(fold_map[c] == f"fold_{i}" for c in common) for i in range(5)},
                "sources": sources, "splits": str(args.splits),
                "splits_sha256": hashlib.sha256(args.splits.read_bytes()).hexdigest(),
                "validation": "Unique IDs, complete case coverage, split agreement, Dice from confusion counts, and existing paper-summary means/SDs verified.",
                "outputs_sha256": {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in outputs}}
    (out / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"fcd_subjects": len(expected), "common_detected_subjects": len(common),
                      "common_detected_by_fold": manifest["common_detected_by_fold"],
                      "common_dice_summary": common_summary}, indent=2))


if __name__ == "__main__":
    main()
