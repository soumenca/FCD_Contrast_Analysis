#!/usr/bin/env python3
"""Extract one bottleneck feature vector per subject from a trained nnU-Net v2 model."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from torch._dynamo import OptimizedModule


CASE_FILE_PATTERN = re.compile(r"^(?P<case_id>.+)_(?P<channel>\d{4})\.nii\.gz$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract nnU-Net encoder bottleneck features for subjects, apply global "
            "average pooling, and save one vector per subject."
        )
    )
    parser.add_argument(
        "--model-folder",
        type=Path,
        required=True,
        help=(
            "nnU-Net training output directory containing dataset.json, plans.json, "
            "and fold_* subfolders."
        ),
    )
    parser.add_argument(
        "--raw-dataset-dir",
        type=Path,
        required=True,
        help="nnU-Net raw dataset directory containing imagesTr/ and labelsTr/.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        help="Which trained fold to load for feature extraction.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default="checkpoint_final.pth",
        help="Checkpoint filename inside fold_<n>/ to load.",
    )
    parser.add_argument(
        "--subset",
        choices=("all", "train", "val"),
        default="all",
        help="Which case subset to extract based on splits_final.json.",
    )
    parser.add_argument(
        "--splits-file",
        type=Path,
        default=None,
        help="Path to splits_final.json. Required when subset is train or val.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Where to save the pooled subject-level feature table as CSV.",
    )
    parser.add_argument(
        "--output-npy",
        type=Path,
        default=None,
        help="Optional .npy file for the feature matrix only.",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=None,
        help="Optional JSON file for extraction metadata.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=("cuda", "cpu", "mps"),
        help="Torch device to use for feature extraction.",
    )
    parser.add_argument(
        "--tile-step-size",
        type=float,
        default=0.5,
        help="Sliding-window step size for patch extraction.",
    )
    return parser.parse_args()


def unwrap_network(network: torch.nn.Module) -> torch.nn.Module:
    if isinstance(network, OptimizedModule):
        network = network._orig_mod
    if hasattr(network, "module"):
        network = network.module
    return network


def get_case_channel_map(images_tr_dir: Path) -> dict[str, list[str]]:
    case_map: dict[str, list[tuple[int, str]]] = {}
    for path in sorted(images_tr_dir.glob("*.nii.gz")):
        match = CASE_FILE_PATTERN.match(path.name)
        if match is None:
            continue
        case_id = match.group("case_id")
        channel_index = int(match.group("channel"))
        case_map.setdefault(case_id, []).append((channel_index, str(path.resolve())))

    if not case_map:
        raise FileNotFoundError(f"No nnU-Net training images found in {images_tr_dir}")

    return {
        case_id: [filename for _, filename in sorted(entries)]
        for case_id, entries in sorted(case_map.items())
    }


def load_subset_case_ids(
    all_case_ids: Iterable[str], subset: str, splits_file: Path | None, fold: int
) -> list[str]:
    all_case_ids = sorted(all_case_ids)
    if subset == "all":
        return all_case_ids

    if splits_file is None:
        raise ValueError("--splits-file is required when --subset is train or val")

    splits = json.loads(splits_file.read_text())
    if fold < 0 or fold >= len(splits):
        raise ValueError(f"Requested fold {fold} is out of range for {splits_file}")

    chosen = splits[fold][subset]
    missing = sorted(set(chosen) - set(all_case_ids))
    if missing:
        raise ValueError(
            f"Some case IDs from {splits_file} are missing in {subset}: {missing[:10]}"
        )
    return sorted(chosen)


def case_label(case_id: str) -> str:
    if case_id.startswith("FCD_"):
        return "FCD"
    if case_id.startswith("CON_"):
        return "control"
    return "unknown"


def preprocess_case(
    predictor: nnUNetPredictor,
    channel_files: list[str],
) -> tuple[torch.Tensor, dict]:
    preprocessor = predictor.configuration_manager.preprocessor_class(
        verbose=predictor.verbose_preprocessing
    )
    data, _seg, data_properties = preprocessor.run_case(
        channel_files,
        seg_file=None,
        plans_manager=predictor.plans_manager,
        configuration_manager=predictor.configuration_manager,
        dataset_json=predictor.dataset_json,
    )
    return torch.from_numpy(data).to(dtype=torch.float32), data_properties


def extract_subject_vector(
    predictor: nnUNetPredictor,
    data: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, int, tuple[int, ...]]:
    network = unwrap_network(predictor.network).to(device)
    network.eval()

    padded = pad_nd_image(
        data,
        new_shape=tuple(predictor.configuration_manager.patch_size),
        mode="constant",
        kwargs={"value": 0},
        return_slicer=False,
        shape_must_be_divisible_by=None,
    )
    if not isinstance(padded, torch.Tensor):
        padded = torch.from_numpy(padded)

    slicers = predictor._internal_get_sliding_window_slicers(tuple(padded.shape[1:]))
    pooled_vectors = []
    bottleneck_shape = None

    with torch.inference_mode():
        for slicer in slicers:
            patch = padded[slicer].unsqueeze(0).to(device, non_blocking=True)
            encoder_outputs = network.encoder(patch)
            bottleneck = encoder_outputs[-1] if isinstance(encoder_outputs, (list, tuple)) else encoder_outputs
            if bottleneck_shape is None:
                bottleneck_shape = tuple(int(i) for i in bottleneck.shape[1:])

            pooled = bottleneck.mean(dim=tuple(range(2, bottleneck.ndim)))
            pooled_vectors.append(pooled.squeeze(0).detach().cpu())

    if not pooled_vectors:
        raise RuntimeError("No sliding-window patches were generated for this subject")

    subject_vector = torch.stack(pooled_vectors, dim=0).mean(dim=0).numpy()
    return subject_vector, len(pooled_vectors), bottleneck_shape


def main() -> int:
    args = parse_args()

    model_folder = args.model_folder.resolve()
    raw_dataset_dir = args.raw_dataset_dir.resolve()
    output_csv = args.output_csv.resolve()
    output_npy = args.output_npy.resolve() if args.output_npy is not None else None
    metadata_json = (
        args.metadata_json.resolve() if args.metadata_json is not None else None
    )
    splits_file = args.splits_file.resolve() if args.splits_file is not None else None

    if not model_folder.exists():
        raise FileNotFoundError(f"Model folder does not exist: {model_folder}")
    if not raw_dataset_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory does not exist: {raw_dataset_dir}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    predictor = nnUNetPredictor(
        tile_step_size=args.tile_step_size,
        use_gaussian=False,
        use_mirroring=False,
        perform_everything_on_device=False,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_folder),
        use_folds=(args.fold,),
        checkpoint_name=args.checkpoint_name,
    )

    case_channel_map = get_case_channel_map(raw_dataset_dir / "imagesTr")
    selected_case_ids = load_subset_case_ids(
        case_channel_map.keys(),
        args.subset,
        splits_file,
        args.fold,
    )

    rows = []
    feature_matrix = []

    for case_id in selected_case_ids:
        channel_files = case_channel_map[case_id]
        data, data_properties = preprocess_case(
            predictor,
            channel_files,
        )
        subject_vector, num_tiles, bottleneck_shape = extract_subject_vector(
            predictor,
            data,
            device,
        )

        feature_matrix.append(subject_vector)
        row = {
            "case_id": case_id,
            "group": case_label(case_id),
            "subset": args.subset,
            "fold": args.fold,
            "num_input_channels": len(channel_files),
            "num_tiles": num_tiles,
            "bottleneck_channels": int(subject_vector.shape[0]),
            "bottleneck_feature_map_shape": "x".join(str(i) for i in bottleneck_shape),
            "preprocessed_shape": "x".join(str(int(i)) for i in data.shape),
            "spacing": data_properties.get("spacing"),
        }
        for idx, value in enumerate(subject_vector):
            row[f"feat_{idx:04d}"] = float(value)
        rows.append(row)
        print(
            f"Extracted {case_id}: {subject_vector.shape[0]} features from {num_tiles} patches"
        )

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved pooled subject features to {output_csv}")

    if output_npy is not None:
        output_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_npy, np.stack(feature_matrix, axis=0))
        print(f"Saved feature matrix to {output_npy}")

    if metadata_json is not None:
        metadata = {
            "model_folder": str(model_folder),
            "raw_dataset_dir": str(raw_dataset_dir),
            "fold": args.fold,
            "subset": args.subset,
            "num_subjects": len(selected_case_ids),
            "checkpoint_name": args.checkpoint_name,
            "device": args.device,
            "tile_step_size": args.tile_step_size,
        }
        metadata_json.parent.mkdir(parents=True, exist_ok=True)
        metadata_json.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Saved metadata to {metadata_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
