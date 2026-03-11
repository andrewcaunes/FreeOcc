from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from FreeOcc.eval.occ_metrics import IoU, Metric_mIoU

GT_ROOT_BY_DATASET = {
    "nuscenes": Path("data/occ3d_nuscenes"),
    "waymo": Path("data/occ3d_waymo"),
}


def _configure_logging() -> None:
    logging.basicConfig(format='[%(module)s | l.%(lineno)d] %(message)s')
    logging.getLogger().setLevel(logging.INFO)


def _parse_scene_name(exp_folder: Path, scene_name: str | None) -> str:
    if scene_name is not None:
        if scene_name.startswith("scene-"):
            return scene_name
        return f"scene-{str(int(scene_name)).zfill(4)}"

    if exp_folder.name.startswith("scene-"):
        return exp_folder.name

    raise ValueError(
        "Could not infer scene name from --exp_folder. "
        "Please pass --scene_name (e.g. scene-0001)."
    )


def _load_or_build_nuscenes_scene_order(
    args: argparse.Namespace,
    scene_name: str,
) -> dict[str, int]:
    if args.dataset_name != "nuscenes":
        return {}

    sample_order_index_path = Path(args.sample_order_index_path)
    if sample_order_index_path.exists():
        with open(sample_order_index_path, "r") as file_handle:
            cached_data = json.load(file_handle)
        scene_to_order = cached_data.get("scene_to_token_order", {})
        order_map = scene_to_order.get(scene_name, {})
        logging.info("Loaded NuScenes sample-order cache from %s", sample_order_index_path)
        return {token: int(idx) for token, idx in order_map.items()}

    metadata_root = Path(args.nuscenes_path) / args.nuscenes_version
    sample_json_path = metadata_root / "sample.json"
    scene_json_path = metadata_root / "scene.json"
    if not sample_json_path.exists() or not scene_json_path.exists():
        logging.warning("NuScenes metadata not found for ordering, using lexicographic sample-token order.")
        return {}

    with open(sample_json_path, "r") as file_handle:
        samples = json.load(file_handle)
    with open(scene_json_path, "r") as file_handle:
        scenes = json.load(file_handle)

    scene_record = next((scene for scene in scenes if scene["name"] == scene_name), None)
    if scene_record is None:
        logging.warning("Scene %s not found in NuScenes metadata. Using lexicographic order.", scene_name)
        return {}

    sample_by_token = {sample["token"]: sample for sample in samples}
    order_map: dict[str, int] = {}
    sample_token = scene_record["first_sample_token"]
    sample_idx = 0
    while sample_token:
        order_map[sample_token] = sample_idx
        sample_idx += 1
        sample_token = sample_by_token[sample_token]["next"]

    try:
        sample_order_index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sample_order_index_path, "w") as file_handle:
            json.dump(
                {
                    "nuscenes_version": args.nuscenes_version,
                    "scene_to_token_order": {scene_name: order_map},
                },
                file_handle,
            )
    except OSError as exception:
        logging.warning("Could not write sample-order cache at %s (%s)", sample_order_index_path, exception)

    return order_map


def _discover_sample_tokens(exp_folder: Path, gt_scene_root: Path, occ_grid_filename: str) -> list[str]:
    if gt_scene_root.exists():
        gt_tokens = sorted(path.name for path in gt_scene_root.iterdir() if path.is_dir())
        if gt_tokens:
            return gt_tokens

    return sorted(
        path.name
        for path in exp_folder.iterdir()
        if path.is_dir() and (path / occ_grid_filename).exists()
    )


def _sort_sample_tokens(sample_tokens: list[str], scene_order: dict[str, int]) -> list[str]:
    if not scene_order:
        return sample_tokens
    return sorted(
        sample_tokens,
        key=lambda token: (token not in scene_order, scene_order.get(token, 10**9), token),
    )


def _empty_result_row(
    scene_name: str,
    sample_token: str,
    status: str,
    sem_class_names: list[str],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "scene_name": scene_name,
        "sample_token": sample_token,
        "status": status,
        "mIoU": np.nan,
        "IoU_binary_occupied": np.nan,
        "IoU_binary_free": np.nan,
    }
    for class_name in sem_class_names:
        row[f"IoU_{class_name}"] = np.nan
    return row


def _evaluate_sample(
    scene_name: str,
    sample_token: str,
    pred_grid_path: Path,
    gt_path: Path,
    num_classes: int,
    with_others: bool,
    sem_class_names: list[str],
) -> dict[str, Any]:
    if not pred_grid_path.exists():
        return _empty_result_row(scene_name, sample_token, "missing_pred", sem_class_names)
    if not gt_path.exists():
        return _empty_result_row(scene_name, sample_token, "missing_gt", sem_class_names)

    pred_grid = np.load(pred_grid_path)
    gt_data = np.load(gt_path)
    gt_semantics = gt_data["semantics"]
    mask_camera = gt_data["mask_camera"].astype(bool)
    mask_lidar = gt_data["mask_lidar"].astype(bool)

    if not with_others:
        other_mask = (gt_semantics == 0) | (gt_semantics == 12)
        mask_camera = mask_camera & (~other_mask)

    sem_metric = Metric_mIoU(
        num_classes=num_classes,
        use_lidar_mask=False,
        use_image_mask=True,
        with_others=with_others,
    )
    occ_metric = IoU(use_image_mask=True)
    sem_metric.add_batch(pred_grid, gt_semantics, mask_lidar, mask_camera)
    occ_metric.add_batch(pred_grid, gt_semantics, mask_lidar, mask_camera)

    class_names, iou_per_class, mean_iou, _ = sem_metric.count_miou()
    _, occ_iou, _ = occ_metric.count_miou()

    row: dict[str, Any] = {
        "scene_name": scene_name,
        "sample_token": sample_token,
        "status": "ok",
        "mIoU": float(mean_iou),
        "IoU_binary_occupied": float(occ_iou[0]),
        "IoU_binary_free": float(occ_iou[1]),
    }
    for class_name, class_iou in zip(class_names, iou_per_class):
        row[f"IoU_{class_name}"] = float(class_iou)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate occupancy grids per sample for one scene folder and save results_per_sample.csv."
    )
    parser.add_argument("--dataset_name", type=str, default="nuscenes", choices=["nuscenes", "waymo"])
    parser.add_argument(
        "--exp_folder",
        type=str,
        required=True,
        help="Full path to scene folder containing sample-token subfolders.",
    )
    parser.add_argument("--scene_name", type=str, default=None, help="Optional scene name override (e.g. scene-0001).")
    parser.add_argument("--dataset_path", type=str, default=None, help="GT root containing gts/.")
    parser.add_argument("--occ_grid_filename", type=str, default="occ_grid_occ3d_nuscenes.npy")
    parser.add_argument("--num_classes", type=int, default=18)
    parser.add_argument("--with_others", action="store_true", default=False)
    parser.add_argument("--nuscenes_path", type=str, default="data/nuscenes")
    parser.add_argument("--nuscenes_version", type=str, default="v1.0-trainval")
    parser.add_argument("--sample_order_index_path", type=str, default=None)
    args = parser.parse_args()

    if args.dataset_path is None:
        args.dataset_path = str(GT_ROOT_BY_DATASET[args.dataset_name])
    if args.sample_order_index_path is None:
        args.sample_order_index_path = str(Path(args.exp_folder).parent / "sample_order_index_nuscenes.json")
    return args


def main(args: argparse.Namespace) -> None:
    _configure_logging()
    exp_folder = Path(args.exp_folder)
    if not exp_folder.exists():
        raise FileNotFoundError(f"--exp_folder not found: {exp_folder}")
    if not exp_folder.is_dir():
        raise ValueError(f"--exp_folder must be a directory: {exp_folder}")

    scene_name = _parse_scene_name(exp_folder, args.scene_name)
    gt_scene_root = Path(args.dataset_path) / "gts" / scene_name

    sample_tokens = _discover_sample_tokens(exp_folder, gt_scene_root, args.occ_grid_filename)
    if not sample_tokens:
        raise ValueError(f"No samples found under {exp_folder}")

    scene_order = _load_or_build_nuscenes_scene_order(args, scene_name)
    sample_tokens = _sort_sample_tokens(sample_tokens, scene_order)

    sem_class_names = Metric_mIoU(
        num_classes=args.num_classes,
        use_lidar_mask=False,
        use_image_mask=True,
        with_others=args.with_others,
    ).class_names

    rows: list[dict[str, Any]] = []
    for sample_token in sample_tokens:
        pred_grid_path = exp_folder / sample_token / args.occ_grid_filename
        gt_path = gt_scene_root / sample_token / "labels.npz"
        rows.append(
            _evaluate_sample(
                scene_name=scene_name,
                sample_token=sample_token,
                pred_grid_path=pred_grid_path,
                gt_path=gt_path,
                num_classes=args.num_classes,
                with_others=args.with_others,
                sem_class_names=sem_class_names,
            )
        )

    columns = [
        "scene_name",
        "sample_token",
        "status",
        "mIoU",
        "IoU_binary_occupied",
        "IoU_binary_free",
    ] + [f"IoU_{class_name}" for class_name in sem_class_names]
    results = pd.DataFrame(rows).reindex(columns=columns)
    output_csv = exp_folder / "results_per_sample.csv"
    results.to_csv(output_csv, index=False)

    num_ok = int((results["status"] == "ok").sum())
    mean_miou = results.loc[results["status"] == "ok", "mIoU"].mean()
    logging.info(
        "Saved %s (%d/%d valid samples, mean mIoU=%.4f)",
        output_csv,
        num_ok,
        len(results),
        float(mean_miou) if pd.notna(mean_miou) else float("nan"),
    )


if __name__ == "__main__":
    main(parse_args())
