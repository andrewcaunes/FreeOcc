
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from FreeOcc.eval.occ_metrics import IoU, Metric_mIoU
from FreeOcc.eval.ray_metrics import main_rayiou, main_raypq
from FreeOcc.tools.eval_ray_metrics import (
    build_scene_infos_context,
    build_scene_infos_for_eval,
    build_scene_infos_sparseocc_exact,
    build_token_to_scene_from_all_occ_npz,
    build_token_maps,
    collect_eval_data,
    collect_eval_data_sparseocc_exact,
    get_results_global_path,
    load_semantic_prediction,
    occ_class_names_occ3d,
    reorder_global_results_columns,
    resolve_occ_gt_root,
    resolve_prediction_source,
)

logging.basicConfig(format='[%(module)s | l.%(lineno)d] %(message)s')
logging.getLogger().setLevel(logging.INFO)


def evaluate_iou_metrics(
    args,
    pred_source,
    sem_path_by_token,
    token_to_sem_gt_path,
    token_to_scene,
):
    sem_metric = Metric_mIoU(
        num_classes=18,
        use_lidar_mask=False,
        use_image_mask=True,
        with_others=args.eval_with_others,
    )
    occ_metric = IoU(use_image_mask=True)

    pred_tokens = set(sem_path_by_token.keys())
    eval_tokens = sorted(set(token_to_sem_gt_path.keys()) & pred_tokens)
    evaluated_tokens = []

    for token in eval_tokens:
        gt_sem_path = token_to_sem_gt_path[token]
        with np.load(gt_sem_path, allow_pickle=True) as gt_data:
            if "semantics" not in gt_data.files:
                continue
            if "mask_camera" not in gt_data.files:
                continue
            if "mask_lidar" not in gt_data.files:
                continue
            semantics_gt = np.reshape(gt_data["semantics"], [200, 200, 16]).astype(np.uint8)
            mask_camera = gt_data["mask_camera"].astype(bool)
            mask_lidar = gt_data["mask_lidar"].astype(bool)

        sem_path = sem_path_by_token[token]
        semantics_pred = load_semantic_prediction(
            pred_source=pred_source,
            sem_path=sem_path,
            pred_sem_key=args.pred_sem_key,
        )
        semantics_pred = np.reshape(semantics_pred, [200, 200, 16]).astype(np.uint8)

        if not args.eval_with_others:
            other_mask = (semantics_gt == 0) | (semantics_gt == 12)
            mask_camera = mask_camera & (~other_mask)

        sem_metric.add_batch(
            semantics_pred=semantics_pred,
            semantics_gt=semantics_gt,
            mask_lidar=mask_lidar,
            mask_camera=mask_camera,
        )
        occ_metric.add_batch(
            semantics_pred,
            semantics_gt,
            mask_lidar,
            mask_camera,
        )
        evaluated_tokens.append(token)

    if sem_metric.cnt == 0:
        raise RuntimeError("No valid samples were loaded for IoU evaluation.")

    class_names, iou_per_class, final_miou, num_samples = sem_metric.count_miou()
    _, occ_iou, _ = occ_metric.count_miou()

    evaluated_scenes = set()
    for token in evaluated_tokens:
        if token in token_to_scene:
            evaluated_scenes.add(token_to_scene[token])

    iou_results = {
        "num_scenes": int(len(evaluated_scenes)),
        "num_samples": int(num_samples),
        "mIoU": float(final_miou),
        "IoU_occupied": float(occ_iou[0]),
        "IoU_free": float(occ_iou[1]),
    }
    for class_name, class_iou in zip(class_names, iou_per_class):
        iou_results[f"IoU_{class_name}"] = float(class_iou)
    return iou_results, evaluated_tokens


def evaluate_ray_metrics(
    args,
    pred_source,
    sem_path_by_token,
    inst_path_by_token,
    token_to_sem_gt_path,
    token_to_scene,
    token_to_inst_gt_path,
    eval_tokens,
):
    if args.sparseocc_exact_mode:
        occ_gt_root_sem = resolve_occ_gt_root(Path(args.occ3d_root), "labels.npz")
        token_to_scene_all_npz = build_token_to_scene_from_all_occ_npz(occ_gt_root_sem)
        if len(token_to_scene_all_npz) == 0:
            token_to_scene_all_npz = token_to_scene

        scene_infos_context = build_scene_infos_sparseocc_exact(
            args=args,
            token_to_scene=token_to_scene_all_npz,
            token_to_sem_gt_path=token_to_sem_gt_path,
        )

        if not args.sparseocc_exact_require_full_predictions:
            num_infos_before_filter = len(scene_infos_context)
            valid_tokens = set(sem_path_by_token.keys())
            if args.compute_raypq:
                valid_tokens = valid_tokens & set(inst_path_by_token.keys())
            scene_infos_context = [
                info for info in scene_infos_context
                if info["token"] in valid_tokens
            ]
            logging.warning(
                "SparseOcc exact mode with partial predictions: using overlap tokens "
                "(before=%d after=%d).",
                num_infos_before_filter,
                len(scene_infos_context),
            )
            if len(scene_infos_context) == 0:
                raise RuntimeError(
                    "SparseOcc exact mode overlap is empty after filtering infos by available predictions."
                )

        logging.info(
            "ray setup (sparseocc_exact_mode): context_frames=%d",
            len(scene_infos_context),
        )

        eval_data = collect_eval_data_sparseocc_exact(
            args=args,
            scene_infos_context=scene_infos_context,
            token_to_sem_gt_path=token_to_sem_gt_path,
            sem_path_by_token=sem_path_by_token,
            token_to_inst_gt_path=token_to_inst_gt_path,
            inst_path_by_token=inst_path_by_token,
            pred_source=pred_source,
        )
    else:
        scene_infos_eval, info_by_token_eval, eval_tokens = build_scene_infos_for_eval(
            args=args,
            eval_tokens=eval_tokens,
        )
        scene_infos_context = build_scene_infos_context(
            args=args,
            eval_tokens=eval_tokens,
            info_by_token_eval=info_by_token_eval,
        )

        logging.info(
            "ray setup: tokens_with_pose=%d context_frames=%d",
            len(scene_infos_eval),
            len(scene_infos_context),
        )

        eval_data = collect_eval_data(
            args=args,
            eval_tokens=eval_tokens,
            scene_infos_context=scene_infos_context,
            token_to_sem_gt_path=token_to_sem_gt_path,
            sem_path_by_token=sem_path_by_token,
            token_to_inst_gt_path=token_to_inst_gt_path,
            inst_path_by_token=inst_path_by_token,
            pred_source=pred_source,
        )

    if len(eval_data["sem_preds"]) == 0:
        raise RuntimeError("No valid samples were loaded for ray evaluation.")

    rayiou_results = main_rayiou(
        eval_data["sem_preds"],
        eval_data["sem_gts"],
        eval_data["lidar_origins"],
        occ_class_names=occ_class_names_occ3d,
    )
    ray_results = dict(rayiou_results)
    ray_results["num_ray_samples"] = len(eval_data["sem_preds"])

    if args.compute_raypq:
        if len(eval_data["sem_preds_pq"]) == 0:
            logging.warning(
                "RayPQ requested but no valid panoptic samples were found. "
                "missing_gt_instances=%d missing_pred_instances=%d. Returning RayPQ=0.",
                eval_data["pq_missing_gt_instances"],
                eval_data["pq_missing_pred_instances"],
            )
            raypq_results = {
                "RayPQ": 0.0,
                "RayPQ@1": 0.0,
                "RayPQ@2": 0.0,
                "RayPQ@4": 0.0,
                "num_raypq_samples": 0,
                "num_raypq_missing_gt_instances": eval_data["pq_missing_gt_instances"],
                "num_raypq_missing_pred_instances": eval_data["pq_missing_pred_instances"],
            }
            if args.thing_only:
                raypq_results["RayPQ_thing"] = 0.0
                raypq_results["RayPQ_thing@1"] = 0.0
                raypq_results["RayPQ_thing@2"] = 0.0
                raypq_results["RayPQ_thing@4"] = 0.0
        else:
            raypq_results = main_raypq(
                eval_data["sem_preds_pq"],
                eval_data["sem_gts_pq"],
                eval_data["inst_preds"],
                eval_data["inst_gts"],
                eval_data["lidar_origins_pq"],
                occ_class_names=occ_class_names_occ3d,
                use_dynamic_id_offset=not args.sparseocc_exact_mode,
                include_per_class=True,
                thing_only=args.thing_only,
            )
            raypq_results["num_raypq_samples"] = len(eval_data["sem_preds_pq"])
            raypq_results["num_raypq_missing_gt_instances"] = eval_data["pq_missing_gt_instances"]
            raypq_results["num_raypq_missing_pred_instances"] = eval_data["pq_missing_pred_instances"]
        ray_results.update(raypq_results)

    return ray_results


def is_empty_value(value):
    if pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def update_results_global_with_all_metrics(args, metrics):
    if args.exp_path is None:
        return

    results_global_path = get_results_global_path(args.exp_path, args.results_global_filename)
    exp_name = Path(args.exp_path).name

    row_data = {
        "timestamp": pd.Timestamp.now().round("1s"),
        "exp_name": exp_name,
        "scene_name": "__all__",
        "scope": "all_scenes_all_points",
    }
    row_data.update(metrics)

    file_exists = results_global_path.exists()
    if file_exists:
        results_df = pd.read_csv(results_global_path, index_col=0)
    else:
        logging.info("Global results file does not exist, creating %s", results_global_path)
        results_df = pd.DataFrame()

    for key in row_data.keys():
        if key not in results_df.columns:
            results_df[key] = np.nan
    if "RayIoU" not in results_df.columns:
        results_df["RayIoU"] = np.nan
    if "RayPQ" not in results_df.columns:
        results_df["RayPQ"] = np.nan

    if len(results_df) == 0:
        row_mask = pd.Series(dtype=bool)
    else:
        exp_mask = results_df["exp_name"].astype(str) == exp_name
        if "scope" in results_df.columns:
            row_mask = exp_mask & (results_df["scope"].astype(str) == "all_scenes_all_points")
        elif "scene_name" in results_df.columns:
            row_mask = exp_mask & (results_df["scene_name"].astype(str) == "__all__")
        else:
            row_mask = exp_mask

    if row_mask.any():
        row_indexes = results_df.index[row_mask]
        if "timestamp" in results_df.columns:
            timestamps = pd.to_datetime(results_df.loc[row_indexes, "timestamp"], errors="coerce")
            if timestamps.notna().any():
                latest_position = timestamps.fillna(pd.Timestamp.min).to_numpy().argmax()
                row_index = row_indexes[latest_position]
            else:
                row_index = row_indexes[-1]
        else:
            row_index = row_indexes[-1]

        for key, value in row_data.items():
            is_ray_key = key.startswith("Ray") or key.startswith("num_ray")
            if args.overwrite or is_ray_key:
                results_df.at[row_index, key] = value
                continue
            current_value = results_df.at[row_index, key]
            if is_empty_value(current_value):
                results_df.at[row_index, key] = value
        action = "Updated"
    else:
        if len(results_df) == 0:
            results_df = pd.DataFrame([row_data])
        else:
            results_df = pd.concat([results_df, pd.DataFrame([row_data])], ignore_index=True)
        action = "Added"

    results_df = reorder_global_results_columns(results_df)

    if file_exists:
        backup_path = results_global_path.with_name(results_global_path.stem + "_backup.csv")
        with open(results_global_path, "rb") as source_file, open(backup_path, "wb") as backup_file:
            backup_file.write(source_file.read())
    results_global_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_global_path)
    logging.info(
        "%s global results row for exp_name=%s in %s",
        action,
        exp_name,
        results_global_path,
    )


def get_output_json_path(args):
    if args.output_json is not None:
        return Path(args.output_json)
    if args.exp_path is not None:
        return Path(args.exp_path) / "final_eval_results_all_scenes.json"
    return Path(args.pred_dir) / "all_metrics.json"


def main(args):
    logging.info("args = %s", args)
    if args.thing_only and not args.compute_raypq:
        logging.warning("--thing_only requires --compute_raypq. Ignoring thing-only RayPQ outputs.")

    pred_source, sem_path_by_token, inst_path_by_token = resolve_prediction_source(
        pred_dir=args.pred_dir,
        exp_path=args.exp_path,
        exp_sem_filename=args.exp_sem_filename,
        exp_inst_filename=args.exp_inst_filename,
    )

    occ_gt_root_sem = resolve_occ_gt_root(Path(args.occ3d_root), "labels.npz")
    token_to_sem_gt_path, token_to_scene = build_token_maps(occ_gt_root_sem, "labels.npz")

    token_to_inst_gt_path = {}
    if args.compute_raypq:
        occ_gt_root_inst = resolve_occ_gt_root(Path(args.occ3d_root), args.gt_label_name)
        token_to_inst_gt_path, _ = build_token_maps(occ_gt_root_inst, args.gt_label_name)

    iou_results, evaluated_tokens = evaluate_iou_metrics(
        args=args,
        pred_source=pred_source,
        sem_path_by_token=sem_path_by_token,
        token_to_sem_gt_path=token_to_sem_gt_path,
        token_to_scene=token_to_scene,
    )
    logging.info("IoU results: mIoU=%.4f num_samples=%d", iou_results["mIoU"], iou_results["num_samples"])

    ray_results = evaluate_ray_metrics(
        args=args,
        pred_source=pred_source,
        sem_path_by_token=sem_path_by_token,
        inst_path_by_token=inst_path_by_token,
        token_to_sem_gt_path=token_to_sem_gt_path,
        token_to_scene=token_to_scene,
        token_to_inst_gt_path=token_to_inst_gt_path,
        eval_tokens=evaluated_tokens,
    )
    logging.info("RayIoU results: %s", {k: v for k, v in ray_results.items() if str(k).startswith("RayIoU")})
    if args.compute_raypq:
        logging.info("RayPQ results: %s", {k: v for k, v in ray_results.items() if str(k).startswith("RayPQ")})

    all_results = {}
    if args.exp_path is not None:
        all_results["exp_name"] = Path(args.exp_path).name
        all_results["scope"] = "all_scenes_all_points"
    all_results.update(iou_results)
    all_results.update(ray_results)

    output_json_path = get_output_json_path(args)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w") as file_handle:
        json.dump(all_results, file_handle, indent=2)
    logging.info("Saved results to %s", output_json_path)

    update_results_global_with_all_metrics(args, all_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recompute mIoU/IoUs and ray metrics from predictions.",
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--pred_dir",
        type=str,
        help="Directory with <token>.npz predictions.",
    )
    source_group.add_argument(
        "--exp_path",
        type=str,
        help="Experiment output directory with scene/token/occ grids.",
    )
    parser.add_argument(
        "--pred_sem_key",
        type=str,
        default="pred",
        help="Semantic key in prediction npz.",
    )
    parser.add_argument(
        "--pred_inst_key",
        type=str,
        default="pano_inst",
        help="Instance key in prediction npz for RayPQ.",
    )
    parser.add_argument(
        "--exp_sem_filename",
        type=str,
        default="occ_grid_occ3d_nuscenes.npy",
        help="Semantic grid filename under exp scene/token folders.",
    )
    parser.add_argument(
        "--exp_inst_filename",
        type=str,
        default="occ_grid_instances.npy",
        help="Instance grid filename under exp scene/token folders for RayPQ.",
    )
    parser.add_argument("--compute_raypq", action="store_true", default=False)
    parser.add_argument(
        "--thing_only",
        action="store_true",
        default=False,
        help="Also compute and output thing-only RayPQ aggregates (RayPQ_thing, RayPQ_thing@1/2/4).",
    )
    parser.add_argument("--eval_with_others", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--nuscenes_root", type=str, default="data/nuscenes")
    parser.add_argument(
        "--nuscenes_split",
        type=str,
        default="val",
        choices=["train", "val", "trainval", "all", "test"],
    )
    parser.add_argument("--occ3d_root", type=str, default="data/occ3d_nuscenes")
    parser.add_argument("--gt_label_name", type=str, default="labels_pano.npz")
    parser.add_argument(
        "--sparseocc_exact_mode",
        action="store_true",
        default=False,
        help=(
            "Use SparseOcc-style PKL infos and strict ordering/path assumptions "
            "for ray metrics."
        ),
    )
    parser.add_argument(
        "--sparseocc_infos_root",
        type=str,
        default="data/nuscenes/nuscenes_infos_with_sweeps",
        help="Folder containing nuscenes_infos_*_sweep.pkl files from SparseOcc.",
    )
    parser.add_argument(
        "--sparseocc_infos_pkl",
        type=str,
        default=None,
        help="Optional explicit SparseOcc infos pkl path. Overrides --sparseocc_infos_root.",
    )
    parser.add_argument(
        "--sparseocc_exact_require_full_predictions",
        action="store_true",
        default=False,
        help=(
            "In sparseocc exact mode, require predictions for all infos in the selected pkl."
        ),
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument(
        "--results_global_filename",
        type=str,
        default="results_global.csv",
        help="Filename of global results table to update when --exp_path is used.",
    )
    cli_args = parser.parse_args()

    main(cli_args)
