
import argparse
import glob
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from FreeOcc.eval.class_conversion import occ3d_nuscenes_classes_names
from FreeOcc.eval.ray_ego_pose_dataset import EgoPoseDataset
from FreeOcc.eval.ray_eval_data import (
    build_scene_infos,
    load_nuscenes_info_by_token,
)
from FreeOcc.eval.ray_metrics import (
    generate_lidar_rays,
    main_raypq,
    main_rayiou,
    process_one_sample,
)
from FreeOcc.eval.ray_pq import Metric_RayPQ

logging.basicConfig(format='[%(module)s | l.%(lineno)d] %(message)s')
logging.getLogger().setLevel(logging.INFO)

occ_class_names_occ3d = occ3d_nuscenes_classes_names


def resolve_occ_gt_root(occ3d_root, label_name):
    candidate_roots = [
        occ3d_root / "gts",
        occ3d_root,
    ]
    for candidate_root in candidate_roots:
        if not candidate_root.is_dir():
            continue
        if list(candidate_root.glob(f"*/*/{label_name}")):
            return candidate_root
    for candidate_root in candidate_roots:
        if candidate_root.is_dir():
            return candidate_root
    raise FileNotFoundError(f"Could not find GT root under {occ3d_root}")


def build_token_maps(occ_gt_root, label_name):
    token_to_label_path = {}
    token_to_scene = {}
    for label_path in glob.glob(str(occ_gt_root / "*" / "*" / label_name)):
        path = Path(label_path)
        token = path.parent.name
        scene_name = path.parent.parent.name
        token_to_label_path[token] = path
        token_to_scene[token] = scene_name
    return token_to_label_path, token_to_scene


def load_prediction_array(pred_path, preferred_key):
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {pred_path}")
    with np.load(pred_path, allow_pickle=True) as pred_data:
        if preferred_key is not None and preferred_key in pred_data.files:
            return pred_data[preferred_key]
        if len(pred_data.files) == 1:
            return pred_data[pred_data.files[0]]
        if "pred" in pred_data.files:
            return pred_data["pred"]
        if "arr_0" in pred_data.files:
            return pred_data["arr_0"]
        raise KeyError(f"Could not resolve prediction key in {pred_path}, keys={pred_data.files}")


def build_pred_maps_from_exp(
    exp_path,
    sem_filename,
    inst_filename,
):
    sem_path_by_token = {}
    inst_path_by_token = {}

    for sem_path_string in glob.glob(str(exp_path / "*" / "*" / sem_filename)):
        sem_path = Path(sem_path_string)
        token = sem_path.parent.name
        sem_path_by_token[token] = sem_path
        inst_path = sem_path.parent / inst_filename
        if inst_path.exists():
            inst_path_by_token[token] = inst_path

    return sem_path_by_token, inst_path_by_token


def build_pred_maps_from_npz(pred_dir):
    sem_path_by_token = {}
    inst_path_by_token = {}
    for pred_path in pred_dir.glob("*.npz"):
        token = pred_path.stem
        sem_path_by_token[token] = pred_path
        inst_path_by_token[token] = pred_path
    return sem_path_by_token, inst_path_by_token


def resolve_prediction_source(
    pred_dir,
    exp_path,
    exp_sem_filename,
    exp_inst_filename,
):
    if exp_path is not None:
        exp_path_obj = Path(exp_path)
        if not exp_path_obj.is_dir():
            raise FileNotFoundError(f"--exp_path not found: {exp_path_obj}")
        sem_path_by_token, inst_path_by_token = build_pred_maps_from_exp(
            exp_path=exp_path_obj,
            sem_filename=exp_sem_filename,
            inst_filename=exp_inst_filename,
        )
        return "exp", sem_path_by_token, inst_path_by_token

    if pred_dir is not None:
        pred_dir_obj = Path(pred_dir)
        if not pred_dir_obj.is_dir():
            raise FileNotFoundError(f"--pred_dir not found: {pred_dir_obj}")
        sem_path_by_token, inst_path_by_token = build_pred_maps_from_npz(pred_dir_obj)
        return "npz", sem_path_by_token, inst_path_by_token

    raise ValueError("One of --exp_path or --pred_dir must be provided.")


def build_token_to_scene_from_all_occ_npz(occ_gt_root):
    token_to_scene = {}
    for label_path in sorted(glob.glob(str(occ_gt_root / "*" / "*" / "*.npz"))):
        path = Path(label_path)
        token = path.parent.name
        scene_name = path.parent.parent.name
        token_to_scene[token] = scene_name
    return token_to_scene


def resolve_sparseocc_infos_pkl(args):
    if args.sparseocc_infos_pkl is not None:
        infos_pkl_path = Path(args.sparseocc_infos_pkl)
        if not infos_pkl_path.exists():
            raise FileNotFoundError(f"--sparseocc_infos_pkl not found: {infos_pkl_path}")
        return infos_pkl_path

    infos_root = Path(args.sparseocc_infos_root)
    split_to_filename = {
        "train": "nuscenes_infos_train_sweep.pkl",
        "val": "nuscenes_infos_val_sweep.pkl",
        "test": "nuscenes_infos_test_sweep.pkl",
    }
    if args.nuscenes_split not in split_to_filename:
        raise ValueError(
            "--sparseocc_exact_mode requires --sparseocc_infos_pkl when nuscenes_split is "
            f"'{args.nuscenes_split}'."
        )
    infos_pkl_path = infos_root / split_to_filename[args.nuscenes_split]
    if not infos_pkl_path.exists():
        raise FileNotFoundError(
            f"Could not find SparseOcc infos pkl at {infos_pkl_path}. "
            "Use --sparseocc_infos_pkl to set it explicitly."
        )
    return infos_pkl_path


def load_infos_from_sparseocc_pkl(infos_pkl_path):
    with open(infos_pkl_path, "rb") as file_handle:
        payload = pickle.load(file_handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected SparseOcc infos format in {infos_pkl_path}")
    infos = payload.get("infos")
    if not isinstance(infos, list):
        raise ValueError(f"Missing infos list in {infos_pkl_path}")
    scene_infos = [dict(info) for info in infos]
    return scene_infos


def build_scene_infos_sparseocc_exact(
    args,
    token_to_scene,
    token_to_sem_gt_path,
):
    infos_pkl_path = resolve_sparseocc_infos_pkl(args)
    scene_infos = load_infos_from_sparseocc_pkl(infos_pkl_path)

    for index, info in enumerate(scene_infos):
        token = info.get("token")
        if token is None:
            raise KeyError(f"Missing token at infos index={index} in {infos_pkl_path}")
        if token not in token_to_scene:
            raise KeyError(
                f"Token {token} from {infos_pkl_path} is missing in GT scene mapping under {args.occ3d_root}"
            )
        if token not in token_to_sem_gt_path:
            raise KeyError(
                f"Token {token} from {infos_pkl_path} is missing semantic GT labels.npz under {args.occ3d_root}"
            )
        info["scene_name"] = token_to_scene[token]

    logging.info(
        "SparseOcc exact mode enabled with infos=%s num_infos=%d",
        infos_pkl_path,
        len(scene_infos),
    )
    return scene_infos


def build_scene_infos_for_eval(args, eval_tokens):
    scene_infos_eval, missing_tokens = build_scene_infos(
        sample_tokens=eval_tokens,
        scene_name="__ray_eval__",
        nuscenes_root=Path(args.nuscenes_root),
        nuscenes_split=args.nuscenes_split,
    )
    if len(missing_tokens) > 0:
        logging.warning("Missing pose infos for %d tokens", len(missing_tokens))

    info_by_token_eval = {info["token"]: info for info in scene_infos_eval}
    eval_tokens = [token for token in eval_tokens if token in info_by_token_eval]
    if len(eval_tokens) == 0:
        raise RuntimeError("No valid tokens with both prediction, GT, and pose info.")

    return scene_infos_eval, info_by_token_eval, eval_tokens


def build_scene_infos_context(args, eval_tokens, info_by_token_eval):
    nuscenes_info_by_token = load_nuscenes_info_by_token(args.nuscenes_root)
    eval_scene_names = {
        info_by_token_eval[token].get("scene_name")
        for token in eval_tokens
        if info_by_token_eval[token].get("scene_name") not in [None, ""]
    }

    eval_count_by_scene = {}
    for token in eval_tokens:
        scene_name = info_by_token_eval[token].get("scene_name")
        if scene_name in [None, ""]:
            continue
        eval_count_by_scene[scene_name] = eval_count_by_scene.get(scene_name, 0) + 1

    tiny_scene_subset = {
        scene_name: count
        for scene_name, count in eval_count_by_scene.items()
        if count < 5
    }
    if len(tiny_scene_subset) > 0:
        logging.warning(
            "Ray metrics are being computed on tiny scene subsets (scene -> num_eval_tokens): %s. "
            "Scores, especially RayPQ, can be unstable.",
            tiny_scene_subset,
        )

    scene_infos_context = [
        dict(info)
        for info in nuscenes_info_by_token.values()
        if info.get("scene_name") in eval_scene_names
    ]
    if len(scene_infos_context) == 0:
        scene_infos_context = [info_by_token_eval[token] for token in eval_tokens]
        logging.warning(
            "Could not expand to full-scene ray context. Falling back to eval-token-only context."
        )
    else:
        logging.info(
            "Expanded ray context: %d eval tokens across %d scenes -> %d scene frames for origin sampling.",
            len(eval_tokens),
            len(eval_scene_names),
            len(scene_infos_context),
        )

    return scene_infos_context


def load_semantic_prediction(pred_source, sem_path, pred_sem_key):
    if pred_source == "exp":
        return np.load(sem_path)
    return load_prediction_array(sem_path, pred_sem_key)


def load_instance_prediction(pred_source, inst_path, pred_inst_key):
    if pred_source == "exp":
        return np.load(inst_path)
    return load_prediction_array(inst_path, pred_inst_key)




def collect_eval_data(
    args,
    eval_tokens,
    scene_infos_context,
    token_to_sem_gt_path,
    sem_path_by_token,
    token_to_inst_gt_path,
    inst_path_by_token,
    pred_source,
):
    eval_tokens_loaded = []
    lidar_origins = []
    sem_gts = []
    sem_preds = []

    eval_tokens_loaded_pq = []
    lidar_origins_pq = []
    sem_gts_pq = []
    sem_preds_pq = []
    inst_gts = []
    inst_preds = []
    pq_missing_gt_instances = 0
    pq_missing_pred_instances = 0

    eval_token_set = set(eval_tokens)
    data_loader = DataLoader(
        EgoPoseDataset(scene_infos_context),
        num_workers=args.num_workers,
    )
    for batch in data_loader:
        token = batch[0][0]
        if token not in eval_token_set:
            continue
        output_origin = batch[1]

        gt_sem_path = token_to_sem_gt_path[token]
        with np.load(gt_sem_path, allow_pickle=True) as gt_data:
            if "semantics" not in gt_data.files:
                continue
            semantics_gt = np.reshape(gt_data["semantics"], [200, 200, 16]).astype(np.uint8)

        if token not in sem_path_by_token:
            continue
        sem_path = sem_path_by_token[token]
        semantics_pred = load_semantic_prediction(
            pred_source=pred_source,
            sem_path=sem_path,
            pred_sem_key=args.pred_sem_key,
        )
        semantics_pred = np.reshape(semantics_pred, [200, 200, 16]).astype(np.uint8)

        eval_tokens_loaded.append(token)
        lidar_origins.append(output_origin)
        sem_gts.append(semantics_gt)
        sem_preds.append(semantics_pred)

        if args.compute_raypq:
            if token not in token_to_inst_gt_path:
                pq_missing_gt_instances += 1
                continue
            if token not in inst_path_by_token:
                pq_missing_pred_instances += 1
                continue

            gt_inst_path = token_to_inst_gt_path[token]
            with np.load(gt_inst_path, allow_pickle=True) as gt_inst_data:
                if "instances" not in gt_inst_data.files:
                    pq_missing_gt_instances += 1
                    continue
                instances_gt = np.reshape(gt_inst_data["instances"], [200, 200, 16]).astype(np.int32)

            inst_path = inst_path_by_token[token]
            instances_pred = load_instance_prediction(
                pred_source=pred_source,
                inst_path=inst_path,
                pred_inst_key=args.pred_inst_key,
            )
            instances_pred = np.reshape(instances_pred, [200, 200, 16]).astype(np.int32)

            eval_tokens_loaded_pq.append(token)
            lidar_origins_pq.append(output_origin)
            sem_gts_pq.append(semantics_gt)
            sem_preds_pq.append(semantics_pred)
            inst_gts.append(instances_gt)
            inst_preds.append(instances_pred)

    return {
        "tokens": eval_tokens_loaded,
        "lidar_origins": lidar_origins,
        "sem_gts": sem_gts,
        "sem_preds": sem_preds,
        "tokens_pq": eval_tokens_loaded_pq,
        "lidar_origins_pq": lidar_origins_pq,
        "sem_gts_pq": sem_gts_pq,
        "sem_preds_pq": sem_preds_pq,
        "inst_gts": inst_gts,
        "inst_preds": inst_preds,
        "pq_missing_gt_instances": pq_missing_gt_instances,
        "pq_missing_pred_instances": pq_missing_pred_instances,
    }


def collect_eval_data_sparseocc_exact(
    args,
    scene_infos_context,
    token_to_sem_gt_path,
    sem_path_by_token,
    token_to_inst_gt_path,
    inst_path_by_token,
    pred_source,
):
    eval_tokens_loaded = []
    lidar_origins = []
    sem_gts = []
    sem_preds = []

    eval_tokens_loaded_pq = []
    lidar_origins_pq = []
    sem_gts_pq = []
    sem_preds_pq = []
    inst_gts = []
    inst_preds = []
    pq_missing_gt_instances = 0
    pq_missing_pred_instances = 0

    data_loader = DataLoader(
        EgoPoseDataset(scene_infos_context, sort_scene_frames=False),
        num_workers=args.num_workers,
    )
    for info_index, batch in enumerate(data_loader):
        output_origin = batch[1]
        info = scene_infos_context[info_index]
        token = info["token"]
        batch_token = batch[0][0]
        if batch_token != token:
            raise RuntimeError(
                f"DataLoader token order mismatch in SparseOcc exact mode: "
                f"batch_token={batch_token} info_token={token} index={info_index}"
            )

        if token not in token_to_sem_gt_path:
            raise KeyError(
                f"Token {token} has no semantic GT labels.npz under {args.occ3d_root}"
            )
        gt_sem_path = token_to_sem_gt_path[token]
        with np.load(gt_sem_path, allow_pickle=True) as gt_data:
            if "semantics" not in gt_data.files:
                raise KeyError(f"Missing semantics in GT file: {gt_sem_path}")
            semantics_gt = np.reshape(gt_data["semantics"], [200, 200, 16]).astype(np.uint8)

        if token not in sem_path_by_token:
            raise FileNotFoundError(
                f"Missing semantic prediction for token {token} in source={pred_source}"
            )
        sem_path = sem_path_by_token[token]
        semantics_pred = load_semantic_prediction(
            pred_source=pred_source,
            sem_path=sem_path,
            pred_sem_key=args.pred_sem_key,
        )
        semantics_pred = np.reshape(semantics_pred, [200, 200, 16]).astype(np.uint8)

        eval_tokens_loaded.append(token)
        lidar_origins.append(output_origin)
        sem_gts.append(semantics_gt)
        sem_preds.append(semantics_pred)

        if args.compute_raypq:
            if token not in token_to_inst_gt_path:
                raise KeyError(
                    f"Missing panoptic GT for token {token} (expected {args.gt_label_name})"
                )
            if token not in inst_path_by_token:
                raise FileNotFoundError(
                    f"Missing instance prediction for token {token} in source={pred_source}"
                )

            gt_inst_path = token_to_inst_gt_path[token]
            with np.load(gt_inst_path, allow_pickle=True) as gt_inst_data:
                if "instances" not in gt_inst_data.files:
                    raise KeyError(f"Missing instances in GT file: {gt_inst_path}")
                instances_gt = np.reshape(gt_inst_data["instances"], [200, 200, 16]).astype(np.int32)

            inst_path = inst_path_by_token[token]
            instances_pred = load_instance_prediction(
                pred_source=pred_source,
                inst_path=inst_path,
                pred_inst_key=args.pred_inst_key,
            )
            instances_pred = np.reshape(instances_pred, [200, 200, 16]).astype(np.int32)

            eval_tokens_loaded_pq.append(token)
            lidar_origins_pq.append(output_origin)
            sem_gts_pq.append(semantics_gt)
            sem_preds_pq.append(semantics_pred)
            inst_gts.append(instances_gt)
            inst_preds.append(instances_pred)

    return {
        "tokens": eval_tokens_loaded,
        "lidar_origins": lidar_origins,
        "sem_gts": sem_gts,
        "sem_preds": sem_preds,
        "tokens_pq": eval_tokens_loaded_pq,
        "lidar_origins_pq": lidar_origins_pq,
        "sem_gts_pq": sem_gts_pq,
        "sem_preds_pq": sem_preds_pq,
        "inst_gts": inst_gts,
        "inst_preds": inst_preds,
        "pq_missing_gt_instances": pq_missing_gt_instances,
        "pq_missing_pred_instances": pq_missing_pred_instances,
    }


def get_output_json_path(args):
    if args.output_json is not None:
        return Path(args.output_json)
    if args.exp_path is not None:
        return Path(args.exp_path) / "ray_metrics.json"
    return Path(args.pred_dir) / "ray_metrics.json"


def get_per_scene_confusion_csv_path(args):
    if args.output_per_scene_confusion_csv is not None:
        return Path(args.output_per_scene_confusion_csv)
    if args.exp_path is not None:
        return Path(args.exp_path) / "raypq_per_scene_confusion.csv"
    return Path(args.pred_dir) / "raypq_per_scene_confusion.csv"


def build_scene_token_indices(tokens_pq, token_to_scene):
    scene_to_indices = {}
    for sample_index, token in enumerate(tokens_pq):
        scene_name = token_to_scene.get(token, "__unknown_scene__")
        if scene_name not in scene_to_indices:
            scene_to_indices[scene_name] = []
        scene_to_indices[scene_name].append(sample_index)
    return scene_to_indices


def compute_class_pq_from_counts(tp, fp, fn, iou_sum, eps):
    sq = float(iou_sum) / max(float(tp), float(eps))
    rq = float(tp) / max(float(tp) + 0.5 * float(fp) + 0.5 * float(fn), float(eps))
    if int(tp) + int(fp) + int(fn) <= 0:
        return np.nan, np.nan, np.nan
    return sq * rq, sq, rq


def compute_raypq_per_scene_confusion(args, eval_data, token_to_scene):
    if len(eval_data["sem_preds_pq"]) == 0:
        logging.warning("Per-scene confusion requested but no valid RayPQ samples were found.")
        return {
            "confusion_rows": 0,
            "summary_rows": 0,
            "confusion_csv": None,
            "summary_csv": None,
            "num_scenes": 0,
            "num_samples": 0,
        }

    if not (
        len(eval_data["tokens_pq"])
        == len(eval_data["sem_preds_pq"])
        == len(eval_data["sem_gts_pq"])
        == len(eval_data["inst_preds"])
        == len(eval_data["inst_gts"])
        == len(eval_data["lidar_origins_pq"])
    ):
        raise RuntimeError(
            "RayPQ per-scene confusion received inconsistent list lengths in eval_data."
        )

    scene_to_indices = build_scene_token_indices(
        tokens_pq=eval_data["tokens_pq"],
        token_to_scene=token_to_scene,
    )
    lidar_rays = torch.from_numpy(generate_lidar_rays())

    confusion_rows = []
    summary_rows = []
    total_processed_samples = 0
    free_class_id = len(occ_class_names_occ3d) - 1

    for scene_name in sorted(scene_to_indices.keys()):
        scene_metric = Metric_RayPQ(
            occ_class_names=occ_class_names_occ3d,
            num_classes=len(occ_class_names_occ3d),
            thresholds=[1, 2, 4],
            use_dynamic_id_offset=not args.sparseocc_exact_mode,
        )
        valid_scene_samples = 0
        for sample_index in scene_to_indices[scene_name]:
            sem_pred = torch.from_numpy(
                np.reshape(eval_data["sem_preds_pq"][sample_index], [200, 200, 16])
            )
            sem_gt = torch.from_numpy(
                np.reshape(eval_data["sem_gts_pq"][sample_index], [200, 200, 16])
            )
            inst_pred = torch.from_numpy(
                np.reshape(eval_data["inst_preds"][sample_index], [200, 200, 16])
            )
            inst_gt = torch.from_numpy(
                np.reshape(eval_data["inst_gts"][sample_index], [200, 200, 16])
            )
            lidar_origins = eval_data["lidar_origins_pq"][sample_index]

            pcd_pred = process_one_sample(
                sem_pred,
                lidar_rays,
                lidar_origins,
                instance_pred=inst_pred,
                occ_class_names=occ_class_names_occ3d,
            )
            pcd_gt = process_one_sample(
                sem_gt,
                lidar_rays,
                lidar_origins,
                instance_pred=inst_gt,
                occ_class_names=occ_class_names_occ3d,
            )

            valid_mask = pcd_gt[:, 0].astype(np.int32) != free_class_id
            pcd_pred = pcd_pred[valid_mask]
            pcd_gt = pcd_gt[valid_mask]
            if pcd_pred.shape != pcd_gt.shape or pcd_pred.shape[0] == 0:
                logging.warning(
                    "Skipping per-scene RayPQ confusion for scene=%s sample_index=%d (shape mismatch or empty).",
                    scene_name,
                    sample_index,
                )
                continue

            sem_gt_ray = pcd_gt[:, 0].astype(np.int32)
            sem_pred_ray = pcd_pred[:, 0].astype(np.int32)
            instances_gt_ray = pcd_gt[:, 1].astype(np.int32)
            instances_pred_ray = pcd_pred[:, 1].astype(np.int32)
            depth_gt_ray = pcd_gt[:, 2]
            depth_pred_ray = pcd_pred[:, 2]
            l1_error = np.abs(depth_pred_ray - depth_gt_ray)
            scene_metric.add_batch(
                sem_pred_ray,
                sem_gt_ray,
                instances_pred_ray,
                instances_gt_ray,
                l1_error,
            )
            valid_scene_samples += 1
            total_processed_samples += 1

        if valid_scene_samples == 0:
            continue

        scene_results = scene_metric.count_pq(
            print_table=False,
            thing_only=args.thing_only,
        )
        summary_rows.append(
            {
                "scene_name": scene_name,
                "num_samples": int(valid_scene_samples),
                "RayPQ": float(scene_results["RayPQ"]),
                "RayPQ@1": float(scene_results["RayPQ@1"]),
                "RayPQ@2": float(scene_results["RayPQ@2"]),
                "RayPQ@4": float(scene_results["RayPQ@4"]),
            }
        )
        if args.thing_only:
            summary_rows[-1]["RayPQ_thing"] = float(scene_results["RayPQ_thing"])
            summary_rows[-1]["RayPQ_thing@1"] = float(scene_results["RayPQ_thing@1"])
            summary_rows[-1]["RayPQ_thing@2"] = float(scene_results["RayPQ_thing@2"])
            summary_rows[-1]["RayPQ_thing@4"] = float(scene_results["RayPQ_thing@4"])

        for threshold_index, threshold in enumerate(scene_metric.thresholds):
            for class_id, class_name in enumerate(occ_class_names_occ3d):
                if class_id == free_class_id:
                    continue
                tp_value = int(scene_metric.pan_tp[threshold_index][class_id])
                fp_value = int(scene_metric.pan_fp[threshold_index][class_id])
                fn_value = int(scene_metric.pan_fn[threshold_index][class_id])
                iou_sum_value = float(scene_metric.pan_iou[threshold_index][class_id])
                pq_value, sq_value, rq_value = compute_class_pq_from_counts(
                    tp=tp_value,
                    fp=fp_value,
                    fn=fn_value,
                    iou_sum=iou_sum_value,
                    eps=scene_metric.eps,
                )
                if tp_value + fp_value + fn_value <= 0:
                    continue
                confusion_rows.append(
                    {
                        "scene_name": scene_name,
                        "threshold": int(threshold),
                        "class_id": int(class_id),
                        "class_name": class_name,
                        "TP": tp_value,
                        "FP": fp_value,
                        "FN": fn_value,
                        "SQ": sq_value,
                        "RQ": rq_value,
                        "PQ": pq_value,
                    }
                )

    confusion_csv_path = get_per_scene_confusion_csv_path(args)
    confusion_csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_csv_path = confusion_csv_path.with_name(
        confusion_csv_path.stem + "_summary.csv"
    )

    confusion_df = pd.DataFrame(confusion_rows)
    if confusion_df.shape[0] > 0:
        confusion_df = confusion_df.sort_values(
            by=["scene_name", "threshold", "class_id"],
            ascending=[True, True, True],
        )
    confusion_df.to_csv(confusion_csv_path, index=False)

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.shape[0] > 0:
        summary_df = summary_df.sort_values(by=["scene_name"], ascending=[True])
    summary_df.to_csv(summary_csv_path, index=False)

    logging.info(
        "Saved per-scene RayPQ confusion to %s (%d rows), summary to %s (%d scenes, %d samples).",
        confusion_csv_path,
        int(confusion_df.shape[0]),
        summary_csv_path,
        int(summary_df.shape[0]),
        int(total_processed_samples),
    )
    return {
        "confusion_rows": int(confusion_df.shape[0]),
        "summary_rows": int(summary_df.shape[0]),
        "confusion_csv": str(confusion_csv_path),
        "summary_csv": str(summary_csv_path),
        "num_scenes": int(summary_df.shape[0]),
        "num_samples": int(total_processed_samples),
    }


def get_results_global_path(exp_path, results_global_filename):
    exp_folder = Path(exp_path)
    if exp_folder.parent.name == "exps":
        return exp_folder.parent.parent / results_global_filename
    return exp_folder.parent / results_global_filename


def reorder_global_results_columns(results_df):
    metadata_columns = []
    for column_name in [
        "timestamp",
        "exp_name",
        "scene_name",
        "scope",
        "num_scenes",
        "num_samples",
    ]:
        if column_name in results_df.columns:
            metadata_columns.append(column_name)

    metric_columns = []
    for column_name in ["mIoU", "RayIoU", "RayPQ", "RayPQ_thing"]:
        if column_name in results_df.columns:
            metric_columns.append(column_name)

    class_iou_columns = []
    for column_name in results_df.columns:
        if not str(column_name).startswith("IoU_"):
            continue
        if column_name in ["IoU_occupied", "IoU_free"]:
            continue
        class_iou_columns.append(column_name)

    ordered_columns = metadata_columns + metric_columns + class_iou_columns
    ordered_columns = list(dict.fromkeys(ordered_columns))
    remaining_columns = [column for column in results_df.columns if column not in ordered_columns]
    return results_df[ordered_columns + remaining_columns]


def update_results_global_with_ray_metrics(exp_path, results_global_filename, ray_metrics):
    results_global_path = get_results_global_path(exp_path, results_global_filename)
    if not results_global_path.exists():
        logging.warning(
            "Could not update global results: %s does not exist.",
            results_global_path,
        )
        return

    results_df = pd.read_csv(results_global_path, index_col=0)
    if "exp_name" not in results_df.columns:
        logging.warning(
            "Could not update global results: missing exp_name column in %s.",
            results_global_path,
        )
        return

    exp_name = Path(exp_path).name
    exp_mask = results_df["exp_name"].astype(str) == exp_name
    if "scope" in results_df.columns:
        row_mask = exp_mask & (results_df["scope"].astype(str) == "all_scenes_all_points")
    elif "scene_name" in results_df.columns:
        row_mask = exp_mask & (results_df["scene_name"].astype(str) == "__all__")
    else:
        row_mask = exp_mask

    if not row_mask.any():
        logging.warning(
            "Could not update global results: no row found for exp_name=%s in %s. "
            "Ray metrics stay in the experiment json only.",
            exp_name,
            results_global_path,
        )
        return

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

    ray_keys = sorted(
        [
            key
            for key in ray_metrics.keys()
            if str(key).startswith("Ray") or str(key).startswith("num_ray")
        ]
    )
    for key in ray_keys:
        if key in ray_metrics and key not in results_df.columns:
            results_df[key] = np.nan
    if "RayIoU" not in results_df.columns:
        results_df["RayIoU"] = np.nan
    if "RayPQ" not in results_df.columns:
        results_df["RayPQ"] = np.nan

    for key in ray_keys:
        if key in ray_metrics:
            results_df.at[row_index, key] = ray_metrics[key]

    results_df = reorder_global_results_columns(results_df)

    backup_path = results_global_path.with_name(results_global_path.stem + "_backup.csv")
    with open(results_global_path, "rb") as source_file, open(backup_path, "wb") as backup_file:
        backup_file.write(source_file.read())
    results_df.to_csv(results_global_path)
    logging.info(
        "Updated global results row for exp_name=%s in %s",
        exp_name,
        results_global_path,
    )


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
    token_to_sem_gt_path, token_to_scene_labels = build_token_maps(occ_gt_root_sem, "labels.npz")
    token_to_scene_all_npz = build_token_to_scene_from_all_occ_npz(occ_gt_root_sem)
    if len(token_to_scene_all_npz) == 0:
        token_to_scene_all_npz = token_to_scene_labels

    token_to_inst_gt_path = {}
    if args.compute_raypq:
        occ_gt_root_inst = resolve_occ_gt_root(Path(args.occ3d_root), args.gt_label_name)
        token_to_inst_gt_path, _ = build_token_maps(occ_gt_root_inst, args.gt_label_name)

    if args.sparseocc_exact_mode:
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
            "source=%s sparseocc_exact_mode=1 tokens_total=%d context_frames=%d",
            pred_source,
            len(token_to_sem_gt_path),
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
        pred_tokens = set(sem_path_by_token.keys())
        eval_tokens = sorted(set(token_to_sem_gt_path.keys()) & pred_tokens)
        if len(eval_tokens) == 0:
            raise RuntimeError(
                "No overlapping tokens between semantic GT (labels.npz) and predictions. "
                "Expected either <sample_token>.npz files or exp outputs in scene/token folders."
            )

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
            "source=%s tokens_total=%d overlap=%d pose_infos=%d usable=%d context_frames=%d",
            pred_source,
            len(token_to_sem_gt_path),
            len(pred_tokens),
            len(scene_infos_eval),
            len(eval_tokens),
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

    logging.info("Loaded predictions for %d samples", len(eval_data["sem_preds"]))
    if len(eval_data["sem_preds"]) == 0:
        raise RuntimeError("No valid samples were loaded for ray evaluation.")

    rayiou_results = main_rayiou(
        eval_data["sem_preds"],
        eval_data["sem_gts"],
        eval_data["lidar_origins"],
        occ_class_names=occ_class_names_occ3d,
    )
    logging.info("RayIoU results: %s", rayiou_results)

    all_results = dict(rayiou_results)
    all_results["num_ray_samples"] = len(eval_data["sem_preds"])
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
            if args.thing_only:
                logging.info(
                    "RayPQ thing-only summary: RayPQ_thing=%.4f RayPQ_thing@1=%.4f RayPQ_thing@2=%.4f RayPQ_thing@4=%.4f",
                    float(raypq_results["RayPQ_thing"]),
                    float(raypq_results["RayPQ_thing@1"]),
                    float(raypq_results["RayPQ_thing@2"]),
                    float(raypq_results["RayPQ_thing@4"]),
                )
        logging.info("RayPQ results: %s", raypq_results)
        all_results.update(raypq_results)
    elif args.get_per_scene_confusion:
        logging.warning(
            "--get_per_scene_confusion requires --compute_raypq. Skipping per-scene confusion."
        )

    if args.get_per_scene_confusion and args.compute_raypq:
        per_scene_confusion = compute_raypq_per_scene_confusion(
            args=args,
            eval_data=eval_data,
            token_to_scene=token_to_scene_all_npz,
        )
        all_results["per_scene_confusion_csv"] = per_scene_confusion["confusion_csv"]
        all_results["per_scene_confusion_summary_csv"] = per_scene_confusion["summary_csv"]
        all_results["num_per_scene_confusion_rows"] = int(per_scene_confusion["confusion_rows"])
        all_results["num_per_scene_confusion_scenes"] = int(per_scene_confusion["num_scenes"])

    output_json_path = get_output_json_path(args)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w") as file_handle:
        json.dump(all_results, file_handle, indent=2)
    logging.info("Saved results to %s", output_json_path)

    if args.exp_path is not None:
        update_results_global_with_ray_metrics(
            exp_path=args.exp_path,
            results_global_filename=args.results_global_filename,
            ray_metrics=all_results,
        )


resolveOccGtRoot = resolve_occ_gt_root
buildTokenMaps = build_token_maps
loadPredictionArray = load_prediction_array
buildPredMapsFromExp = build_pred_maps_from_exp
buildPredMapsFromNpz = build_pred_maps_from_npz
resolvePredictionSource = resolve_prediction_source


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate RayIoU / RayPQ from token-wise predictions.",
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
    parser.add_argument(
        "--get_per_scene_confusion",
        action="store_true",
        default=False,
        help="For RayPQ, export per-scene per-class TP/FP/FN confusion tables.",
    )
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
            "Use SparseOcc-style PKL infos and strict ordering/path assumptions. "
            "This reproduces SparseOcc evaluation flow as closely as possible."
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
            "In sparseocc exact mode, require predictions for all infos in the selected pkl. "
            "By default, exact mode intersects pkl infos with available predictions for subset eval."
        ),
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--translate_z",
        type=int,
        default=0,
        help=(
            "Translate predicted occupancy grids along z before evaluation. "
            "Positive values move up, negative values move down, out-of-grid voxels are cropped."
        ),
    )
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument(
        "--output_per_scene_confusion_csv",
        type=str,
        default=None,
        help="Optional output csv path for per-scene RayPQ confusion. Default: <exp_path>/raypq_per_scene_confusion.csv",
    )
    parser.add_argument(
        "--results_global_filename",
        type=str,
        default="results_global.csv",
        help="Filename of global results table to update when --exp_path is used.",
    )
    cli_args = parser.parse_args()

    main(cli_args)
