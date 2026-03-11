"""
Copied from https://github.com/boschresearch/GaussianFlowOcc/mmdet3d/datasets/occ_metrics.py
"""


import logging
import numpy as np
import os
import torch

from FreeOcc.configs.config_loader import get_panoptic_thing_class_names

np.seterr(divide='ignore', invalid='ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class IoU():
    def __init__(self,
                save_dir='.',
                use_lidar_mask=False,
                use_image_mask=False,
                eval_tr=0.5,):
        self.save_dir = save_dir
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.class_names = ['occupied', 'free']
        self.eval_threshold = eval_tr

        self.point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
        self.occupancy_size = [0.4, 0.4, 0.4]
        self.voxel_size = 0.4
        self.occ_xdim = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.occupancy_size[0])
        self.occ_ydim = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.occupancy_size[1])
        self.occ_zdim = int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.occupancy_size[2])
        self.voxel_num = self.occ_xdim * self.occ_ydim * self.occ_zdim
        self.num_classes = 2
        self.hist = np.zeros((self.num_classes, self.num_classes))
        self.cnt = 0

    def hist_info(self, n_cl, pred, gt):
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        return (
            np.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2
            ).reshape(n_cl, n_cl),
            correct,
            labeled,
        )

    def per_class_iu(self, hist):

        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def compute_mIoU(self, pred, label, n_classes):
        hist = np.zeros((n_classes, n_classes))
        new_hist, correct, labeled = self.hist_info(n_classes, pred.flatten(), label.flatten())
        hist += new_hist
        mIoUs = self.per_class_iu(hist)
        return np.nanmean(mIoUs), hist


    def add_batch(self,semantics_pred,semantics_gt,mask_lidar,mask_camera):
        self.cnt += 1
        if self.use_image_mask:
            masked_semantics_gt = semantics_gt[mask_camera].copy()
            masked_semantics_pred = semantics_pred[mask_camera].copy()
        elif self.use_lidar_mask:
            masked_semantics_gt = semantics_gt[mask_lidar].copy()
            masked_semantics_pred = semantics_pred[mask_lidar].copy()
        else:
            masked_semantics_gt = semantics_gt.copy()
            masked_semantics_pred = semantics_pred.copy()
        
        is_free_pred = (masked_semantics_pred == 17)
        is_free_gt = (masked_semantics_gt == 17)
        is_ignore_gt = (masked_semantics_gt == 255)

        masked_semantics_pred[:] = 0
        masked_semantics_pred[is_free_pred] = 1

        masked_semantics_gt[:] = 0
        masked_semantics_gt[is_free_gt] = 1
        masked_semantics_gt[is_ignore_gt] = 255

        _, _hist = self.compute_mIoU(masked_semantics_pred, masked_semantics_gt, self.num_classes)
        self.hist += _hist

    def count_miou(self):
        mIoU = self.per_class_iu(self.hist)

        return self.class_names, mIoU, self.cnt

class Metric_mIoU():
    def __init__(self,
                 save_dir='.',
                 num_classes=18,
                 use_lidar_mask=False,
                 use_image_mask=False,
                 class_weighted=False,
                 eval_tr=0.5,
                 with_others=False,
                 ):
        self.class_names = ['others','barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                            'driveable_surface', 'other_flat', 'sidewalk',
                            'terrain', 'manmade', 'vegetation','free']
        self.save_dir = save_dir
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.num_classes = num_classes
        self.eval_threshold = eval_tr

        self.point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
        self.occupancy_size = [0.4, 0.4, 0.4]
        self.voxel_size = 0.4
        self.occ_xdim = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.occupancy_size[0])
        self.occ_ydim = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.occupancy_size[1])
        self.occ_zdim = int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.occupancy_size[2])
        self.voxel_num = self.occ_xdim * self.occ_ydim * self.occ_zdim
        self.hist = np.zeros((self.num_classes, self.num_classes))
        self.cnt = 0
        self.with_others = with_others
        self.eval_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16]
        self.class_weighted = class_weighted
        if self.class_weighted:
            occurences = torch.tensor([1163161, 2309034, 188743, 2997643, 20317180, 852476, 243808, 2457947, 
            497017, 2731022, 7224789, 214411435, 5565043, 63191967, 76098082, 128860031, 
            141625221])

    def hist_info(self, n_cl, pred, gt):
        """
        build confusion matrix
        non-empty class: 0-16
        free voxel class: 17

        Args:
            n_cl (int): num_classes_occupancy
            pred (1-d array): pred_occupancy_label
            gt (1-d array): gt_occupancu_label

        Returns:
            tuple:(hist, correctly number_predicted_labels, num_labelled_sample)
        """
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        return (
            np.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2
            ).reshape(n_cl, n_cl),
            correct,
            labeled,
        )

    def per_class_iu(self, hist):

        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def compute_mIoU(self, pred, label, n_classes):
        hist = np.zeros((n_classes, n_classes))
        new_hist, correct, labeled = self.hist_info(n_classes, pred.flatten(), label.flatten())
        hist += new_hist
        mIoUs = self.per_class_iu(hist)
        return np.nanmean(mIoUs), hist


    def add_batch(self,semantics_pred,semantics_gt,mask_lidar,mask_camera):
        self.cnt += 1
        if self.use_image_mask:
            masked_semantics_gt = semantics_gt[mask_camera]
            masked_semantics_pred = semantics_pred[mask_camera]
        elif self.use_lidar_mask:
            masked_semantics_gt = semantics_gt[mask_lidar]
            masked_semantics_pred = semantics_pred[mask_lidar]
        else:
            masked_semantics_gt = semantics_gt
            masked_semantics_pred = semantics_pred
        _, _hist = self.compute_mIoU(masked_semantics_pred, masked_semantics_gt, self.num_classes)
        self.hist += _hist

    def count_miou(self):
        mIoU = self.per_class_iu(self.hist)

        if self.with_others:
            meanIoU = np.nanmean(mIoU[:self.num_classes-1])
        else:
            meanIoU = np.nanmean(mIoU[self.eval_classes])

        return self.class_names, mIoU, meanIoU, self.cnt


class Metric_PQ():
    def __init__(self,
                 save_dir='.',
                 num_classes=18,
                 use_lidar_mask=False,
                 use_image_mask=False,
                 with_others=False,
                 min_num_points=10):
        self.class_names = ['others','barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                            'driveable_surface', 'other_flat', 'sidewalk',
                            'terrain', 'manmade', 'vegetation','free']
        self.save_dir = save_dir
        self.num_classes = num_classes
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.with_others = with_others
        self.min_num_points = int(min_num_points)
        self.eps = 1e-5
        self.cnt = 0

        if self.with_others:
            self.eval_classes = [i for i in range(self.num_classes - 1)]
        else:
            self.eval_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16]

        thing_class_names = set(get_panoptic_thing_class_names())
        self.thing_class_ids = [
            class_id
            for class_id, class_name in enumerate(self.class_names[:-1])
            if class_name in thing_class_names
        ]
        self.thing_eval_classes = [
            class_id for class_id in self.eval_classes if class_id in self.thing_class_ids
        ]

        self.pan_tp = np.zeros([self.num_classes], dtype=np.int64)
        self.pan_iou = np.zeros([self.num_classes], dtype=np.float64)
        self.pan_fp = np.zeros([self.num_classes], dtype=np.int64)
        self.pan_fn = np.zeros([self.num_classes], dtype=np.int64)

    def build_class_instances(self, semantics, instances, class_id, is_thing):
        class_instances = np.zeros_like(instances, dtype=np.int64)
        class_mask = semantics == class_id
        if not np.any(class_mask):
            return class_instances
        if is_thing:
            class_instances[class_mask] = instances[class_mask].astype(np.int64)
            class_instances[class_instances <= 0] = 0
        else:
            class_instances[class_mask] = 1
        return class_instances

    def add_class_sample(self, class_id, pred_instances, gt_instances):
        pred_positive = pred_instances[pred_instances > 0]
        gt_positive = gt_instances[gt_instances > 0]
        unique_pred, counts_pred = np.unique(pred_positive, return_counts=True)
        unique_gt, counts_gt = np.unique(gt_positive, return_counts=True)

        if unique_pred.shape[0] == 0 and unique_gt.shape[0] == 0:
            return

        matched_pred = np.zeros(unique_pred.shape[0], dtype=bool)
        matched_gt = np.zeros(unique_gt.shape[0], dtype=bool)
        id2idx_pred = {instance_id: idx for idx, instance_id in enumerate(unique_pred)}
        id2idx_gt = {instance_id: idx for idx, instance_id in enumerate(unique_gt)}

        overlap_mask = (pred_instances > 0) & (gt_instances > 0)
        if np.any(overlap_mask):
            pred_overlap = pred_instances[overlap_mask].astype(np.int64)
            gt_overlap = gt_instances[overlap_mask].astype(np.int64)

            max_pred_instance_id = int(np.max(pred_overlap)) if pred_overlap.shape[0] > 0 else 0
            max_gt_instance_id = int(np.max(gt_overlap)) if gt_overlap.shape[0] > 0 else 0
            id_offset = max(max_pred_instance_id, max_gt_instance_id) + 1
            if id_offset <= 0:
                id_offset = 1

            pair_ids = pred_overlap + np.int64(id_offset) * gt_overlap
            unique_pairs, pair_counts = np.unique(pair_ids, return_counts=True)

            gt_pair_ids = unique_pairs // np.int64(id_offset)
            pred_pair_ids = unique_pairs % np.int64(id_offset)
            ious = np.zeros_like(pair_counts, dtype=np.float64)
            gt_pair_idxs = np.zeros_like(pair_counts, dtype=np.int64)
            pred_pair_idxs = np.zeros_like(pair_counts, dtype=np.int64)
            valid_pairs = np.zeros_like(pair_counts, dtype=bool)

            for pair_idx in range(unique_pairs.shape[0]):
                gt_instance_id = gt_pair_ids[pair_idx]
                pred_instance_id = pred_pair_ids[pair_idx]
                if (gt_instance_id not in id2idx_gt) or (pred_instance_id not in id2idx_pred):
                    continue
                gt_idx = id2idx_gt[gt_instance_id]
                pred_idx = id2idx_pred[pred_instance_id]

                intersection = pair_counts[pair_idx]
                union = counts_gt[gt_idx] + counts_pred[pred_idx] - intersection
                if union <= 0:
                    continue
                gt_pair_idxs[pair_idx] = gt_idx
                pred_pair_idxs[pair_idx] = pred_idx
                ious[pair_idx] = intersection / union
                valid_pairs[pair_idx] = True

            valid_pair_idxs = np.where(valid_pairs)[0]
            if valid_pair_idxs.shape[0] > 0:
                sorted_valid_pair_idxs = valid_pair_idxs[np.argsort(-ious[valid_pair_idxs])]
                for pair_idx in sorted_valid_pair_idxs:
                    if ious[pair_idx] <= 0.5:
                        break
                    gt_idx = gt_pair_idxs[pair_idx]
                    pred_idx = pred_pair_idxs[pair_idx]
                    if matched_gt[gt_idx] or matched_pred[pred_idx]:
                        continue
                    matched_gt[gt_idx] = True
                    matched_pred[pred_idx] = True
                    self.pan_tp[class_id] += 1
                    self.pan_iou[class_id] += ious[pair_idx]

        if unique_gt.shape[0] > 0:
            valid_gt = counts_gt >= self.min_num_points
            self.pan_fn[class_id] += np.sum(np.logical_and(valid_gt, ~matched_gt))
        if unique_pred.shape[0] > 0:
            valid_pred = counts_pred >= self.min_num_points
            self.pan_fp[class_id] += np.sum(np.logical_and(valid_pred, ~matched_pred))

    def add_panoptic_sample(self, semantics_pred, semantics_gt, instances_pred, instances_gt):
        for class_id in self.eval_classes:
            is_thing = class_id in self.thing_class_ids
            pred_instances = self.build_class_instances(
                semantics=semantics_pred,
                instances=instances_pred,
                class_id=class_id,
                is_thing=is_thing,
            )
            gt_instances = self.build_class_instances(
                semantics=semantics_gt,
                instances=instances_gt,
                class_id=class_id,
                is_thing=is_thing,
            )
            self.add_class_sample(
                class_id=class_id,
                pred_instances=pred_instances,
                gt_instances=gt_instances,
            )

    def add_batch(self, semantics_pred, semantics_gt, instances_pred, instances_gt, mask_lidar, mask_camera):
        self.cnt += 1
        if self.use_image_mask:
            masked_semantics_gt = semantics_gt[mask_camera]
            masked_semantics_pred = semantics_pred[mask_camera]
            masked_instances_gt = instances_gt[mask_camera]
            masked_instances_pred = instances_pred[mask_camera]
        elif self.use_lidar_mask:
            masked_semantics_gt = semantics_gt[mask_lidar]
            masked_semantics_pred = semantics_pred[mask_lidar]
            masked_instances_gt = instances_gt[mask_lidar]
            masked_instances_pred = instances_pred[mask_lidar]
        else:
            masked_semantics_gt = semantics_gt
            masked_semantics_pred = semantics_pred
            masked_instances_gt = instances_gt
            masked_instances_pred = instances_pred

        valid_gt_mask = (masked_semantics_gt >= 0) & (masked_semantics_gt < self.num_classes)
        if np.any(valid_gt_mask):
            self.add_panoptic_sample(
                semantics_pred=masked_semantics_pred[valid_gt_mask].astype(np.int32),
                semantics_gt=masked_semantics_gt[valid_gt_mask].astype(np.int32),
                instances_pred=masked_instances_pred[valid_gt_mask].astype(np.int64),
                instances_gt=masked_instances_gt[valid_gt_mask].astype(np.int64),
            )

    def count_pq(self, print_table=False):
        sq_all = self.pan_iou.astype(np.float64) / np.maximum(
            self.pan_tp.astype(np.float64),
            self.eps,
        )
        rq_all = self.pan_tp.astype(np.float64) / np.maximum(
            self.pan_tp.astype(np.float64)
            + 0.5 * self.pan_fp.astype(np.float64)
            + 0.5 * self.pan_fn.astype(np.float64),
            self.eps,
        )
        pq_all = sq_all * rq_all

        valid_mask = (self.pan_tp + self.pan_fp + self.pan_fn) > 0
        pq_all[~valid_mask] = np.nan
        sq_all[~valid_mask] = np.nan
        rq_all[~valid_mask] = np.nan

        pq = np.nanmean(pq_all[self.eval_classes]) if len(self.eval_classes) > 0 else np.nan
        sq = np.nanmean(sq_all[self.eval_classes]) if len(self.eval_classes) > 0 else np.nan
        rq = np.nanmean(rq_all[self.eval_classes]) if len(self.eval_classes) > 0 else np.nan

        pq_thing = np.nan
        if len(self.thing_eval_classes) > 0:
            pq_thing = np.nanmean(pq_all[self.thing_eval_classes])

        if print_table:
            logging.info("PQ=%.4f SQ=%.4f RQ=%.4f PQ_thing=%.4f", pq, sq, rq, pq_thing)

        return {
            "PQ": pq,
            "SQ": sq,
            "RQ": rq,
            "PQ_thing": pq_thing,
            "pq_per_class": pq_all,
            "sq_per_class": sq_all,
            "rq_per_class": rq_all,
            "tp_per_class": self.pan_tp.copy(),
            "fp_per_class": self.pan_fp.copy(),
            "fn_per_class": self.pan_fn.copy(),
            "num_samples": self.cnt,
        }
