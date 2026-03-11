"""
RayPQ metric.
Computation is kept aligned with SparseOcc/loaders/ray_pq.py.
"""

import numpy as np
import logging

try:
    from prettytable import PrettyTable
except Exception:  # pragma: no cover - optional dependency fallback
    class PrettyTable:  # type: ignore[override]
        def __init__(self, field_names):
            self.field_names = field_names
            self.rows = []
            self.float_format = ".3"

        def add_row(self, row, divider=False):
            self.rows.append(row)

        def __str__(self):
            lines = [" | ".join([str(x) for x in self.field_names])]
            for row in self.rows:
                lines.append(" | ".join([str(x) for x in row]))
            return "\n".join(lines)


def get_nanmean(values):
    valid_mask = np.isfinite(values)
    if not np.any(valid_mask):
        return float("nan")
    return float(np.mean(values[valid_mask]))


class Metric_RayPQ:
    def __init__(
        self,
        occ_class_names,
        num_classes=18,
        thresholds=None,
        use_dynamic_id_offset=True,
    ):
        if thresholds is None:
            thresholds = [1, 2, 4]
        if num_classes == 18 or num_classes == 17:
            self.class_names = occ_class_names
        else:
            raise ValueError

        self.num_classes = num_classes
        self.id_offset = 2 ** 16
        self.use_dynamic_id_offset = use_dynamic_id_offset
        self.eps = 1e-5
        self.thresholds = thresholds

        self.min_num_points = 10
        self.include = np.array(
            [n for n in range(self.num_classes - 1)],
            dtype=int,
        )
        self.thing_class_names = {
            "car",
            "truck",
            "construction_vehicle",
            "bus",
            "trailer",
            "motorcycle",
            "bicycle",
            "pedestrian",
        }
        self.thing_class_ids = np.array(
            [
                class_id
                for class_id in range(self.num_classes - 1)
                if self.class_names[class_id] in self.thing_class_names
            ],
            dtype=int,
        )
        self.cnt = 0

        self.pan_tp = np.zeros([len(self.thresholds), num_classes], dtype=int)
        self.pan_iou = np.zeros([len(self.thresholds), num_classes], dtype=np.double)
        self.pan_fp = np.zeros([len(self.thresholds), num_classes], dtype=int)
        self.pan_fn = np.zeros([len(self.thresholds), num_classes], dtype=int)
        self.fixed_offset_fallback_warning_emitted = False

    def add_batch(
        self,
        semantics_pred,
        semantics_gt,
        instances_pred,
        instances_gt,
        l1_error,
    ):
        self.cnt += 1
        self.add_panoptic_sample(
            semantics_pred,
            semantics_gt,
            instances_pred,
            instances_gt,
            l1_error,
        )

    def add_panoptic_sample(
        self,
        semantics_pred,
        semantics_gt,
        instances_pred,
        instances_gt,
        l1_error,
    ):
        instance_class_ids = [self.num_classes - 1]
        for i in range(1, instances_gt.max() + 1):
            class_id = np.unique(semantics_gt[instances_gt == i])
            if class_id.shape[0] == 1:
                instance_class_ids.append(class_id[0])
            else:
                instance_class_ids.append(self.num_classes - 1)
        instance_class_ids = np.array(instance_class_ids)

        instance_count = 1
        final_instances = np.zeros_like(instances_gt)

        for class_id in range(self.num_classes - 1):
            if np.sum(semantics_gt == class_id) == 0:
                continue

            if self.class_names[class_id] in [
                "car",
                "truck",
                "construction_vehicle",
                "bus",
                "trailer",
                "motorcycle",
                "bicycle",
                "pedestrian",
            ]:
                for instance_id in range(len(instance_class_ids)):
                    if instance_class_ids[instance_id] != class_id:
                        continue
                    final_instances[instances_gt == instance_id] = instance_count
                    instance_count += 1
            else:
                final_instances[semantics_gt == class_id] = instance_count
                instance_count += 1

        instances_gt = final_instances

        instances_pred = instances_pred + 1
        instances_gt = instances_gt + 1

        for j, threshold in enumerate(self.thresholds):
            tp_dist_mask = l1_error < threshold
            for cl in self.include:
                pred_inst_in_cl_mask = semantics_pred == cl
                gt_inst_in_cl_mask = semantics_gt == cl

                pred_inst_in_cl = instances_pred * pred_inst_in_cl_mask.astype(int)
                gt_inst_in_cl = instances_gt * gt_inst_in_cl_mask.astype(int)

                unique_pred, counts_pred = np.unique(
                    pred_inst_in_cl[pred_inst_in_cl > 0],
                    return_counts=True,
                )
                id2idx_pred = {
                    instance_id: idx
                    for idx, instance_id in enumerate(unique_pred)
                }
                matched_pred = np.array([False] * unique_pred.shape[0])

                unique_gt, counts_gt = np.unique(
                    gt_inst_in_cl[gt_inst_in_cl > 0],
                    return_counts=True,
                )
                id2idx_gt = {
                    instance_id: idx
                    for idx, instance_id in enumerate(unique_gt)
                }
                matched_gt = np.array([False] * unique_gt.shape[0])

                valid_combos = np.logical_and(pred_inst_in_cl > 0, gt_inst_in_cl > 0)
                valid_combos = np.logical_and(valid_combos, tp_dist_mask)

                pred_instances_valid = pred_inst_in_cl[valid_combos].astype(np.int64)
                gt_instances_valid = gt_inst_in_cl[valid_combos].astype(np.int64)
                max_pred_instance_id = int(unique_pred.max()) if unique_pred.size > 0 else 0
                max_gt_instance_id = int(unique_gt.max()) if unique_gt.size > 0 else 0
                if self.use_dynamic_id_offset:
                    id_offset = max(max_pred_instance_id, max_gt_instance_id) + 1
                    if id_offset <= 0:
                        id_offset = 1
                else:
                    if max_pred_instance_id >= self.id_offset:
                        id_offset = max(max_pred_instance_id, max_gt_instance_id) + 1
                        if id_offset <= 0:
                            id_offset = 1
                        if not self.fixed_offset_fallback_warning_emitted:
                            logging.warning(
                                "RayPQ fixed id_offset=%d is unsafe for max predicted instance id=%d. "
                                "Falling back to dynamic id_offset=%d to avoid invalid decoding.",
                                self.id_offset,
                                max_pred_instance_id,
                                id_offset,
                            )
                            self.fixed_offset_fallback_warning_emitted = True
                    else:
                        id_offset = self.id_offset
                id_offset_combo = pred_instances_valid + np.int64(id_offset) * gt_instances_valid
                unique_combo, counts_combo = np.unique(id_offset_combo, return_counts=True)

                gt_labels = unique_combo // np.int64(id_offset)
                pred_labels = unique_combo % np.int64(id_offset)
                valid_pairs_mask = np.array(
                    [
                        (instance_id_gt in id2idx_gt) and (instance_id_pred in id2idx_pred)
                        for instance_id_gt, instance_id_pred in zip(gt_labels, pred_labels)
                    ],
                    dtype=bool,
                )
                if not valid_pairs_mask.all():
                    gt_labels = gt_labels[valid_pairs_mask]
                    pred_labels = pred_labels[valid_pairs_mask]
                    counts_combo = counts_combo[valid_pairs_mask]
                    if gt_labels.size == 0:
                        continue
                gt_areas = np.array([counts_gt[id2idx_gt[instance_id]] for instance_id in gt_labels])
                pred_areas = np.array([counts_pred[id2idx_pred[instance_id]] for instance_id in pred_labels])
                intersections = counts_combo
                unions = gt_areas + pred_areas - intersections
                ious = intersections.astype(float) / unions.astype(float)

                tp_indexes = ious > 0.5
                self.pan_tp[j][cl] += np.sum(tp_indexes)
                self.pan_iou[j][cl] += np.sum(ious[tp_indexes])

                matched_gt[
                    [id2idx_gt[instance_id] for instance_id in gt_labels[tp_indexes]]
                ] = True
                matched_pred[
                    [id2idx_pred[instance_id] for instance_id in pred_labels[tp_indexes]]
                ] = True

                if len(counts_gt) > 0:
                    self.pan_fn[j][cl] += np.sum(
                        np.logical_and(
                            counts_gt >= self.min_num_points,
                            ~matched_gt,
                        )
                    )

                if len(matched_pred) > 0:
                    self.pan_fp[j][cl] += np.sum(
                        np.logical_and(
                            counts_pred >= self.min_num_points,
                            ~matched_pred,
                        )
                    )

    def count_pq(self, print_table=True, include_per_class=False, thing_only=False):
        sq_all = self.pan_iou.astype(np.double) / np.maximum(
            self.pan_tp.astype(np.double),
            self.eps,
        )
        rq_all = self.pan_tp.astype(np.double) / np.maximum(
            self.pan_tp.astype(np.double)
            + 0.5 * self.pan_fp.astype(np.double)
            + 0.5 * self.pan_fn.astype(np.double),
            self.eps,
        )
        pq_all = sq_all * rq_all

        mask = (self.pan_tp + self.pan_fp + self.pan_fn) > 0
        pq_all[~mask] = float("nan")

        if print_table:
            table = PrettyTable(
                [
                    "Class Names",
                    "RayPQ@%d" % self.thresholds[0],
                    "RayPQ@%d" % self.thresholds[1],
                    "RayPQ@%d" % self.thresholds[2],
                ]
            )
            table.float_format = ".3"

            for i in range(len(self.class_names) - 1):
                table.add_row(
                    [self.class_names[i], pq_all[0][i], pq_all[1][i], pq_all[2][i]],
                    divider=(i == len(self.class_names) - 2),
                )
            table.add_row(
                [
                    "MEAN",
                    get_nanmean(pq_all[0]),
                    get_nanmean(pq_all[1]),
                    get_nanmean(pq_all[2]),
                ]
            )
            logging.info("\n%s", table)

        results = {
            "RayPQ": get_nanmean(pq_all),
            "RayPQ@1": get_nanmean(pq_all[0]),
            "RayPQ@2": get_nanmean(pq_all[1]),
            "RayPQ@4": get_nanmean(pq_all[2]),
        }
        if include_per_class:
            for class_id in range(len(self.class_names) - 1):
                class_name = self.class_names[class_id]
                results[f"RayPQ_{class_name}"] = get_nanmean(pq_all[:, class_id])
                results[f"RayPQ@1_{class_name}"] = pq_all[0, class_id]
                results[f"RayPQ@2_{class_name}"] = pq_all[1, class_id]
                results[f"RayPQ@4_{class_name}"] = pq_all[2, class_id]
        if thing_only:
            if self.thing_class_ids.size == 0:
                results["RayPQ_thing"] = float("nan")
                results["RayPQ_thing@1"] = float("nan")
                results["RayPQ_thing@2"] = float("nan")
                results["RayPQ_thing@4"] = float("nan")
            else:
                thing_pq_all = pq_all[:, self.thing_class_ids]
                results["RayPQ_thing"] = get_nanmean(thing_pq_all)
                results["RayPQ_thing@1"] = get_nanmean(thing_pq_all[0])
                results["RayPQ_thing@2"] = get_nanmean(thing_pq_all[1])
                results["RayPQ_thing@4"] = get_nanmean(thing_pq_all[2])
        return results
