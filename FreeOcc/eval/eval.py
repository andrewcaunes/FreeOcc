

from FreeOcc.eval.occ_metrics import Metric_mIoU, IoU
import os
import numpy as np


def evaluate_mIoU(occ_results, eval_dict):
        semantic_mIoU = [Metric_mIoU(
        num_classes=self.num_classes,
        use_lidar_mask=False,
        use_image_mask=True, eval_tr=i, with_others=self.with_others) for i in self.eval_threshold_range]
            
        general_IoU = [IoU(use_image_mask=True, eval_tr=i) for i in self.eval_threshold_range]
        
        for index, occ_pred in enumerate(occ_results):
            info = self.data_infos[index]

            occ_path = os.path.join(self.gt_root, info['scene_name'], info['token'], 'labels.npz')
            occ_gt = np.load(occ_path)
            gt_semantics = occ_gt['semantics']
            mask_lidar = occ_gt['mask_lidar'].astype(bool)
            mask_camera = occ_gt['mask_camera'].astype(bool)
            if not self.with_others:
                other_mask = (gt_semantics == 0) | (gt_semantics == 12)
                mask_camera = mask_camera & (~other_mask)

            preds = occ_pred['occupancy']
            for i, t in enumerate(self.eval_threshold_range):
                preds_i = preds.copy()
                preds_i[occ_pred['free_space'][i]] = 17
                general_IoU[i].add_batch(preds_i, gt_semantics, mask_lidar, mask_camera)
                semantic_mIoU[i].add_batch(preds_i, gt_semantics, mask_lidar, mask_camera)

        top_mIoU = 0
        top_IoU = 0
        for i, t in enumerate(self.eval_threshold_range):
            print(f"Eval threshold {t}:")
            general_iou_metric = general_IoU[i]
            iou = general_iou_metric.count_miou()[1][0]
            eval_dict.update({f'IoU_{t}': iou * 100})
            if iou > top_IoU:
                top_IoU = iou

            metric = semantic_mIoU[i]
            miou = metric.count_miou()[2]
            eval_dict.update({f'mIoU_{t}': miou * 100})
            if miou > top_mIoU:
                top_mIoU = miou
        eval_dict.update({'top_mIoU': top_mIoU})
        eval_dict.update({'top_IoU': top_IoU})

        return eval_dict
