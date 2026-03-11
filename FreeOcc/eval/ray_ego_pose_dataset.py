"""
Helper dataset for RayIoU / RayPQ origin sampling.
Computation is aligned with SparseOcc/loaders/ego_pose_dataset.py.
"""

import numpy as np
import torch
from pyquaternion import Quaternion
from torch.utils.data import Dataset

np.set_printoptions(precision=3, suppress=True)


def build_transform_matrix(translation, rotation):
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation.rotation_matrix
    transform_matrix[:3, 3] = translation
    return transform_matrix


class EgoPoseDataset(Dataset):
    def __init__(self, data_infos, sort_scene_frames=True):
        super().__init__()

        self.data_infos = data_infos
        self.sort_scene_frames = sort_scene_frames
        self.scene_frames = {}

        for info in data_infos:
            scene_name = info["scene_name"]
            if scene_name not in self.scene_frames:
                self.scene_frames[scene_name] = []
            self.scene_frames[scene_name].append(info)
        if self.sort_scene_frames:
            for scene_name in self.scene_frames:
                self.scene_frames[scene_name] = sorted(
                    self.scene_frames[scene_name],
                    key=lambda info: int(info.get("timestamp", 0)),
                )

    def __len__(self):
        return len(self.data_infos)

    def get_ego_from_lidar(self, info):
        ego_from_lidar = build_transform_matrix(
            np.array(info["lidar2ego_translation"]),
            Quaternion(info["lidar2ego_rotation"]),
        )
        return ego_from_lidar

    def get_global_pose(self, info, inverse=False):
        global_from_ego = build_transform_matrix(
            np.array(info["ego2global_translation"]),
            Quaternion(info["ego2global_rotation"]),
        )
        ego_from_lidar = build_transform_matrix(
            np.array(info["lidar2ego_translation"]),
            Quaternion(info["lidar2ego_rotation"]),
        )
        pose = global_from_ego.dot(ego_from_lidar)
        if inverse:
            pose = np.linalg.inv(pose)
        return pose

    def __getitem__(self, idx):
        info = self.data_infos[idx]

        ref_sample_token = info["token"]
        ref_lidar_from_global = self.get_global_pose(info, inverse=True)
        ref_ego_from_lidar = self.get_ego_from_lidar(info)

        scene_frame = self.scene_frames[info["scene_name"]]
        ref_index = scene_frame.index(info)

        output_origin_list = []
        for curr_index in range(len(scene_frame)):
            if curr_index == ref_index:
                origin_tf = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            else:
                global_from_curr = self.get_global_pose(scene_frame[curr_index], inverse=False)
                ref_from_curr = ref_lidar_from_global.dot(global_from_curr)
                origin_tf = np.array(ref_from_curr[:3, 3], dtype=np.float32)

            origin_tf_pad = np.ones([4])
            origin_tf_pad[:3] = origin_tf
            origin_tf = np.dot(ref_ego_from_lidar[:3], origin_tf_pad.T).T

            if np.abs(origin_tf[0]) < 39 and np.abs(origin_tf[1]) < 39:
                output_origin_list.append(origin_tf)

        if len(output_origin_list) > 8:
            select_idx = np.round(np.linspace(0, len(output_origin_list) - 1, 8)).astype(np.int64)
            output_origin_list = [output_origin_list[i] for i in select_idx]

        output_origin_tensor = torch.from_numpy(np.stack(output_origin_list))
        return (ref_sample_token, output_origin_tensor)


# Backward-compatible alias for older imports.
trans_matrix = build_transform_matrix
