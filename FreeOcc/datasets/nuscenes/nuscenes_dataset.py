import os
import shutil
import argparse
import logging

from tqdm import tqdm
logging.basicConfig(format='[%(module)s | l.%(lineno)d] %(message)s')
logging.getLogger().setLevel(logging.INFO)
import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
from FreeOcc.datasets.nuscenes.utils_nuscenes import (
    get_scene_from_name,
    get_closest_camera_tokens,
    get_T_camera_to_ref_camera,
    get_T_ego_to_camera,
)




def load_data(
             nusc,
             scene_name,
             num_samples=-1,
             sample_every=1,
             starting_from=0,
             camera_channels=["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"],
             load_image_paths=True,
             load_intrinsics=True,
             load_extrinsics=True,
             ):
    """
    Load data from a NuScenes scene.
    
    Returns a list of dicts, each containing:
    - timestamp: timestamp of the sample, from the lidar
    - image_paths: dict mapping camera channel to image path
    - extrinsics: dict mapping camera channel to 4x4 camera-to-reference-camera transform matrix
                  (all poses expressed in the frame of the first CAM_FRONT image)
    - intrinsics: dict mapping camera channel to intrinsic parameters (fx, fy, cx, cy)
    """
    scenes = get_scene_from_name(nusc, scene_name)
    if not scenes:
        logging.warning(f"No scene found with name {scene_name}")
        return []
    if len(scenes) > 1:
        logging.warning(f"Multiple scenes found with name {scene_name}, using first one")
    scene = scenes[0]
    
    results = []
    sample_count = 0
    sample_token = scene['first_sample_token']
    for _ in range(starting_from):
        if not sample_token or sample_token == '':
            return results
        sample = nusc.get('sample', sample_token)
        sample_token = sample['next']
    if not sample_token or sample_token == '':
        return results
    ref_sample = nusc.get('sample', sample_token)
    ref_lidar_token = ref_sample['data']['LIDAR_TOP']
    camera_tokens_dict_ref = get_closest_camera_tokens(nusc, ref_lidar_token)
    ref_camera_token = camera_tokens_dict_ref['CAM_FRONT']
    ref_camera_data = nusc.get('sample_data', ref_camera_token)
    from scipy.spatial.transform import Rotation as R
    
    while sample_token and sample_token != '' and (num_samples == -1 or sample_count < num_samples*sample_every):
        sample = nusc.get('sample', sample_token)
        if sample_count % sample_every != 0:
            sample_token = sample['next']
            sample_count += 1
            continue
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = nusc.get('sample_data', lidar_token)
        timestamp = lidar_data['timestamp']
        camera_tokens_dict = get_closest_camera_tokens(nusc, lidar_token)
        sample_dict = {
            'timestamp': timestamp,
            'sample_token': sample_token,
            'image_paths': {},
            'extrinsics': {},
            'intrinsics': {},
            'ref_cam_to_ego': {},
        }
        ref_camera_to_ego = np.linalg.inv(
            get_T_ego_to_camera(nusc, ref_camera_data, lidar_data)
        )
        sample_dict['ref_cam_to_ego'] = ref_camera_to_ego
        for camera_channel in camera_channels:
            camera_token = camera_tokens_dict[camera_channel]
            camera_data = nusc.get('sample_data', camera_token)
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_data = nusc.get('sample_data', lidar_token)
            if load_image_paths:
                image_path = os.path.join(nusc.dataroot, camera_data['filename'])
                sample_dict['image_paths'][camera_channel] = image_path
            if load_intrinsics:
                calibrated_sensor = nusc.get('calibrated_sensor', camera_data['calibrated_sensor_token'])
                intrinsic_matrix = np.array(calibrated_sensor['camera_intrinsic'])
                fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
                cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
                sample_dict['intrinsics'][camera_channel] = {
                    'fx': fx,
                    'fy': fy,
                    'cx': cx,
                    'cy': cy
                }
            if load_extrinsics:
                T_cam_to_ref_cam = get_T_camera_to_ref_camera(nusc, camera_data, ref_camera_data)
                
                sample_dict['extrinsics'][camera_channel] = T_cam_to_ref_cam
        
        results.append(sample_dict)
        sample_count += 1
        sample_token = sample['next']
    
    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--output_folder', help='', default="nuscenes_images")
    parser.add_argument('--nuscenes_scenes', help='List of scenes of nuscenes to use (int indices)', nargs='+', type=int, default=None)
    parser.add_argument('--nuscenes_path', help='', default="/media/andrew/andrewSSD/datasets/nuscenes")
    parser.add_argument('--nuscenes_mode', help='', default="trainval")
    parser.add_argument('--cam', help='', default="CAM_FRONT", type=str)

    args = parser.parse_args()

    main(args)
