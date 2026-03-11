import os
import shutil
import argparse
import logging
logging.basicConfig(format='[%(module)s | l.%(lineno)d] %(message)s')
logging.getLogger().setLevel(logging.INFO)
import numpy as np
import json
from FreeOcc.configs.config_loader import (
    get_occ3d_nuscenes_class_mapping,
    get_occ3d_nuscenes_class_names,
)

eval_class_names_occ3d_nuscenes = get_occ3d_nuscenes_class_names()
eval_class_name_to_id_occ3d_nuscenes = {
    class_name: i for i, class_name in enumerate(eval_class_names_occ3d_nuscenes)
}
prompt_to_occ3d_nuscenes_class_name_map = get_occ3d_nuscenes_class_mapping()

occ3d_nuscenes_classes_names = eval_class_names_occ3d_nuscenes
occ3d_nuscenes_class_to_id = eval_class_name_to_id_occ3d_nuscenes
class_mapping_to_occ3d_nuscenes = prompt_to_occ3d_nuscenes_class_name_map


def main(args):
    logging.info("args = %s", args)
    
    prompt_class_name_to_id = json.load(open(args.class_name_to_id_json_path, 'r'))
    prompt_class_name_to_id['free'] = -1
    prompt_class_name_to_id['unlabeled'] = 0
    logging.info('\nprompt_class_name_to_id :')
    for k,v in prompt_class_name_to_id.items():
        logging.info('%s -> %s',k,v)

    logging.info('\nprompt_to_occ3d_nuscenes_class_name_map :')
    for k,v in prompt_to_occ3d_nuscenes_class_name_map.items():
        logging.info('%s -> %s',k,v)

    logging.info('\neval_class_name_to_id_occ3d_nuscenes :')
    for k,v in eval_class_name_to_id_occ3d_nuscenes.items():
        logging.info('%s -> %s',k,v)


    semantic_occ_grid = np.load(args.semantic_occ_grid_path)
    logging.info('semantic_occ_grid.shape=%s',semantic_occ_grid.shape)
    logging.info('np.unique(semantic_occ_grid, return_counts=True)=%s',np.unique(semantic_occ_grid, return_counts=True))

    semantic_occ_grid = convert_semantic_occ_grid_to_occ3d_nuscenes(
        semantic_occ_grid,
        prompt_class_name_to_id,
    )
    logging.info('semantic_occ_grid.shape=%s',semantic_occ_grid.shape)
    logging.info('np.unique(semantic_occ_grid, return_counts=True)=%s',np.unique(semantic_occ_grid, return_counts=True))

    np.save(args.output_path, semantic_occ_grid)


def convert_semantic_occ_grid_to_occ3d_nuscenes(
    semantic_occ_grid,
    prompt_class_name_to_id=None,
    class_name_to_id=None,
):
    """ Convert an occ grid labeled using prompt class IDs
    to an occ grid labeled using Occ3D-NuScenes evaluation class IDs
    """

    if prompt_class_name_to_id is None:
        prompt_class_name_to_id = class_name_to_id
    if prompt_class_name_to_id is None:
        raise ValueError(
            "convert_semantic_occ_grid_to_occ3d_nuscenes expects "
            "prompt_class_name_to_id (or legacy class_name_to_id)."
        )

    prompt_class_name_to_id['free'] = -1
    prompt_class_name_to_id['unlabeled'] = 0
    semantic_occ_grid = semantic_occ_grid.copy()
    new_semantic_occ_grid = np.zeros_like(semantic_occ_grid)
    original_id_to_occ3d_nuscenes_id = {}
    for class_name in prompt_class_name_to_id:
        original_class_id = prompt_class_name_to_id[class_name]
        target_class_name = prompt_to_occ3d_nuscenes_class_name_map.get(class_name)
        if target_class_name is None:
            normalized = class_name.strip().lower()
            candidates = [
                normalized,
                normalized.replace("_", " "),
                normalized.replace(" ", "_"),
                normalized.replace("-", " "),
                normalized.replace("-", "_"),
            ]
            for candidate in candidates:
                if candidate in prompt_to_occ3d_nuscenes_class_name_map:
                    target_class_name = prompt_to_occ3d_nuscenes_class_name_map[candidate]
                    break

        if target_class_name is None:
            logging.warning(
                "Class '%s' is not in prompt_to_occ3d_nuscenes_class_name_map; mapping to 'others'.",
                class_name,
            )
            target_class_name = "others"

        occ3d_nuscenes_class_id = eval_class_name_to_id_occ3d_nuscenes[target_class_name]
        original_id_to_occ3d_nuscenes_id[original_class_id] = occ3d_nuscenes_class_id

    for original_class_id, occ3d_nuscenes_class_id in original_id_to_occ3d_nuscenes_id.items():
        new_semantic_occ_grid[semantic_occ_grid == original_class_id] = occ3d_nuscenes_class_id
    
    return new_semantic_occ_grid
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--class_name_to_id_json_path', help='', default=None)
    parser.add_argument('--semantic_occ_grid_path', help='', default=None)
    parser.add_argument('--output_path', help='', default=None)
    args = parser.parse_args()

    main(args)
