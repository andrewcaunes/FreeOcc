"""
Direct NuScenes info loading for RayIoU/RayPQ evaluation.
This module intentionally avoids MMDet3D/SparseOcc PKL dependencies.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from FreeOcc.datasets.nuscenes.utils_nuscenes import get_ray_info_from_sample_token

nuscenes_by_token_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
nuscenes_object_cache: Dict[str, Any] = {}


def resolve_nuscenes_version(nuscenes_root: Path) -> str:
    for version in ["v1.0-trainval", "v1.0-mini", "v1.0-test"]:
        if (nuscenes_root / version).exists():
            return version
    return "v1.0-trainval"


def load_nuscenes_info_by_token(
    nuscenes_root: Union[str, Path],
) -> Dict[str, Dict[str, Any]]:
    nuscenes_root = Path(nuscenes_root).resolve()
    version = resolve_nuscenes_version(nuscenes_root)
    cache_key = f"{nuscenes_root}::{version}"
    if cache_key in nuscenes_by_token_cache:
        return nuscenes_by_token_cache[cache_key]

    if cache_key not in nuscenes_object_cache:
        from nuscenes import NuScenes

        logging.info(
            "Loading NuScenes metadata for ray metrics from %s (version=%s)",
            nuscenes_root,
            version,
        )
        nuscenes_object_cache[cache_key] = NuScenes(
            version=version,
            dataroot=str(nuscenes_root),
            verbose=False,
        )
    nusc = nuscenes_object_cache[cache_key]

    info_by_token: Dict[str, Dict[str, Any]] = {}
    for sample in nusc.sample:
        token = sample["token"]
        ray_info = get_ray_info_from_sample_token(
            nusc=nusc,
            sample_token=token,
        )
        if ray_info is None:
            continue
        info_by_token[token] = ray_info

    nuscenes_by_token_cache[cache_key] = info_by_token
    logging.info("Prepared %d direct NuScenes ray infos", len(info_by_token))
    return info_by_token


def build_scene_infos(
    sample_tokens: List[str],
    scene_name: str,
    nuscenes_root: Union[str, Path],
    nuscenes_split: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    if nuscenes_split not in [None, "", "train", "val", "trainval", "all", "test"]:
        logging.warning(
            "Unknown nuscenes_split=%s; proceeding with direct metadata lookup.",
            nuscenes_split,
        )

    sample_tokens = [str(token) for token in sample_tokens]
    info_by_token = load_nuscenes_info_by_token(nuscenes_root=nuscenes_root)

    scene_infos: List[Dict[str, Any]] = []
    missing_tokens: List[str] = []
    for token in sample_tokens:
        info = info_by_token.get(token)
        if info is None:
            missing_tokens.append(token)
            continue
        info_copy = dict(info)
        if "scene_name" not in info_copy or info_copy["scene_name"] in [None, ""]:
            info_copy["scene_name"] = scene_name
        scene_infos.append(info_copy)

    logging.info(
        "Ray infos resolved from raw NuScenes metadata (%d/%d scene tokens found)",
        len(scene_infos),
        len(sample_tokens),
    )
    return scene_infos, missing_tokens


# Backward-compatible aliases for older call sites.
resolveNuScenesVersion = resolve_nuscenes_version
loadNuScenesInfoByToken = load_nuscenes_info_by_token
buildSceneInfos = build_scene_infos
