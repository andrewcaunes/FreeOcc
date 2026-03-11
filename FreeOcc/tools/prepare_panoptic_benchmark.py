
from __future__ import annotations

import argparse
import glob
import json
import logging
import multiprocessing
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import tqdm
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import points_in_box
from pyquaternion import Quaternion

try:
    from nuscenes.utils.splits import create_splits_scenes
except Exception:  # pragma: no cover - import-time guard
    create_splits_scenes = None

logging.basicConfig(format='[%(module)s | l.%(lineno)d] %(message)s')
logging.getLogger().setLevel(logging.INFO)


occClassNames = [
    "others",
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
    "free",
]
detClassNames = {
    "car",
    "truck",
    "trailer",
    "bus",
    "construction_vehicle",
    "bicycle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "barrier",
}
occClassNameToId = {className: classId for classId, className in enumerate(occClassNames)}

workerVoxelPoints: np.ndarray | None = None
workerNumClasses: int = 18
workerBoxMargin: float = 0.8
workerInstanceDtype: Any = np.uint16


def resolveOccGtRoot(occ3dRoot: Path, inputLabelName: str) -> Path:
    candidateRoots = [occ3dRoot / "gts", occ3dRoot]
    for candidateRoot in candidateRoots:
        if not candidateRoot.is_dir():
            continue
        pattern = str(candidateRoot / "*" / "*" / inputLabelName)
        if glob.glob(pattern):
            logging.info("Using occupancy GT root: %s", candidateRoot)
            return candidateRoot
    for candidateRoot in candidateRoots:
        if candidateRoot.is_dir():
            logging.info(
                "Using occupancy GT root: %s (no %s indexed yet; will validate later)",
                candidateRoot,
                inputLabelName,
            )
            return candidateRoot
    raise FileNotFoundError(f"Could not find occupancy root under {occ3dRoot}")


def resolveFirstExistingInfoPath(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolveInfoPaths(nuscenesRoot: Path, infoPath: str | None, sceneSplit: str) -> list[Path]:
    if infoPath is not None:
        resolved = Path(infoPath)
        if not resolved.exists():
            raise FileNotFoundError(f"--info_path not found: {resolved}")
        return [resolved]

    valCandidates = [
        nuscenesRoot / "nuscenes_infos_val_sweep.pkl",
        nuscenesRoot / "nuscenes_infos_val.pkl",
    ]
    trainCandidates = [
        nuscenesRoot / "nuscenes_infos_train_sweep.pkl",
        nuscenesRoot / "nuscenes_infos_train.pkl",
    ]

    if sceneSplit == "val":
        selected = resolveFirstExistingInfoPath(valCandidates)
        if selected is None:
            raise FileNotFoundError(
                "Could not find val infos pkl automatically. Pass --info_path explicitly."
            )
        logging.info("Using infos file: %s", selected)
        return [selected]

    if sceneSplit == "train":
        selected = resolveFirstExistingInfoPath(trainCandidates)
        if selected is None:
            raise FileNotFoundError(
                "Could not find train infos pkl automatically. Pass --info_path explicitly."
            )
        logging.info("Using infos file: %s", selected)
        return [selected]

    selectedPaths: list[Path] = []
    selectedTrain = resolveFirstExistingInfoPath(trainCandidates)
    selectedVal = resolveFirstExistingInfoPath(valCandidates)
    if selectedTrain is not None:
        selectedPaths.append(selectedTrain)
    else:
        logging.warning("Could not find train infos pkl for --scene_split=all.")
    if selectedVal is not None:
        selectedPaths.append(selectedVal)
    else:
        logging.warning("Could not find val infos pkl for --scene_split=all.")

    if len(selectedPaths) == 0:
        raise FileNotFoundError(
            "Could not find train/val infos pkl automatically. Pass --info_path explicitly."
        )

    for selectedPath in selectedPaths:
        logging.info("Using infos file: %s", selectedPath)
    return selectedPaths


def inferNuscenesClassIdToName(metainfo: dict[str, Any] | None) -> dict[int, str]:
    defaultMapping = {
        0: "car",
        1: "truck",
        2: "trailer",
        3: "bus",
        4: "construction_vehicle",
        5: "bicycle",
        6: "motorcycle",
        7: "pedestrian",
        8: "traffic_cone",
        9: "barrier",
    }
    if metainfo is None:
        return defaultMapping

    categories = metainfo.get("categories")
    if categories is None:
        return defaultMapping
    if isinstance(categories, dict):
        idToName: dict[int, str] = {}
        for className, classId in categories.items():
            if isinstance(classId, (int, np.integer)):
                idToName[int(classId)] = str(className)
        if len(idToName) > 0:
            return idToName
    if isinstance(categories, list):
        idToName = {}
        for idx, className in enumerate(categories):
            idToName[idx] = str(className)
        if len(idToName) > 0:
            return idToName
    return defaultMapping


def convertMmdet3dDataListToInfos(
    dataList: list[dict[str, Any]],
    metainfo: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    labelToName = inferNuscenesClassIdToName(metainfo)
    convertedInfos: list[dict[str, Any]] = []
    missingLidar2ego = 0
    unknownLabelCount = 0

    for sample in dataList:
        token = sample.get("token")
        if token is None:
            token = str(sample.get("sample_idx", ""))

        instances = sample.get("instances", [])
        gtBoxes: list[list[float]] = []
        gtNames: list[str] = []
        for instance in instances:
            bbox3d = instance.get("bbox_3d")
            label3d = instance.get("bbox_label_3d")
            if bbox3d is None or label3d is None:
                continue
            try:
                labelId = int(label3d)
            except Exception:
                continue
            className = labelToName.get(labelId)
            if className is None:
                unknownLabelCount += 1
                continue

            bboxArray = np.asarray(bbox3d, dtype=np.float32)
            if bboxArray.shape[0] < 7:
                continue
            gtBoxes.append(bboxArray[:7].tolist())
            gtNames.append(className)

        lidar2egoRotation = [1.0, 0.0, 0.0, 0.0]
        lidar2egoTranslation = [0.0, 0.0, 0.0]
        lidarPoints = sample.get("lidar_points", {})
        lidar2ego = lidarPoints.get("lidar2ego")
        if lidar2ego is not None:
            lidar2egoArray = np.asarray(lidar2ego, dtype=np.float32)
            if lidar2egoArray.shape == (4, 4):
                rotationMatrix = lidar2egoArray[:3, :3]
                translation = lidar2egoArray[:3, 3]
                rotationQuat = Quaternion(matrix=rotationMatrix)
                lidar2egoRotation = [rotationQuat.w, rotationQuat.x, rotationQuat.y, rotationQuat.z]
                lidar2egoTranslation = translation.tolist()
            else:
                missingLidar2ego += 1
        else:
            missingLidar2ego += 1

        convertedInfos.append(
            {
                "token": str(token),
                "scene_name": sample.get("scene_name"),
                "gt_boxes": np.asarray(gtBoxes, dtype=np.float32).reshape((-1, 7)) if len(gtBoxes) > 0 else np.zeros((0, 7), dtype=np.float32),
                "gt_names": np.asarray(gtNames, dtype=object),
                "lidar2ego_rotation": lidar2egoRotation,
                "lidar2ego_translation": lidar2egoTranslation,
            }
        )

    logging.info(
        "Converted MMDet3D v1 data_list -> infos: %d samples (unknown labels skipped=%d, missing/invalid lidar2ego=%d)",
        len(convertedInfos),
        unknownLabelCount,
        missingLidar2ego,
    )
    return convertedInfos


def loadInfos(infoPath: Path) -> list[dict[str, Any]]:
    with open(infoPath, "rb") as fileHandle:
        loaded = pickle.load(fileHandle)
    if isinstance(loaded, dict):
        infos = loaded.get("infos")
        if infos is not None:
            if not isinstance(infos, list):
                raise ValueError(f"'infos' in {infoPath} is not a list")
            return infos

        dataList = loaded.get("data_list")
        if dataList is not None:
            if not isinstance(dataList, list):
                raise ValueError(f"'data_list' in {infoPath} is not a list")
            metainfo = loaded.get("metainfo")
            if metainfo is not None and not isinstance(metainfo, dict):
                metainfo = None
            return convertMmdet3dDataListToInfos(dataList, metainfo)

        raise ValueError(f"'infos' key not found in {infoPath}")
    if isinstance(loaded, list):
        return loaded
    raise ValueError(f"Unsupported info structure in {infoPath}: {type(loaded)}")


def loadInfosFromPaths(infoPaths: list[Path]) -> tuple[list[dict[str, Any]], int]:
    mergedInfos: list[dict[str, Any]] = []
    seenTokens: set[str] = set()
    duplicateTokens = 0

    for infoPath in infoPaths:
        infos = loadInfos(infoPath)
        logging.info("Loaded %d infos from %s", len(infos), infoPath)
        for info in infos:
            token = str(info.get("token", ""))
            if token != "":
                if token in seenTokens:
                    duplicateTokens += 1
                    continue
                seenTokens.add(token)
            mergedInfos.append(info)

    if duplicateTokens > 0:
        logging.info("Dropped %d duplicate tokens while merging infos files.", duplicateTokens)
    return mergedInfos, duplicateTokens


def buildTokenMaps(occGtRoot: Path, inputLabelName: str) -> tuple[dict[str, Path], dict[str, str], int]:
    pattern = str(occGtRoot / "*" / "*" / inputLabelName)
    labelPaths = sorted(glob.glob(pattern))
    tokenToPath: dict[str, Path] = {}
    tokenToScene: dict[str, str] = {}
    duplicateCount = 0

    for labelPath in labelPaths:
        path = Path(labelPath)
        token = path.parent.name
        sceneName = path.parent.parent.name
        if token in tokenToPath:
            duplicateCount += 1
        tokenToPath[token] = path
        tokenToScene[token] = sceneName

    logging.info(
        "Indexed %d occupancy labels (%d duplicate tokens overwritten)",
        len(tokenToPath),
        duplicateCount,
    )
    return tokenToPath, tokenToScene, duplicateCount


def decodeNames(gtNamesRaw: Any) -> list[str]:
    if gtNamesRaw is None:
        return []
    decodedNames: list[str] = []
    for className in np.asarray(gtNamesRaw).tolist():
        if isinstance(className, bytes):
            decodedNames.append(className.decode("utf-8"))
        else:
            decodedNames.append(str(className))
    return decodedNames


def buildVoxelCenters(occSize: list[int], pointCloudRange: list[float]) -> np.ndarray:
    width, height, depth = occSize
    xMin, yMin, zMin, xMax, yMax, zMax = pointCloudRange

    xs = np.linspace(xMin + (xMax - xMin) / (2 * width), xMax - (xMax - xMin) / (2 * width), width)
    ys = np.linspace(yMin + (yMax - yMin) / (2 * height), yMax - (yMax - yMin) / (2 * height), height)
    zs = np.linspace(zMin + (zMax - zMin) / (2 * depth), zMax - (zMax - zMin) / (2 * depth), depth)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).astype(np.float32)
    return grid


def convertBoxesToNuscenes(gtBoxes: np.ndarray) -> list[Box | None]:
    boxes: list[Box | None] = []
    for boxValues in gtBoxes:
        boxArray = np.asarray(boxValues, dtype=np.float32)
        if boxArray.shape[0] < 7:
            boxes.append(None)
            continue
        yaw = -float(boxArray[6]) - np.pi / 2.0
        orientation = Quaternion(axis=[0, 0, 1], radians=yaw).inverse
        boxes.append(
            Box(
                center=[float(boxArray[0]), float(boxArray[1]), float(boxArray[2])],
                size=[float(boxArray[3]), float(boxArray[4]), float(boxArray[5])],
                orientation=orientation,
            )
        )
    return boxes


def initWorker(
    occSize: list[int],
    pointCloudRange: list[float],
    numClasses: int,
    boxMargin: float,
    instanceDtype: str,
) -> None:
    global workerVoxelPoints, workerNumClasses, workerBoxMargin, workerInstanceDtype
    workerVoxelPoints = buildVoxelCenters(occSize, pointCloudRange)
    workerNumClasses = numClasses
    workerBoxMargin = boxMargin
    workerInstanceDtype = np.dtype(instanceDtype)


def computeInstances(
    semantics: np.ndarray,
    gtBoxes: np.ndarray,
    gtNames: list[str],
    lidar2egoRotation: list[float],
    lidar2egoTranslation: list[float],
) -> tuple[np.ndarray, int]:
    if workerVoxelPoints is None:
        raise RuntimeError("Worker was not initialized")
    if semantics.shape != workerVoxelPoints.shape[:3]:
        raise ValueError(
            f"Unexpected semantics shape {semantics.shape}, expected {workerVoxelPoints.shape[:3]}"
        )

    freeClassId = workerNumClasses - 1
    validMask = semantics < freeClassId
    instances = np.zeros(semantics.shape, dtype=workerInstanceDtype)
    if not np.any(validMask):
        return instances, 0

    flattenedSemantics = semantics[validMask]
    flattenedInstances = instances[validMask]
    flattenedPoints = workerVoxelPoints[validMask]

    boxes = convertBoxesToNuscenes(gtBoxes)
    numBoxes = min(len(boxes), len(gtNames))
    currentInstanceId = 1
    instanceBoxes: list[Box] = []
    instanceClassIds: list[int] = []

    for boxIndex in range(numBoxes):
        className = gtNames[boxIndex]
        if className not in occClassNameToId:
            continue
        if boxes[boxIndex] is None:
            continue
        classId = occClassNameToId[className]
        box = boxes[boxIndex]
        box.rotate(Quaternion(lidar2egoRotation))
        box.translate(np.asarray(lidar2egoTranslation, dtype=np.float32))

        mask = points_in_box(box, flattenedPoints.transpose(1, 0))
        if not np.any(mask):
            continue
        mask[mask] = flattenedSemantics[mask] == classId
        mask[mask] = flattenedInstances[mask] == 0
        if np.any(mask):
            flattenedInstances[mask] = currentInstanceId
            currentInstanceId += 1

            enlargedBox = box.copy()
            enlargedBox.wlh = enlargedBox.wlh + workerBoxMargin
            instanceBoxes.append(enlargedBox)
            instanceClassIds.append(classId)

    presentClassIds = set(np.unique(flattenedSemantics).tolist())
    for classId, className in enumerate(occClassNames):
        if classId not in presentClassIds:
            continue
        if className in detClassNames or className == "free":
            continue
        classMask = flattenedSemantics == classId
        flattenedInstances[classMask] = currentInstanceId
        currentInstanceId += 1

    uncoverIndices = np.where(flattenedInstances == 0)[0]
    if uncoverIndices.size > 0 and len(instanceBoxes) > 0:
        uncoverPoints = flattenedPoints[uncoverIndices]
        uncoverSemantics = flattenedSemantics[uncoverIndices]
        uncoverInstances = np.zeros(uncoverPoints.shape[0], dtype=workerInstanceDtype)
        uncoverDist = np.full(uncoverPoints.shape[0], 1e8, dtype=np.float32)

        for instanceIndex, box in enumerate(instanceBoxes):
            predictedInstanceId = instanceIndex + 1
            classId = instanceClassIds[instanceIndex]
            mask = points_in_box(box, uncoverPoints.transpose(1, 0))
            if not np.any(mask):
                continue
            mask[uncoverSemantics != classId] = False
            if not np.any(mask):
                continue
            dist = np.sum((box.center - uncoverPoints) ** 2, axis=-1)
            mask[dist >= uncoverDist] = False
            if not np.any(mask):
                continue
            uncoverDist[mask] = dist[mask]
            uncoverInstances[mask] = predictedInstanceId

        flattenedInstances[uncoverIndices] = uncoverInstances

    instances[validMask] = flattenedInstances
    unassignedNonFree = int(np.sum((instances == 0) & validMask))
    return instances, unassignedNonFree


def processSample(task: dict[str, Any]) -> dict[str, Any]:
    token = task["token"]
    inputPath = task["input_path"]
    outputPath = task["output_path"]
    try:
        with np.load(inputPath, allow_pickle=False) as occLabels:
            if "semantics" not in occLabels:
                return {"status": "error", "token": token, "message": "Missing semantics key"}
            semantics = occLabels["semantics"]
            instances, unassignedNonFree = computeInstances(
                semantics=semantics,
                gtBoxes=task["gt_boxes"],
                gtNames=task["gt_names"],
                lidar2egoRotation=task["lidar2ego_rotation"],
                lidar2egoTranslation=task["lidar2ego_translation"],
            )
            outputLabels = {key: occLabels[key] for key in occLabels.files}

        outputLabels["instances"] = instances
        Path(outputPath).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(outputPath, **outputLabels)

        contiguousIds = np.unique(instances).shape[0] == int(instances.max()) + 1
        return {
            "status": "ok",
            "token": token,
            "scene_name": task["scene_name"],
            "unassigned_non_free": unassignedNonFree,
            "contiguous_ids": bool(contiguousIds),
        }
    except Exception as exception:  # pragma: no cover - worker guard
        return {
            "status": "error",
            "token": token,
            "scene_name": task.get("scene_name"),
            "message": repr(exception),
        }


def loadOfficialValScenes() -> list[str]:
    if create_splits_scenes is None:
        logging.warning("Could not import nuScenes create_splits_scenes().")
        return []
    try:
        splits = create_splits_scenes()
    except Exception as exception:
        logging.warning("Could not load official nuScenes splits (%s).", exception)
        return []
    valScenes = splits.get("val", [])
    if len(valScenes) == 0:
        logging.warning("Official nuScenes val split is empty from create_splits_scenes().")
    return valScenes


def filterInfosForSplit(
    infos: list[dict[str, Any]],
    tokenToScene: dict[str, str],
    sceneSplit: str,
    valScenes: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    useValFilter = sceneSplit == "val" and len(valScenes) > 0
    valSceneSet = set(valScenes) if useValFilter else None

    selectedInfos: list[dict[str, Any]] = []
    selectedScenes: set[str] = set()
    seenTokens: set[str] = set()
    duplicateTokens = 0
    missingToken = 0
    droppedNonSplit = 0
    unknownScene = 0

    for info in infos:
        token = str(info.get("token", ""))
        if token == "":
            missingToken += 1
            continue
        if token in seenTokens:
            duplicateTokens += 1
            continue
        seenTokens.add(token)

        sceneName = info.get("scene_name")
        if sceneName is None or sceneName == "":
            sceneName = tokenToScene.get(token)
        if sceneName is None:
            unknownScene += 1
        if valSceneSet is not None and sceneName is not None and sceneName not in valSceneSet:
            droppedNonSplit += 1
            continue

        infoCopy = dict(info)
        if sceneName is not None:
            infoCopy["scene_name"] = sceneName
            selectedScenes.add(sceneName)
        selectedInfos.append(infoCopy)

    stats: dict[str, Any] = {
        "selected_infos": len(selectedInfos),
        "selected_scenes": sorted(selectedScenes),
        "selected_scene_count": len(selectedScenes),
        "duplicate_tokens": duplicateTokens,
        "missing_token": missingToken,
        "unknown_scene": unknownScene,
        "dropped_non_split": droppedNonSplit,
        "use_val_filter": useValFilter,
        "val_scene_count_reference": len(valScenes),
    }
    return selectedInfos, stats


def makeTasks(
    infos: list[dict[str, Any]],
    tokenToPath: dict[str, Path],
    outputLabelName: str,
    overwrite: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    alreadyDone = 0
    missingInputLabels = 0
    missingAnnData = 0

    for info in infos:
        token = str(info.get("token", ""))
        if token == "":
            missingAnnData += 1
            continue
        inputPath = tokenToPath.get(token)
        if inputPath is None:
            missingInputLabels += 1
            continue
        outputPath = inputPath.with_name(outputLabelName)
        if outputPath.exists() and not overwrite:
            alreadyDone += 1
            continue

        gtBoxesRaw = info.get("gt_boxes")
        if gtBoxesRaw is None:
            gtBoxes = np.zeros((0, 7), dtype=np.float32)
        else:
            gtBoxes = np.asarray(gtBoxesRaw, dtype=np.float32)
            if gtBoxes.ndim == 1:
                gtBoxes = gtBoxes.reshape(1, -1)
            if gtBoxes.shape[0] == 0:
                gtBoxes = np.zeros((0, 7), dtype=np.float32)
        gtNames = decodeNames(info.get("gt_names"))
        if gtBoxes.shape[0] != len(gtNames):
            minLength = min(gtBoxes.shape[0], len(gtNames))
            gtBoxes = gtBoxes[:minLength]
            gtNames = gtNames[:minLength]

        lidar2egoRotation = info.get("lidar2ego_rotation", [1.0, 0.0, 0.0, 0.0])
        lidar2egoTranslation = info.get("lidar2ego_translation", [0.0, 0.0, 0.0])

        tasks.append(
            {
                "token": token,
                "scene_name": info.get("scene_name"),
                "input_path": str(inputPath),
                "output_path": str(outputPath),
                "gt_boxes": gtBoxes,
                "gt_names": gtNames,
                "lidar2ego_rotation": list(np.asarray(lidar2egoRotation, dtype=np.float32).tolist()),
                "lidar2ego_translation": list(np.asarray(lidar2egoTranslation, dtype=np.float32).tolist()),
            }
        )

    stats = {
        "tasks_to_run": len(tasks),
        "already_done": alreadyDone,
        "missing_input_labels": missingInputLabels,
        "missing_annotation_data": missingAnnData,
    }
    return tasks, stats


def writeProgressSummary(progressPath: Path, summary: dict[str, Any]) -> None:
    progressPath.parent.mkdir(parents=True, exist_ok=True)
    with open(progressPath, "w") as fileHandle:
        json.dump(summary, fileHandle, indent=2)


def main(args: argparse.Namespace) -> None:
    logging.info("args = %s", args)
    startTime = time.time()

    if args.input_path is not None:
        args.nuscenes_root = args.input_path
    if args.output_path is not None:
        args.occ3d_root = args.output_path

    if args.output_label_name == args.input_label_name and not args.overwrite:
        raise ValueError(
            "--output_label_name equals --input_label_name. Use --overwrite if you intentionally replace labels."
        )
    if args.num_workers < 1:
        raise ValueError("--num_workers must be >= 1")
    if args.chunksize < 1:
        raise ValueError("--chunksize must be >= 1")

    nuscenesRoot = Path(args.nuscenes_root)
    occ3dRoot = Path(args.occ3d_root)
    occGtRoot = resolveOccGtRoot(occ3dRoot, args.input_label_name)
    infoPaths = resolveInfoPaths(nuscenesRoot, args.info_path, args.scene_split)
    infos, duplicateInfoTokens = loadInfosFromPaths(infoPaths)
    tokenToPath, tokenToScene, duplicateLabelTokens = buildTokenMaps(occGtRoot, args.input_label_name)
    if len(tokenToPath) == 0:
        raise FileNotFoundError(
            f"No {args.input_label_name} found under {occGtRoot}/<scene>/<token>/"
        )

    valScenes = loadOfficialValScenes() if args.scene_split == "val" else []
    selectedInfos, splitStats = filterInfosForSplit(
        infos=infos,
        tokenToScene=tokenToScene,
        sceneSplit=args.scene_split,
        valScenes=valScenes,
    )
    if args.max_samples is not None:
        selectedInfos = selectedInfos[: args.max_samples]
        logging.info("Using max_samples=%d, selected infos reduced to %d", args.max_samples, len(selectedInfos))

    if splitStats["use_val_filter"]:
        valSceneCount = splitStats["selected_scene_count"]
        if valSceneCount != splitStats["val_scene_count_reference"]:
            logging.warning(
                "Selected %d/%d val scenes from infos.",
                valSceneCount,
                splitStats["val_scene_count_reference"],
            )
        else:
            logging.info("Selected all %d val scenes.", valSceneCount)

    tasks, taskStats = makeTasks(
        infos=selectedInfos,
        tokenToPath=tokenToPath,
        outputLabelName=args.output_label_name,
        overwrite=args.overwrite,
    )

    totalSelected = len(selectedInfos)
    initialProgress = taskStats["already_done"] + taskStats["missing_input_labels"] + taskStats["missing_annotation_data"]
    if initialProgress > totalSelected:
        initialProgress = totalSelected

    progressPath = (
        Path(args.progress_path)
        if args.progress_path is not None
        else Path.cwd() / "prepare_panoptic_benchmark_progress.json"
    )
    failedTokensPath = (
        Path(args.failed_tokens_path)
        if args.failed_tokens_path is not None
        else Path.cwd() / "prepare_panoptic_benchmark_failed_tokens.txt"
    )

    summary: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scene_split": args.scene_split,
        "nuscenes_root": str(nuscenesRoot),
        "occ3d_root": str(occ3dRoot),
        "occ_gt_root": str(occGtRoot),
        "info_paths": [str(path) for path in infoPaths],
        "duplicate_info_tokens": duplicateInfoTokens,
        "input_label_name": args.input_label_name,
        "output_label_name": args.output_label_name,
        "num_workers": args.num_workers,
        "chunksize": args.chunksize,
        "split_stats": splitStats,
        "task_stats": taskStats,
        "duplicate_label_tokens": duplicateLabelTokens,
        "total_selected_infos": totalSelected,
        "resolved_before_run": initialProgress,
        "processed_now": 0,
        "success_now": 0,
        "failed_now": 0,
        "non_contiguous_ids_now": 0,
        "sum_unassigned_non_free_now": 0,
        "elapsed_seconds": 0.0,
        "status": "running",
    }
    writeProgressSummary(progressPath, summary)
    logging.info("Progress summary: %s", progressPath)

    if totalSelected == 0:
        summary["status"] = "done"
        summary["elapsed_seconds"] = round(time.time() - startTime, 2)
        writeProgressSummary(progressPath, summary)
        logging.warning("No infos selected. Nothing to do.")
        return

    failedResults: list[dict[str, Any]] = []

    progressBar = tqdm.tqdm(
        total=totalSelected,
        initial=initialProgress,
        desc=f"Preparing {args.output_label_name}",
    )
    try:
        if len(tasks) > 0:
            workerInitArgs = (
                [int(value) for value in args.occ_size],
                [float(value) for value in args.point_cloud_range],
                int(args.num_classes),
                float(args.box_margin),
                args.instance_dtype,
            )

            if args.num_workers == 1:
                initWorker(*workerInitArgs)
                resultIterator = map(processSample, tasks)
                for result in resultIterator:
                    summary["processed_now"] += 1
                    if result["status"] == "ok":
                        summary["success_now"] += 1
                        summary["sum_unassigned_non_free_now"] += int(result["unassigned_non_free"])
                        if not result["contiguous_ids"]:
                            summary["non_contiguous_ids_now"] += 1
                    else:
                        summary["failed_now"] += 1
                        failedResults.append(result)
                    progressBar.update(1)
                    if summary["processed_now"] % args.progress_update_every == 0:
                        summary["elapsed_seconds"] = round(time.time() - startTime, 2)
                        writeProgressSummary(progressPath, summary)
            else:
                with multiprocessing.Pool(
                    processes=args.num_workers,
                    initializer=initWorker,
                    initargs=workerInitArgs,
                ) as pool:
                    for result in pool.imap_unordered(processSample, tasks, chunksize=args.chunksize):
                        summary["processed_now"] += 1
                        if result["status"] == "ok":
                            summary["success_now"] += 1
                            summary["sum_unassigned_non_free_now"] += int(result["unassigned_non_free"])
                            if not result["contiguous_ids"]:
                                summary["non_contiguous_ids_now"] += 1
                        else:
                            summary["failed_now"] += 1
                            failedResults.append(result)
                        progressBar.update(1)
                        if summary["processed_now"] % args.progress_update_every == 0:
                            summary["elapsed_seconds"] = round(time.time() - startTime, 2)
                            writeProgressSummary(progressPath, summary)
    finally:
        progressBar.close()

    summary["status"] = "done"
    summary["elapsed_seconds"] = round(time.time() - startTime, 2)
    summary["failed_tokens"] = [result["token"] for result in failedResults]
    summary["completed_total"] = initialProgress + summary["processed_now"]
    writeProgressSummary(progressPath, summary)

    if len(failedResults) > 0:
        failedTokensPath.parent.mkdir(parents=True, exist_ok=True)
        with open(failedTokensPath, "w") as fileHandle:
            for failedResult in failedResults:
                token = failedResult.get("token")
                sceneName = failedResult.get("scene_name")
                message = failedResult.get("message")
                fileHandle.write(f"{sceneName}\t{token}\t{message}\n")
        logging.warning("Failed tokens saved to %s", failedTokensPath)

    logging.info("Done in %.2fs", summary["elapsed_seconds"])
    logging.info(
        "Selected infos=%d, already_done=%d, missing_input_labels=%d, processed_now=%d, success_now=%d, failed_now=%d",
        totalSelected,
        taskStats["already_done"],
        taskStats["missing_input_labels"],
        summary["processed_now"],
        summary["success_now"],
        summary["failed_now"],
    )
    if summary["non_contiguous_ids_now"] > 0:
        logging.warning(
            "Non-contiguous instance-id samples: %d",
            summary["non_contiguous_ids_now"],
        )
    if summary["sum_unassigned_non_free_now"] > 0:
        logging.warning(
            "Total non-free voxels left unassigned across new outputs: %d",
            summary["sum_unassigned_non_free_now"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate panoptic occupancy labels for NuScenes Occ3D benchmark."
    )
    parser.add_argument("--nuscenes_root", type=str, default="data/nuscenes")
    parser.add_argument("--occ3d_root", type=str, default="data/occ3d_nuscenes")
    parser.add_argument("--info_path", type=str, default=None, help="Optional explicit infos pkl path.")
    parser.add_argument("--scene_split", type=str, choices=["train", "val", "all"], default="val")
    parser.add_argument("--input_label_name", type=str, default="labels.npz")
    parser.add_argument("--output_label_name", type=str, default="labels_pano.npz")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=max(1, multiprocessing.cpu_count() - 1))
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument("--progress_update_every", type=int, default=50)
    parser.add_argument("--progress_path", type=str, default=None)
    parser.add_argument("--failed_tokens_path", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--box_margin", type=float, default=0.8)
    parser.add_argument("--occ_size", nargs=3, type=int, default=[200, 200, 16])
    parser.add_argument("--point_cloud_range", nargs=6, type=float, default=[-40, -40, -1.0, 40, 40, 5.4])
    parser.add_argument("--num_classes", type=int, default=18)
    parser.add_argument("--instance_dtype", type=str, default="uint16", choices=["uint8", "uint16", "uint32"])
    parser.add_argument("--input_path", type=str, default=None, help="Alias for --nuscenes_root.")
    parser.add_argument("--output_path", type=str, default=None, help="Alias for --occ3d_root.")
    parsedArgs = parser.parse_args()

    main(parsedArgs)
