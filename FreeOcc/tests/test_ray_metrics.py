
import argparse
import ast
import json
import logging
import math
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from FreeOcc.eval.ray_ego_pose_dataset import EgoPoseDataset
from FreeOcc.eval.ray_eval_data import (
    build_scene_infos,
    load_nuscenes_info_by_token,
)
from FreeOcc.tools.eval_ray_metrics import buildTokenMaps, resolveOccGtRoot, resolvePredictionSource

logging.basicConfig(format='[%(module)s | l.%(lineno)d] %(message)s')
logging.getLogger().setLevel(logging.INFO)

occClassNamesOcc3d = [
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

freeClassIdOcc3d = len(occClassNamesOcc3d) - 1
pointCloudRange = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
voxelSize = 0.4
gridSize = (200, 200, 16)


def readText(path):
    with open(path, "r") as fileHandle:
        return fileHandle.read()


def findFunctionNode(sourceText, functionName):
    tree = ast.parse(sourceText)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == functionName:
            return node
    raise ValueError(f"Function {functionName} not found")


def findClassNode(sourceText, className):
    tree = ast.parse(sourceText)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == className:
            return node
    raise ValueError(f"Class {className} not found")


def getFunctionSource(path, functionName):
    sourceText = readText(path)
    node = findFunctionNode(sourceText, functionName)
    source = ast.get_source_segment(sourceText, node)
    if source is None:
        raise ValueError(f"Could not extract source for function {functionName} in {path}")
    return source


def getClassSource(path, className):
    sourceText = readText(path)
    node = findClassNode(sourceText, className)
    source = ast.get_source_segment(sourceText, node)
    if source is None:
        raise ValueError(f"Could not extract source for class {className} in {path}")
    return source


def loadFunctionFromSource(path, functionName, namespace):
    functionSource = getFunctionSource(path, functionName)
    localNamespace = dict(namespace)
    exec(functionSource, localNamespace)
    return localNamespace[functionName]


def loadClassFromSource(path, className, namespace):
    classSource = getClassSource(path, className)
    localNamespace = dict(namespace)
    exec(classSource, localNamespace)
    return localNamespace[className]


def assertDistinctSourcePaths(sparseRayMetricsPath, gaussianRayMetricsPath, roadRayMetricsPath, sparseRayPqPath, roadRayPqPath):
    sparseRayMetricsResolved = Path(sparseRayMetricsPath).resolve()
    gaussianRayMetricsResolved = Path(gaussianRayMetricsPath).resolve()
    roadRayMetricsResolved = Path(roadRayMetricsPath).resolve()
    sparseRayPqResolved = Path(sparseRayPqPath).resolve()
    roadRayPqResolved = Path(roadRayPqPath).resolve()

    if sparseRayMetricsResolved == roadRayMetricsResolved:
        raise AssertionError(
            f"SparseOcc ray_metrics path equals FreeOcc path: {sparseRayMetricsResolved}"
        )
    if gaussianRayMetricsResolved == roadRayMetricsResolved:
        raise AssertionError(
            f"GaussianFlowOcc ray_metrics path equals FreeOcc path: {gaussianRayMetricsResolved}"
        )
    if sparseRayPqResolved == roadRayPqResolved:
        raise AssertionError(
            f"SparseOcc ray_pq path equals FreeOcc path: {sparseRayPqResolved}"
        )


def resolveRepoPath(explicitPath, repoName, repoRoot):
    if explicitPath is not None:
        path = Path(explicitPath).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{repoName} path does not exist: {path}")
        return path

    candidates = [
        repoRoot / repoName,
        repoRoot.parent / repoName,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def resolveSparseRayMetricsPath(sparseRoot):
    candidates = [
        sparseRoot / "loaders" / "ray_metrics.py",
        sparseRoot / "ray_metrics.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find SparseOcc ray_metrics.py under {sparseRoot}")


def resolveSparseRayPqPath(sparseRoot):
    candidates = [
        sparseRoot / "loaders" / "ray_pq.py",
        sparseRoot / "ray_pq.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find SparseOcc ray_pq.py under {sparseRoot}")


def resolveGaussianRayMetricsPath(gaussianRoot):
    candidates = [
        gaussianRoot / "mmdet3d" / "datasets" / "ray_miou_metric" / "ray_metrics.py",
        gaussianRoot / "ray_metrics.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find GaussianFlowOcc ray_metrics.py under {gaussianRoot}")


def placeBox(semGrid, instGrid, classId, instId, x0, x1, y0, y1, z0, z1):
    semGrid[x0:x1, y0:y1, z0:z1] = classId
    instGrid[x0:x1, y0:y1, z0:z1] = instId


def clearBox(semGrid, instGrid, x0, x1, y0, y1, z0, z1):
    semGrid[x0:x1, y0:y1, z0:z1] = freeClassIdOcc3d
    instGrid[x0:x1, y0:y1, z0:z1] = 0


def buildSyntheticGrids():
    semGt = np.full(gridSize, freeClassIdOcc3d, dtype=np.uint8)
    instGt = np.zeros(gridSize, dtype=np.int32)

    placeBox(semGt, instGt, classId=11, instId=5, x0=95, x1=170, y0=74, y1=126, z0=0, z1=1)
    placeBox(semGt, instGt, classId=15, instId=6, x0=175, x1=179, y0=70, y1=130, z0=1, z1=10)
    placeBox(semGt, instGt, classId=16, instId=7, x0=145, x1=170, y0=130, y1=150, z0=1, z1=7)

    placeBox(semGt, instGt, classId=4, instId=1, x0=118, x1=124, y0=97, y1=103, z0=2, z1=6)
    placeBox(semGt, instGt, classId=4, instId=2, x0=132, x1=138, y0=106, y1=112, z0=2, z1=6)
    placeBox(semGt, instGt, classId=10, instId=3, x0=148, x1=156, y0=88, y1=95, z0=2, z1=7)
    placeBox(semGt, instGt, classId=7, instId=4, x0=124, x1=126, y0=90, y1=92, z0=2, z1=6)

    semPred = semGt.copy()
    instPred = np.zeros_like(instGt)

    instPred[semPred == 11] = 1011
    instPred[semPred == 15] = 1015
    instPred[semPred == 16] = 1016

    clearBox(semPred, instPred, x0=118, x1=124, y0=97, y1=103, z0=2, z1=6)
    placeBox(semPred, instPred, classId=4, instId=21, x0=119, x1=125, y0=98, y1=104, z0=2, z1=6)

    clearBox(semPred, instPred, x0=132, x1=138, y0=106, y1=112, z0=2, z1=6)
    placeBox(semPred, instPred, classId=4, instId=22, x0=132, x1=135, y0=106, y1=112, z0=2, z1=6)
    placeBox(semPred, instPred, classId=4, instId=23, x0=135, x1=138, y0=106, y1=112, z0=2, z1=6)

    clearBox(semPred, instPred, x0=148, x1=156, y0=88, y1=95, z0=2, z1=7)
    placeBox(semPred, instPred, classId=4, instId=24, x0=149, x1=157, y0=89, y1=96, z0=2, z1=7)

    clearBox(semPred, instPred, x0=124, x1=126, y0=90, y1=92, z0=2, z1=6)
    placeBox(semPred, instPred, classId=7, instId=25, x0=125, x1=127, y0=91, y1=93, z0=2, z1=6)

    placeBox(semPred, instPred, classId=4, instId=26, x0=160, x1=165, y0=120, y1=125, z0=2, z1=6)

    return semGt, instGt, semPred, instPred


def buildSyntheticLidarOrigins():
    return np.array(
        [
            [-8.0, -1.2, 1.5],
            [-6.0, -0.9, 1.5],
            [-4.0, -0.6, 1.5],
            [-2.0, -0.3, 1.5],
            [0.0, 0.0, 1.5],
            [2.0, 0.3, 1.5],
            [4.0, 0.6, 1.5],
            [6.0, 0.9, 1.5],
        ],
        dtype=np.float32,
    )


def loadArrayFromGridPath(path, preferredKey=None):
    pathObj = Path(path).expanduser().resolve()
    if not pathObj.exists():
        raise FileNotFoundError(f"Grid file not found: {pathObj}")

    suffix = pathObj.suffix.lower()
    if suffix == ".npy":
        return np.load(pathObj)
    if suffix != ".npz":
        raise ValueError(f"Unsupported grid extension for {pathObj}. Use .npy or .npz")

    with np.load(pathObj, allow_pickle=True) as data:
        if preferredKey is not None and preferredKey in data.files:
            return data[preferredKey]
        if len(data.files) == 1:
            return data[data.files[0]]
        if "arr_0" in data.files:
            return data["arr_0"]
        raise KeyError(f"Could not resolve key in {pathObj}. keys={list(data.files)}")


def loadGridsFromArgs(args):
    hasExternalGridArgs = any(
        path is not None
        for path in [args.pred_semantic_grid_path, args.pred_instance_grid_path, args.gt_label_pano_path]
    )
    if not hasExternalGridArgs:
        semGt, instGt, semPred, instPred = buildSyntheticGrids()
        return semGt, instGt, semPred, instPred, "synthetic"

    required = {
        "pred_semantic_grid_path": args.pred_semantic_grid_path,
        "pred_instance_grid_path": args.pred_instance_grid_path,
        "gt_label_pano_path": args.gt_label_pano_path,
    }
    missing = [name for name, value in required.items() if value is None]
    if len(missing) > 0:
        raise ValueError(
            f"External-grid mode requires all of {list(required.keys())}. Missing: {missing}"
        )

    semPred = loadArrayFromGridPath(args.pred_semantic_grid_path, preferredKey=args.pred_semantic_key)
    instPred = loadArrayFromGridPath(args.pred_instance_grid_path, preferredKey=args.pred_instance_key)
    gtLabelPath = Path(args.gt_label_pano_path).expanduser().resolve()
    if not gtLabelPath.exists():
        raise FileNotFoundError(f"GT label file not found: {gtLabelPath}")
    if gtLabelPath.suffix.lower() != ".npz":
        raise ValueError(f"--gt_label_pano_path must be a .npz file, got {gtLabelPath}")
    with np.load(gtLabelPath, allow_pickle=True) as gtData:
        if "semantics" not in gtData.files:
            raise KeyError(f"'semantics' key not found in {gtLabelPath}")
        if "instances" not in gtData.files:
            raise KeyError(f"'instances' key not found in {gtLabelPath}")
        semGt = gtData["semantics"]
        instGt = gtData["instances"]

    semGt = np.asarray(semGt).astype(np.uint8)
    semPred = np.asarray(semPred).astype(np.uint8)
    instGt = np.asarray(instGt).astype(np.int32)
    instPred = np.asarray(instPred).astype(np.int32)

    if semGt.shape != gridSize:
        raise ValueError(f"GT semantic shape must be {gridSize}, got {semGt.shape}")
    if semPred.shape != gridSize:
        raise ValueError(f"Pred semantic shape must be {gridSize}, got {semPred.shape}")
    if instGt.shape != gridSize:
        raise ValueError(f"GT instance shape must be {gridSize}, got {instGt.shape}")
    if instPred.shape != gridSize:
        raise ValueError(f"Pred instance shape must be {gridSize}, got {instPred.shape}")

    return semGt, instGt, semPred, instPred, "external"


def loadLidarOriginsFromArgs(args):
    if args.lidar_origins_path is None:
        return buildSyntheticLidarOrigins(), "synthetic"

    lidarOrigins = loadArrayFromGridPath(args.lidar_origins_path, preferredKey=args.lidar_origins_key)
    lidarOrigins = np.asarray(lidarOrigins, dtype=np.float32)
    if lidarOrigins.ndim == 3 and lidarOrigins.shape[0] == 1 and lidarOrigins.shape[2] == 3:
        lidarOrigins = lidarOrigins[0]
    if lidarOrigins.ndim != 2 or lidarOrigins.shape[1] != 3:
        raise ValueError(
            f"Lidar origins must have shape (N,3) or (1,N,3), got {lidarOrigins.shape}"
        )
    if lidarOrigins.shape[0] == 0:
        raise ValueError("Lidar origins array is empty.")
    return lidarOrigins, "external"


def buildSamplesFromExpPath(args):
    if args.exp_path is None:
        raise ValueError("buildSamplesFromExpPath requires --exp_path.")
    if args.lidar_origins_path is not None:
        raise ValueError("--lidar_origins_path is not supported with --exp_path mode.")

    predSource, semPathByToken, instPathByToken = resolvePredictionSource(
        predDir=None,
        expPath=args.exp_path,
        expSemFilename=args.exp_sem_filename,
        expInstFilename=args.exp_inst_filename,
    )
    logging.info("Prediction source=%s tokens=%d", predSource, len(semPathByToken))

    occGtRootSem = resolveOccGtRoot(Path(args.occ3d_root), "labels.npz")
    tokenToSemGtPath, _ = buildTokenMaps(occGtRootSem, "labels.npz")
    occGtRootInst = resolveOccGtRoot(Path(args.occ3d_root), args.gt_label_name)
    tokenToInstGtPath, _ = buildTokenMaps(occGtRootInst, args.gt_label_name)

    evalTokens = sorted(
        set(tokenToSemGtPath.keys())
        & set(tokenToInstGtPath.keys())
        & set(semPathByToken.keys())
        & set(instPathByToken.keys())
    )
    if args.max_tokens is not None:
        evalTokens = evalTokens[: int(args.max_tokens)]
    if len(evalTokens) == 0:
        raise RuntimeError("No overlapping tokens between exp predictions and GT labels.")

    sceneInfosEval, missingTokens = build_scene_infos(
        sample_tokens=evalTokens,
        scene_name="__ray_eval__",
        nuscenes_root=Path(args.nuscenes_root),
        nuscenes_split=args.nuscenes_split,
    )
    if len(missingTokens) > 0:
        logging.warning("Missing pose infos for %d tokens", len(missingTokens))

    infoByTokenEval = {info["token"]: info for info in sceneInfosEval}
    evalTokens = [token for token in evalTokens if token in infoByTokenEval]
    if len(evalTokens) == 0:
        raise RuntimeError("No valid tokens with both prediction, GT, and pose info.")

    nuScenesInfoByToken = load_nuscenes_info_by_token(args.nuscenes_root)
    evalSceneNames = {
        infoByTokenEval[token].get("scene_name")
        for token in evalTokens
        if infoByTokenEval[token].get("scene_name") not in [None, ""]
    }
    sceneInfosContext = [
        dict(info)
        for info in nuScenesInfoByToken.values()
        if info.get("scene_name") in evalSceneNames
    ]
    if len(sceneInfosContext) == 0:
        sceneInfosContext = [infoByTokenEval[token] for token in evalTokens]
        logging.warning(
            "Could not expand to full-scene context; falling back to eval-token-only context."
        )

    samples = []
    evalTokenSet = set(evalTokens)
    dataLoader = DataLoader(EgoPoseDataset(sceneInfosContext), num_workers=args.num_workers)
    for batch in dataLoader:
        token = batch[0][0]
        if token not in evalTokenSet:
            continue

        lidarOrigins = batch[1]
        if hasattr(lidarOrigins, "cpu"):
            lidarOrigins = lidarOrigins.cpu().numpy()
        if lidarOrigins.ndim == 3 and lidarOrigins.shape[0] == 1:
            lidarOrigins = lidarOrigins[0]
        lidarOrigins = np.asarray(lidarOrigins, dtype=np.float32)

        gtSemPath = tokenToSemGtPath[token]
        with np.load(gtSemPath, allow_pickle=True) as gtSemData:
            semGt = np.asarray(gtSemData["semantics"]).astype(np.uint8)

        gtInstPath = tokenToInstGtPath[token]
        with np.load(gtInstPath, allow_pickle=True) as gtInstData:
            semGtPano = np.asarray(gtInstData["semantics"]).astype(np.uint8)
            instGt = np.asarray(gtInstData["instances"]).astype(np.int32)
        if not np.array_equal(semGt, semGtPano):
            logging.warning("GT semantics mismatch between labels.npz and %s for token=%s", args.gt_label_name, token)

        semPred = np.asarray(
            loadArrayFromGridPath(semPathByToken[token], preferredKey=args.pred_sem_key)
        ).astype(np.uint8)
        instPred = np.asarray(
            loadArrayFromGridPath(instPathByToken[token], preferredKey=args.pred_inst_key)
        ).astype(np.int32)

        if semGt.shape != gridSize or instGt.shape != gridSize or semPred.shape != gridSize or instPred.shape != gridSize:
            logging.warning(
                "Skipping token=%s due to shape mismatch gt_sem=%s gt_inst=%s pred_sem=%s pred_inst=%s",
                token,
                semGt.shape,
                instGt.shape,
                semPred.shape,
                instPred.shape,
            )
            continue
        if lidarOrigins.ndim != 2 or lidarOrigins.shape[1] != 3 or lidarOrigins.shape[0] == 0:
            logging.warning("Skipping token=%s due to invalid lidar origins shape=%s", token, lidarOrigins.shape)
            continue

        samples.append(
            {
                "token": token,
                "sem_gt": semGt,
                "inst_gt": instGt,
                "sem_pred": semPred,
                "inst_pred": instPred,
                "lidar_origins": lidarOrigins,
            }
        )

    if len(samples) == 0:
        raise RuntimeError("No valid exp samples could be loaded.")

    modeStats = {
        "num_eval_tokens_requested": int(len(evalTokens)),
        "num_samples_loaded": int(len(samples)),
        "num_scene_context_frames": int(len(sceneInfosContext)),
    }
    return samples, modeStats


def renderPcdFromGrid(semGrid, instGrid, lidarOrigins, lidarRays, stepSize, maxDepth):
    if semGrid.shape != gridSize:
        raise ValueError(f"Unexpected semantic grid shape {semGrid.shape}, expected {gridSize}")
    if instGrid is not None and instGrid.shape != gridSize:
        raise ValueError(f"Unexpected instance grid shape {instGrid.shape}, expected {gridSize}")

    depths = np.arange(0.0, maxDepth + 1e-6, stepSize, dtype=np.float32)
    allRows = []
    numRays = lidarRays.shape[0]

    for origin in lidarOrigins:
        labels = np.full(numRays, freeClassIdOcc3d, dtype=np.int32)
        hitDepths = np.full(numRays, maxDepth, dtype=np.float32)
        if instGrid is not None:
            instances = np.zeros(numRays, dtype=np.int32)
        else:
            instances = None
        alive = np.ones(numRays, dtype=bool)

        for depth in depths:
            if not np.any(alive):
                break
            aliveRayIdx = np.where(alive)[0]
            rayPoints = origin[None, :] + lidarRays[aliveRayIdx] * depth

            xIdx = np.floor((rayPoints[:, 0] - pointCloudRange[0]) / voxelSize).astype(np.int32)
            yIdx = np.floor((rayPoints[:, 1] - pointCloudRange[1]) / voxelSize).astype(np.int32)
            zIdx = np.floor((rayPoints[:, 2] - pointCloudRange[2]) / voxelSize).astype(np.int32)

            inBounds = (
                (xIdx >= 0)
                & (xIdx < gridSize[0])
                & (yIdx >= 0)
                & (yIdx < gridSize[1])
                & (zIdx >= 0)
                & (zIdx < gridSize[2])
            )
            if not np.any(inBounds):
                continue

            boundedRayIdx = aliveRayIdx[inBounds]
            xBound = xIdx[inBounds]
            yBound = yIdx[inBounds]
            zBound = zIdx[inBounds]
            semVals = semGrid[xBound, yBound, zBound]
            hitMask = semVals != freeClassIdOcc3d
            if not np.any(hitMask):
                continue

            hitRayIdx = boundedRayIdx[hitMask]
            labels[hitRayIdx] = semVals[hitMask]
            hitDepths[hitRayIdx] = depth
            if instances is not None:
                instVals = instGrid[xBound, yBound, zBound]
                instances[hitRayIdx] = instVals[hitMask]
            alive[hitRayIdx] = False

        if instances is None:
            rows = np.stack([labels.astype(np.float32), hitDepths], axis=1)
        else:
            rows = np.stack([labels.astype(np.float32), instances.astype(np.float32), hitDepths], axis=1)
        allRows.append(rows)

    return np.concatenate(allRows, axis=0)


def renderPcdFromGridDvr(semGrid, instGrid, lidarOrigins, lidarRays):
    try:
        import torch
        from FreeOcc.eval.ray_metrics import process_one_sample as processOneSampleRoad
    except Exception as exception:
        raise RuntimeError(f"Could not import FreeOcc DVR ray backend: {exception}")

    semTensor = torch.from_numpy(np.reshape(semGrid, [200, 200, 16]))
    instTensor = None
    if instGrid is not None:
        instTensor = torch.from_numpy(np.reshape(instGrid, [200, 200, 16]))

    lidarOrigins = np.asarray(lidarOrigins, dtype=np.float32)
    if lidarOrigins.ndim != 2 or lidarOrigins.shape[1] != 3:
        raise ValueError(f"lidarOrigins must have shape (N,3), got {lidarOrigins.shape}")
    outputOrigin = torch.from_numpy(lidarOrigins)[None, :, :]
    lidarRaysTensor = torch.from_numpy(np.asarray(lidarRays, dtype=np.float32))

    try:
        pcd = processOneSampleRoad(
            semTensor,
            lidarRaysTensor,
            outputOrigin,
            instance_pred=instTensor,
            occ_class_names=occClassNamesOcc3d,
        )
    except Exception as exception:
        raise RuntimeError(
            "DVR backend failed. This backend requires a working CUDA/PyTorch extension setup "
            "(same as production ray eval). In multi-user environments, you may need a CUDA-enabled "
            "runtime and a valid TORCH_CUDA_ARCH_LIST."
        ) from exception
    return pcd


def summarizeRayiouFromIouList(iouList):
    rayiouAt1 = float(np.nanmean(iouList[0]))
    rayiouAt2 = float(np.nanmean(iouList[1]))
    rayiouAt4 = float(np.nanmean(iouList[2]))
    return {
        "RayIoU": float(np.nanmean(iouList)),
        "RayIoU@1": rayiouAt1,
        "RayIoU@2": rayiouAt2,
        "RayIoU@4": rayiouAt4,
    }


def assertCloseArrays(arrayA, arrayB, label, atol=1e-12):
    if not np.allclose(arrayA, arrayB, atol=atol, rtol=0.0, equal_nan=True):
        maxDiff = float(np.nanmax(np.abs(arrayA - arrayB)))
        raise AssertionError(f"{label} mismatch: max_diff={maxDiff}")


def main(args):
    logging.info("args = %s", args)

    repoRoot = Path(__file__).resolve().parents[2]
    outputDir = Path(args.output_dir).expanduser().resolve()
    outputDir.mkdir(parents=True, exist_ok=True)

    sparseRoot = resolveRepoPath(args.sparseocc_root, "SparseOcc", repoRoot)
    gaussianRoot = resolveRepoPath(args.gaussianflowocc_root, "GaussianFlowOcc", repoRoot)
    if sparseRoot is None:
        raise FileNotFoundError("SparseOcc repo not found. Provide --sparseocc_root.")
    if gaussianRoot is None:
        raise FileNotFoundError("GaussianFlowOcc repo not found. Provide --gaussianflowocc_root.")

    sparseRayMetricsPath = resolveSparseRayMetricsPath(sparseRoot)
    sparseRayPqPath = resolveSparseRayPqPath(sparseRoot)
    gaussianRayMetricsPath = resolveGaussianRayMetricsPath(gaussianRoot)
    roadRayMetricsPath = repoRoot / "FreeOcc" / "eval" / "ray_metrics.py"
    roadRayPqPath = repoRoot / "FreeOcc" / "eval" / "ray_pq.py"

    logging.info("Using SparseOcc ray_metrics: %s", sparseRayMetricsPath)
    logging.info("Using SparseOcc ray_pq: %s", sparseRayPqPath)
    logging.info("Using GaussianFlowOcc ray_metrics: %s", gaussianRayMetricsPath)
    logging.info("Using FreeOcc ray_metrics: %s", roadRayMetricsPath)
    logging.info("Using FreeOcc ray_pq: %s", roadRayPqPath)
    assertDistinctSourcePaths(
        sparseRayMetricsPath=sparseRayMetricsPath,
        gaussianRayMetricsPath=gaussianRayMetricsPath,
        roadRayMetricsPath=roadRayMetricsPath,
        sparseRayPqPath=sparseRayPqPath,
        roadRayPqPath=roadRayPqPath,
    )

    if args.exp_path is not None:
        if any(
            value is not None
            for value in [args.pred_semantic_grid_path, args.pred_instance_grid_path, args.gt_label_pano_path]
        ):
            raise ValueError("Use either --exp_path mode OR direct-grid mode, not both.")
        samples, expModeStats = buildSamplesFromExpPath(args)
        gridSourceMode = "exp_path"
        lidarSourceMode = "exp_path"
        logging.info("Loaded %d samples from exp_path=%s", len(samples), args.exp_path)
    else:
        semGt, instGt, semPred, instPred, gridSourceMode = loadGridsFromArgs(args)
        lidarOrigins, lidarSourceMode = loadLidarOriginsFromArgs(args)
        samples = [
            {
                "token": "sample_0000",
                "sem_gt": semGt,
                "inst_gt": instGt,
                "sem_pred": semPred,
                "inst_pred": instPred,
                "lidar_origins": lidarOrigins,
            }
        ]
        expModeStats = {}
    logging.info("Grid source mode: %s", gridSourceMode)
    logging.info("Lidar-origin source mode: %s", lidarSourceMode)
    logging.info("Ray rendering backend: %s", args.backend)

    calcSparseRayiou = loadFunctionFromSource(sparseRayMetricsPath, "calc_rayiou", {"np": np})
    calcGaussianRayiou = loadFunctionFromSource(gaussianRayMetricsPath, "calc_rayiou", {"np": np})
    calcRoadRayiou = loadFunctionFromSource(roadRayMetricsPath, "calc_rayiou", {"np": np})
    generateLidarRaysSparse = loadFunctionFromSource(
        sparseRayMetricsPath,
        "generate_lidar_rays",
        {"np": np, "math": math},
    )

    try:
        from prettytable import PrettyTable
    except Exception:
        class PrettyTable:  # type: ignore
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

    sparseMetricClass = loadClassFromSource(sparseRayPqPath, "Metric_RayPQ", {"np": np, "PrettyTable": PrettyTable})
    roadMetricClass = loadClassFromSource(roadRayPqPath, "Metric_RayPQ", {"np": np, "PrettyTable": PrettyTable})

    allRays = generateLidarRaysSparse()
    lidarRays = allRays[:: max(1, int(args.ray_stride))]
    if args.backend == "dvr":
        if args.ray_step_size != 0.2 or args.ray_max_depth != 80.0:
            logging.info(
                "--ray_step_size and --ray_max_depth are ignored with --backend=dvr."
            )
    logging.info(
        "Lidar rays: original=%d stride=%d selected=%d",
        allRays.shape[0],
        args.ray_stride,
        lidarRays.shape[0],
    )

    pcdPredRayiou = []
    pcdGtRayiou = []
    sparseRaypqMetric = sparseMetricClass(occ_class_names=occClassNamesOcc3d, num_classes=18, thresholds=[1, 2, 4])
    roadRaypqMetric = roadMetricClass(occ_class_names=occClassNamesOcc3d, num_classes=18, thresholds=[1, 2, 4])
    totalRaysRendered = 0
    totalNonFreeGtRays = 0
    sampleOutputs = []

    for sampleIdx, sample in enumerate(samples):
        if args.backend == "cpu":
            pcdPred = renderPcdFromGrid(
                semGrid=sample["sem_pred"],
                instGrid=sample["inst_pred"],
                lidarOrigins=sample["lidar_origins"],
                lidarRays=lidarRays,
                stepSize=args.ray_step_size,
                maxDepth=args.ray_max_depth,
            )
            pcdGt = renderPcdFromGrid(
                semGrid=sample["sem_gt"],
                instGrid=sample["inst_gt"],
                lidarOrigins=sample["lidar_origins"],
                lidarRays=lidarRays,
                stepSize=args.ray_step_size,
                maxDepth=args.ray_max_depth,
            )
        elif args.backend == "dvr":
            pcdPred = renderPcdFromGridDvr(
                semGrid=sample["sem_pred"],
                instGrid=sample["inst_pred"],
                lidarOrigins=sample["lidar_origins"],
                lidarRays=lidarRays,
            )
            pcdGt = renderPcdFromGridDvr(
                semGrid=sample["sem_gt"],
                instGrid=sample["inst_gt"],
                lidarOrigins=sample["lidar_origins"],
                lidarRays=lidarRays,
            )
        else:
            raise ValueError(f"Unknown --backend={args.backend}")
        validMask = pcdGt[:, 0].astype(np.int32) != freeClassIdOcc3d
        pcdPredNonFree = pcdPred[validMask]
        pcdGtNonFree = pcdGt[validMask]
        if pcdGtNonFree.shape[0] == 0:
            logging.warning("Skipping token=%s (index=%d): no non-free GT rays.", sample["token"], sampleIdx)
            continue

        totalRaysRendered += int(pcdGt.shape[0])
        totalNonFreeGtRays += int(pcdGtNonFree.shape[0])
        pcdPredRayiou.append(pcdPredNonFree[:, [0, 2]])
        pcdGtRayiou.append(pcdGtNonFree[:, [0, 2]])

        semPredRay = pcdPredNonFree[:, 0].astype(np.int32)
        semGtRay = pcdGtNonFree[:, 0].astype(np.int32)
        instPredRay = pcdPredNonFree[:, 1].astype(np.int32)
        instGtRay = pcdGtNonFree[:, 1].astype(np.int32)
        l1Error = np.abs(pcdPredNonFree[:, 2] - pcdGtNonFree[:, 2])

        sparseRaypqMetric.add_batch(semPredRay, semGtRay, instPredRay, instGtRay, l1Error)
        roadRaypqMetric.add_batch(semPredRay, semGtRay, instPredRay, instGtRay, l1Error)

        sampleOutputs.append(
            {
                "token": sample["token"],
                "pcd_pred_nonfree": pcdPredNonFree,
                "pcd_gt_nonfree": pcdGtNonFree,
                "sem_pred": sample["sem_pred"],
                "sem_gt": sample["sem_gt"],
                "inst_pred": sample["inst_pred"],
                "inst_gt": sample["inst_gt"],
            }
        )

    if len(pcdPredRayiou) == 0:
        raise RuntimeError("No valid non-free ray samples were rendered.")
    logging.info(
        "Rendered rays across samples: total=%d non_free_gt=%d used_samples=%d",
        totalRaysRendered,
        totalNonFreeGtRays,
        len(pcdPredRayiou),
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        iouSparse = calcSparseRayiou(pcdPredRayiou, pcdGtRayiou, occClassNamesOcc3d)
        iouGaussian = calcGaussianRayiou(pcdPredRayiou, pcdGtRayiou, occClassNamesOcc3d)
        iouRoad = calcRoadRayiou(pcdPredRayiou, pcdGtRayiou, occClassNamesOcc3d)

    for idx in range(len(iouSparse)):
        assertCloseArrays(
            np.asarray(iouSparse[idx]),
            np.asarray(iouGaussian[idx]),
            f"RayIoU SparseOcc vs GaussianFlowOcc @idx={idx}",
            atol=args.atol,
        )
        assertCloseArrays(
            np.asarray(iouSparse[idx]),
            np.asarray(iouRoad[idx]),
            f"RayIoU SparseOcc vs FreeOcc @idx={idx}",
            atol=args.atol,
        )

    rayiouSparse = summarizeRayiouFromIouList(iouSparse)
    rayiouGaussian = summarizeRayiouFromIouList(iouGaussian)
    rayiouRoad = summarizeRayiouFromIouList(iouRoad)

    raypqSparse = sparseRaypqMetric.count_pq()
    raypqRoad = roadRaypqMetric.count_pq(print_table=False)

    for key in ["RayPQ", "RayPQ@1", "RayPQ@2", "RayPQ@4"]:
        if not np.isclose(raypqSparse[key], raypqRoad[key], atol=args.atol, rtol=0.0, equal_nan=True):
            raise AssertionError(
                f"RayPQ mismatch for {key}: SparseOcc={raypqSparse[key]} FreeOcc={raypqRoad[key]}"
            )

    savedSamplesRoot = outputDir / "sample_grids"
    savedSamplesRoot.mkdir(parents=True, exist_ok=True)
    maxSave = max(1, int(args.max_save_samples))
    for sampleData in sampleOutputs[:maxSave]:
        sampleDir = savedSamplesRoot / sampleData["token"]
        sampleDir.mkdir(parents=True, exist_ok=True)
        np.save(sampleDir / "gt_occ_grid_occ3d_nuscenes.npy", sampleData["sem_gt"].astype(np.uint8))
        np.save(sampleDir / "gt_occ_grid_instances.npy", sampleData["inst_gt"].astype(np.int32))
        np.save(sampleDir / "pred_occ_grid_occ3d_nuscenes.npy", sampleData["sem_pred"].astype(np.uint8))
        np.save(sampleDir / "pred_occ_grid_instances.npy", sampleData["inst_pred"].astype(np.int32))
        np.save(sampleDir / "pcd_gt_nonfree.npy", sampleData["pcd_gt_nonfree"])
        np.save(sampleDir / "pcd_pred_nonfree.npy", sampleData["pcd_pred_nonfree"])

    firstSample = sampleOutputs[0]
    np.save(outputDir / "gt_occ_grid_occ3d_nuscenes.npy", firstSample["sem_gt"].astype(np.uint8))
    np.save(outputDir / "gt_occ_grid_instances.npy", firstSample["inst_gt"].astype(np.int32))
    np.save(outputDir / "pred_occ_grid_occ3d_nuscenes.npy", firstSample["sem_pred"].astype(np.uint8))
    np.save(outputDir / "pred_occ_grid_instances.npy", firstSample["inst_pred"].astype(np.int32))
    np.save(outputDir / "pcd_gt_nonfree.npy", firstSample["pcd_gt_nonfree"])
    np.save(outputDir / "pcd_pred_nonfree.npy", firstSample["pcd_pred_nonfree"])

    resultData = {
        "evaluation_mode": {
            "grid_source": gridSourceMode,
            "lidar_origin_source": lidarSourceMode,
            "ray_backend": args.backend,
        },
        "grid_shape": list(gridSize),
        "voxel_size": voxelSize,
        "point_cloud_range": pointCloudRange,
        "num_samples_used": int(len(pcdPredRayiou)),
        "num_lidar_origins_first_sample": int(samples[0]["lidar_origins"].shape[0]),
        "num_lidar_rays_selected": int(lidarRays.shape[0]),
        "num_ray_points_nonfree_total": int(totalNonFreeGtRays),
        "rayiou_sparseocc": rayiouSparse,
        "rayiou_gaussianflowocc": rayiouGaussian,
        "rayiou_roadocc": rayiouRoad,
        "raypq_sparseocc": raypqSparse,
        "raypq_roadocc": raypqRoad,
        "comparison": {
            "rayiou_sparse_equals_gaussian": True,
            "rayiou_sparse_equals_road": True,
            "raypq_sparse_equals_road": True,
        },
        "metric_sources": {
            "sparseocc": {
                "ray_metrics_path": str(Path(sparseRayMetricsPath).resolve()),
                "ray_pq_path": str(Path(sparseRayPqPath).resolve()),
            },
            "gaussianflowocc": {
                "ray_metrics_path": str(Path(gaussianRayMetricsPath).resolve()),
            },
            "roadocc": {
                "ray_metrics_path": str(Path(roadRayMetricsPath).resolve()),
                "ray_pq_path": str(Path(roadRayPqPath).resolve()),
            },
        },
        "files": {
            "gt_semantic": str(outputDir / "gt_occ_grid_occ3d_nuscenes.npy"),
            "gt_instances": str(outputDir / "gt_occ_grid_instances.npy"),
            "pred_semantic": str(outputDir / "pred_occ_grid_occ3d_nuscenes.npy"),
            "pred_instances": str(outputDir / "pred_occ_grid_instances.npy"),
            "pcd_gt_nonfree": str(outputDir / "pcd_gt_nonfree.npy"),
            "pcd_pred_nonfree": str(outputDir / "pcd_pred_nonfree.npy"),
            "saved_samples_root": str(savedSamplesRoot),
        },
    }
    if args.exp_path is not None:
        resultData["exp_mode_stats"] = expModeStats
        resultData["input_paths"] = resultData.get("input_paths", {})
        resultData["input_paths"]["exp_path"] = str(Path(args.exp_path).expanduser().resolve())
        resultData["input_paths"]["occ3d_root"] = str(Path(args.occ3d_root).expanduser().resolve())
        resultData["input_paths"]["nuscenes_root"] = str(Path(args.nuscenes_root).expanduser().resolve())
    if gridSourceMode == "external":
        resultData["input_paths"] = {
            "pred_semantic_grid_path": str(Path(args.pred_semantic_grid_path).expanduser().resolve()),
            "pred_instance_grid_path": str(Path(args.pred_instance_grid_path).expanduser().resolve()),
            "gt_label_pano_path": str(Path(args.gt_label_pano_path).expanduser().resolve()),
        }
    if lidarSourceMode == "external":
        resultData["input_paths"] = resultData.get("input_paths", {})
        resultData["input_paths"]["lidar_origins_path"] = str(
            Path(args.lidar_origins_path).expanduser().resolve()
        )

    if gridSourceMode == "synthetic":
        outputJsonFilename = "synthetic_ray_metrics_results.json"
    elif gridSourceMode == "external":
        outputJsonFilename = "external_ray_metrics_results.json"
    else:
        outputJsonFilename = "exp_ray_metrics_results.json"
    outputJsonPath = outputDir / outputJsonFilename
    with open(outputJsonPath, "w") as fileHandle:
        json.dump(resultData, fileHandle, indent=2)

    logging.info("Saved results to %s", outputJsonPath)
    logging.info(
        "Visualize GT semantic: python -m FreeOcc.visualization.visualize_occ3d_grid --input_file %s",
        outputDir / "gt_occ_grid_occ3d_nuscenes.npy",
    )
    logging.info(
        "Visualize GT instances: python -m FreeOcc.visualization.visualize_occ3d_grid --input_file %s --use_instances",
        outputDir / "gt_occ_grid_instances.npy",
    )
    logging.info(
        "Visualize Pred semantic: python -m FreeOcc.visualization.visualize_occ3d_grid --input_file %s",
        outputDir / "pred_occ_grid_occ3d_nuscenes.npy",
    )
    logging.info(
        "Visualize Pred instances: python -m FreeOcc.visualization.visualize_occ3d_grid --input_file %s --use_instances",
        outputDir / "pred_occ_grid_instances.npy",
    )
    logging.info("RayIoU SparseOcc = %s", rayiouSparse)
    logging.info("RayIoU GaussianFlowOcc = %s", rayiouGaussian)
    logging.info("RayIoU FreeOcc = %s", rayiouRoad)
    logging.info("RayPQ SparseOcc = %s", raypqSparse)
    logging.info("RayPQ FreeOcc = %s", raypqRoad)
    logging.info("Ray-metric test PASSED.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deterministic synthetic RayIoU/RayPQ cross-repo test with saved grids for visualization."
    )
    parser.add_argument("--output_dir", type=str, default="results/tests/ray_metrics_synthetic")
    parser.add_argument("--backend", type=str, default="cpu", choices=["cpu", "dvr"], help="Ray rendering backend. 'dvr' uses the same CUDA backend as eval_ray_metrics.py.")
    parser.add_argument("--ray_stride", type=int, default=8, help="Subsample lidar rays for faster synthetic rendering.")
    parser.add_argument("--ray_step_size", type=float, default=0.2, help="Ray-marching step size in meters.")
    parser.add_argument("--ray_max_depth", type=float, default=80.0, help="Maximum ray depth in meters.")
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--pred_semantic_grid_path", type=str, default=None, help="Optional .npy/.npz path to predicted semantic grid (occ_grid_occ3d_nuscenes.npy).")
    parser.add_argument("--pred_instance_grid_path", type=str, default=None, help="Optional .npy/.npz path to predicted instance grid (occ_grid_instances.npy).")
    parser.add_argument("--gt_label_pano_path", type=str, default=None, help="Optional labels_pano.npz path containing GT semantics+instances.")
    parser.add_argument("--pred_semantic_key", type=str, default="semantics", help="Preferred key when --pred_semantic_grid_path is .npz.")
    parser.add_argument("--pred_instance_key", type=str, default="instances", help="Preferred key when --pred_instance_grid_path is .npz.")
    parser.add_argument("--lidar_origins_path", type=str, default=None, help="Optional .npy/.npz path of lidar origins with shape (N,3) or (1,N,3).")
    parser.add_argument("--lidar_origins_key", type=str, default="origins", help="Preferred key when --lidar_origins_path is .npz.")
    parser.add_argument("--exp_path", type=str, default=None, help="Optional experiment path like results/.../exps/<exp_name> (same layout as eval_ray_metrics.py).")
    parser.add_argument("--occ3d_root", type=str, default="data/occ3d_nuscenes")
    parser.add_argument("--nuscenes_root", type=str, default="data/nuscenes")
    parser.add_argument("--nuscenes_split", type=str, default="val", choices=["train", "val", "trainval", "all", "test"])
    parser.add_argument("--gt_label_name", type=str, default="labels_pano.npz")
    parser.add_argument("--exp_sem_filename", type=str, default="occ_grid_occ3d_nuscenes.npy")
    parser.add_argument("--exp_inst_filename", type=str, default="occ_grid_instances.npy")
    parser.add_argument("--pred_sem_key", type=str, default="pred")
    parser.add_argument("--pred_inst_key", type=str, default="pano_inst")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--max_save_samples", type=int, default=5)
    parser.add_argument("--sparseocc_root", type=str, default=None, help="Optional SparseOcc repo path")
    parser.add_argument("--gaussianflowocc_root", type=str, default=None, help="Optional GaussianFlowOcc repo path")
    args = parser.parse_args()

    main(args)
