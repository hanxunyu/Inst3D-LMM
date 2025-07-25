import paths
from pathlib import Path

ALL_PATHS = [
    paths.WEIGHTS_ROOT,
    paths.DATASETS_ROOT,
    paths.OUTPUT_ROOT,
    paths.CKPT_MASK3D,
    paths.CKPT_UNI3D,
    paths.CKPT_CLIP_EVA02,
    paths.CKPT_SAM,
    paths.CKPT_VICUNA,
    paths.SCANS_PATH,
    paths.SCANNET_PROC,
    paths.MASKS_DIR,
    paths.FEATS3D_DIR,
    paths.FEATS2D_DIR,
    paths.ANNO_ROOT,
]

def test_paths_exist_call():
    for p in ALL_PATHS:
        Path(p).exists()
