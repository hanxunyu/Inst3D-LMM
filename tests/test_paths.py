import os, sys, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import paths
from pathlib import Path

ALL_PATHS = [
    paths.WEIGHTS_ROOT,
    paths.DATASETS_ROOT,
    paths.OUTPUT_ROOT,
    paths.CKPT_MASK3D,
    paths.CKPT_UNI3D,
    paths.CKPT_CLIP_EVA02,
    paths.TIMM_EVA_GIANT,
    paths.CKPT_SAM,
    paths.CKPT_CLIP336,
    paths.CKPT_VICUNA,
    paths.SCANS_PATH,
    paths.SCANNET_PROC,
    paths.MASKS_DIR,
    paths.FEATS3D_DIR,
    paths.FEATS2D_DIR,
    paths.ANNO_ROOT,
]

def test_paths_exist_call():
    subprocess.run([sys.executable, "-m", "paths", "--mk"], check=True)
    for p in ALL_PATHS:
        assert Path(p).exists()
