#!/bin/bash
python - <<'PY'
import paths
names = [
    'SCANS_PATH','SCANNET_PROC','CKPT_MASK3D','CKPT_UNI3D','CKPT_CLIP_EVA02',
    'TIMM_EVA_GIANT','CKPT_SAM','CKPT_CLIP336','CKPT_VICUNA','MASKS_DIR',
    'FEATS3D_DIR','FEATS2D_DIR','ANNO_ROOT','OUTPUT_ROOT'
]
for n in names:
    print(f"export {n}={getattr(paths, n)}")
PY


