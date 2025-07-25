#!/bin/bash

export CKPT_MASK3D=$(python - <<'PY'
import paths
print(paths.CKPT_MASK3D)
PY
)
export CKPT_UNI3D=$(python - <<'PY'
import paths
print(paths.CKPT_UNI3D)
PY
)
export CKPT_CLIP_EVA02=$(python - <<'PY'
import paths
print(paths.CKPT_CLIP_EVA02)
PY
)
export TIMM_EVA_GIANT=$(python - <<'PY'
import paths
print(paths.TIMM_EVA_GIANT)
PY
)
export CKPT_SAM=$(python - <<'PY'
import paths
print(paths.CKPT_SAM)
PY
)
export CKPT_VICUNA=$(python - <<'PY'
import paths
print(paths.CKPT_VICUNA)
PY
)
export SCANS_PATH=$(python - <<'PY'
import paths
print(paths.SCANS_PATH)
PY
)
export SCANNET_PROC=$(python - <<'PY'
import paths
print(paths.SCANNET_PROC)
PY
)
export MASKS_DIR=$(python - <<'PY'
import paths
print(paths.MASKS_DIR)
PY
)
export FEATS3D_DIR=$(python - <<'PY'
import paths
print(paths.FEATS3D_DIR)
PY
)
export FEATS2D_DIR=$(python - <<'PY'
import paths
print(paths.FEATS2D_DIR)
PY
)
export CKPT_CLIP336=$(python - <<'PY'
import paths
print(paths.CKPT_CLIP336)
PY
)
export OUTPUT_ROOT=$(python - <<'PY'
import paths
print(paths.OUTPUT_ROOT)
PY
)

