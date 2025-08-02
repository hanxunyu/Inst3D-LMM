from __future__ import annotations
from pathlib import Path
import argparse

root = Path(__file__).resolve().parent

WEIGHTS_ROOT = root / 'weights'
DATASETS_ROOT = root / 'datasets'
OUTPUT_ROOT = root / 'outputs'
ANNO_ROOT = root / 'annotations'

CKPT_MASK3D = WEIGHTS_ROOT / 'mask3d' / 'scannet200_val.ckpt'
CKPT_UNI3D = WEIGHTS_ROOT / 'uni3d' / 'uni3d_g.pth'
CKPT_CLIP_EVA02 = WEIGHTS_ROOT / 'open_clip' / 'EVA02-E-14-plus' / 'open_clip_pytorch_model.bin'
TIMM_EVA_GIANT = WEIGHTS_ROOT / 'timm' / 'eva_giant_patch14_560.m30m_ft_in22k_in1k' / 'model.safetensors'
CKPT_SAM = WEIGHTS_ROOT / 'sam' / 'sam_vit_h_4b8939.pth'
CKPT_CLIP336 = WEIGHTS_ROOT / 'clip' / 'clip-vit-large-patch14-336'
CKPT_VICUNA = WEIGHTS_ROOT / 'llm' / 'vicuna-7b-v1.5'

SCANS_PATH = DATASETS_ROOT / 'scannetv2_frames'
SCANNET_PROC = DATASETS_ROOT / 'scannet200_processed'
MASKS_DIR = OUTPUT_ROOT / '3d_masks'
FEATS3D_DIR = OUTPUT_ROOT / '3d_feats'
FEATS2D_DIR = OUTPUT_ROOT / '2d_feats'


def _mk_dirs() -> None:
    dirs = [
        WEIGHTS_ROOT,
        DATASETS_ROOT,
        OUTPUT_ROOT,
        ANNO_ROOT,
        MASKS_DIR,
        FEATS3D_DIR,
        FEATS2D_DIR,
        SCANS_PATH,
        SCANNET_PROC,
        CKPT_MASK3D.parent,
        CKPT_UNI3D.parent,
        CKPT_CLIP_EVA02.parent,
        TIMM_EVA_GIANT.parent,
        CKPT_SAM.parent,
        CKPT_CLIP336,
        CKPT_VICUNA,
    ]
    for p in dirs:
        Path(p).mkdir(parents=True, exist_ok=True)


def _print() -> None:
    for name in [
        'SCANS_PATH','SCANNET_PROC','CKPT_MASK3D','CKPT_UNI3D','CKPT_CLIP_EVA02',
        'TIMM_EVA_GIANT','CKPT_SAM','CKPT_CLIP336','CKPT_VICUNA','MASKS_DIR',
        'FEATS3D_DIR','FEATS2D_DIR','ANNO_ROOT','OUTPUT_ROOT'
    ]:
        value = globals()[name]
        print(f"{name}={value.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mk', action='store_true', help='create folder structure')
    parser.add_argument('--print', dest='dump', action='store_true', help='print all paths')
    args = parser.parse_args()
    if args.mk:
        _mk_dirs()
    if args.dump:
        _print()

if __name__ == '__main__':
    main()

