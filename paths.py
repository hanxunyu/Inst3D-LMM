from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parent

WEIGHTS_ROOT = ROOT / "weights"
DATASETS_ROOT = ROOT / "datasets"
OUTPUT_ROOT = ROOT / "outputs"

CKPT_MASK3D = WEIGHTS_ROOT / "mask3d" / "scannet200_val.ckpt"
CKPT_UNI3D = WEIGHTS_ROOT / "uni3d" / "uni3d_g.pth"
CKPT_CLIP_EVA02 = WEIGHTS_ROOT / "clip" / "EVA02-E-14-plus" / "open_clip_pytorch_model.bin"
TIMM_EVA_GIANT = WEIGHTS_ROOT / "timm" / "eva_giant_patch14_560" / "model.bin"
CKPT_SAM = WEIGHTS_ROOT / "sam" / "sam_vit_h_4b8939.pth"
CKPT_CLIP336 = WEIGHTS_ROOT / "clip" / "clip-vit-large-patch14-336" / "pytorch_model.bin"
CKPT_VICUNA = WEIGHTS_ROOT / "llm" / "vicuna-7b-v1.5"

SCANS_PATH = DATASETS_ROOT / "scannetv2_frames"
SCANNET_PROC = DATASETS_ROOT / "scannet200_processed"
MASKS_DIR = OUTPUT_ROOT / "3d_masks"
FEATS3D_DIR = OUTPUT_ROOT / "3d_feats"
FEATS2D_DIR = OUTPUT_ROOT / "2d_feats"
ANNO_ROOT = DATASETS_ROOT / "annotations"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mk", action="store_true", help="create folder structure")
    args = parser.parse_args()

    if args.mk:
        dirs = [
            WEIGHTS_ROOT,
            DATASETS_ROOT,
            OUTPUT_ROOT,
            MASKS_DIR,
            FEATS3D_DIR,
            FEATS2D_DIR,
            ANNO_ROOT,
            SCANS_PATH,
            SCANNET_PROC,
            CKPT_MASK3D.parent,
            CKPT_UNI3D.parent,
            CKPT_CLIP_EVA02.parent,
            TIMM_EVA_GIANT.parent,
            CKPT_SAM.parent,
            CKPT_CLIP336.parent,
            CKPT_VICUNA,
        ]
        for p in dirs:
            Path(p).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
