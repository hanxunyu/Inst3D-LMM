from pathlib import Path

ROOT = Path(__file__).resolve().parent

WEIGHTS_ROOT = ROOT / "weights"
DATASETS_ROOT = ROOT / "datasets"
OUTPUT_ROOT = ROOT / "outputs"

CKPT_MASK3D = WEIGHTS_ROOT / "mask3d" / "scannet200_val.ckpt"
CKPT_UNI3D = WEIGHTS_ROOT / "uni3d" / "uni3d_g.pth"
CKPT_CLIP_EVA02 = WEIGHTS_ROOT / "clip" / "EVA02-E-14-plus" / "open_clip_pytorch_model.bin"
CKPT_SAM = WEIGHTS_ROOT / "sam" / "sam_vit_h_4b8939.pth"
CKPT_VICUNA = WEIGHTS_ROOT / "llm" / "vicuna-7b-v1.5"

SCANS_PATH = DATASETS_ROOT / "scannetv2_frames"
SCANNET_PROC = DATASETS_ROOT / "scannet200_processed"
MASKS_DIR = OUTPUT_ROOT / "3d_masks"
FEATS3D_DIR = OUTPUT_ROOT / "3d_feats"
FEATS2D_DIR = OUTPUT_ROOT / "2d_feats"
ANNO_ROOT = DATASETS_ROOT / "annotations"


def main():
    for p in [WEIGHTS_ROOT, DATASETS_ROOT, OUTPUT_ROOT, MASKS_DIR, FEATS3D_DIR, FEATS2D_DIR, ANNO_ROOT]:
        Path(p).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
