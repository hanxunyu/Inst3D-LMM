from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths

REQUIRED = [
    'WEIGHTS_ROOT','DATASETS_ROOT','OUTPUT_ROOT','ANNO_ROOT','CKPT_MASK3D','CKPT_UNI3D',
    'CKPT_CLIP_EVA02','TIMM_EVA_GIANT','CKPT_SAM','CKPT_CLIP336','CKPT_VICUNA',
    'SCANS_PATH','SCANNET_PROC','MASKS_DIR','FEATS3D_DIR','FEATS2D_DIR'
]

def test_constants_present():
    for name in REQUIRED:
        assert hasattr(paths, name)

BAD_LITERALS = [
    'openai/clip-vit-large-patch14-336',
    'laion2b_s9b_b144k'
]

def test_no_bad_literals():
    for p in Path('.').rglob('*.py'):
        if p.name == Path(__file__).name:
            continue
        text = p.read_text(errors='ignore')
        for bad in BAD_LITERALS:
            assert bad not in text
        if 'timm.create_model(' in text:
            assert 'pretrained=True' not in text

