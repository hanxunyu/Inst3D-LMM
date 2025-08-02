<h2 align="center">
  <b>Inst3D-LMM: Instance-Aware 3D Scene Understanding with Multi-modal Instruction Tuning</b>

  <b>[CVPR 2025 Highlight ⭐]</b>
</h2>

This is the official implementation of ["Inst3D-LMM: Instance-Aware 3D Scene Understanding with Multi-modal Instruction Tuning"](https://arxiv.org/pdf/2503.00513).

All results of our Inst3D-LMM are evaluated on the same model **without fine-tuning on specific tasks**. 
#
### 📰 News
* **`Apr. 5th, 2025`:** Inst3D-LMM is accepted by CVPR 2025 (**Highlight, 2.9%**)!
* **`Mar. 1st, 2025`:** Paper is available at [arXiv](https://arxiv.org/pdf/2503.00513). ☕️
* **`Feb. 27th, 2025`:** We released our code! Paper is coming soon. Please stay tuned! ☕️

## 🔍 Abstract

Despite encouraging progress in 3D scene understanding, it remains challenging to develop an effective Large Multi-modal Model (LMM) that is capable of understanding and reasoning in complex 3D environments. Most previous methods typically encode 3D point and 2D image features separately, neglecting interactions between 2D semantics and 3D object properties, as well as the spatial relationships within the 3D environment. This limitation not only hinders comprehensive representations of 3D scene, but also compromises training and inference efficiency. To address these challenges, we propose a unified Instance-aware 3D Large Multi-modal Model (Inst3D-LMM) to deal with multiple 3D scene understanding tasks simultaneously. To obtain the fine-grained instance-level visual tokens, we first introduce a novel Multi-view Cross-Modal Fusion (MCMF) module to inject the multi-view 2D semantics into their corresponding 3D geometric features. For scene-level relation-aware tokens, we further present a 3D Instance Spatial Relation (3D-ISR) module to capture the intricate pairwise spatial relationships among objects. Additionally, we perform end-to-end multi-task instruction tuning simultaneously without the subsequent task-specific fine-tuning. Extensive experiments demonstrate that our approach outperforms the state-of-the-art methods across 3D scene understanding, reasoning and grounding tasks. 

<div align="left">
<img src="assets/pipeline.png" width="99%" alt="Inst3D-LLM">
</div>

## 🛠️ Preparation

- Prepare the environment:
  
  ```shell
  conda create -n inst3d-lmm python=3.8 # create a virtual environment
  conda activate inst3d-lmm # activate it
  bash requirements.sh # installation requirements
  pip install -e . # install current repository in editable mode
  python -m paths --mk  # create folder structure
  ```
  
- Download LLM and other foundation models backbone:
  -  We use Vicuna-7B v1.5 ([Hugging Face](https://huggingface.co/lmsys)), the vision encoder from CLIP-ViT-L/14-336px ([Hugging Face](https://huggingface.co/openai/clip-vit-large-patch14-336)) and the ViT-H-based SAM ([Hugging Face](https://huggingface.co/facebook/sam-vit-huge));

### 完全离线准备

1. 运行 `python -m paths --mk` 创建目录结构。
2. 手动下载以下权重并放置到指定相对路径：
   - Vicuna‑7B v1.5 → `weights/llm/vicuna-7b-v1.5/`
   - OpenAI CLIP ViT‑L/14‑336 → `weights/clip/clip-vit-large-patch14-336/`
   - SAM ViT‑H → `weights/sam/sam_vit_h_4b8939.pth`
   - Mask3D ckpt → `weights/mask3d/scannet200_val.ckpt`
   - Uni3D‑g → `weights/uni3d/uni3d_g.pth`
   - OpenCLIP EVA02‑E‑14‑plus → `weights/open_clip/EVA02-E-14-plus/open_clip_pytorch_model.bin`
   - timm EVA‑giant‑560 → `weights/timm/eva_giant_patch14_560.m30m_ft_in22k_in1k/`
3. 本地生成 ScanNet200 processed 数据集至 `datasets/scannet200_processed/`。
4. `source scripts/export_paths.sh` 后即可运行各脚本。

  - Change their path in `scripts/config.py` to the corresponding download location if different.
  
- Dataset Preprocessing:
  
  - Download the full ScanNetv2 dataset and original ScanRefer, Multi3DRefer, ScanQA and Scan2Cap to `annotations/`;
  - run `bash scripts/preprocess_dataset.sh`
  
- Our system messages, instructions, and prompts are provided at `instruction_templates/`. 

## 🚀 Training 

​	**Step1:** Instance-level 3D feature extraction (corresponding to the folder `3d_feature_extraction`):

- We use [Mask3D](https://github.com/JonasSchult/Mask3D) (the model trained on the ScanNet200 training set) to obtain segmented 3D proposals in a ***class-agnostic*** manner;
- Then we use [Uni3D](https://github.com/baaivision/Uni3D?tab=readme-ov-file) (the pre-trained model [uni3d-g](https://huggingface.co/BAAI/Uni3D/blob/main/modelzoo/uni3d-g/model.pt)) to extract 3D instance-level features ;
- run `bash scripts/3d_feature_extraction.sh`.

​	**Step2:** Multi-view 2D feature extraction (corresponding to the folder `2d_feature_extraction`):

- Based on the 3D instance-level segmentation results, we use [SAM](https://github.com/facebookresearch/segment-anything) and [CLIP](https://github.com/openai/CLIP) to extract multi-view 2D features for each 3D instance;
- run `bash scripts/2d_feature_extraction.sh`.

​	**Step3:** End-to-end multi-task instruction-tuning:

- The code for training and evaluating the model is provided at `/run/train.py` and `/run/eval.py`;
- Modify `train_tag` in `scripts/run_train.sh` to specify the datasets used for end-to-end joint training. You can try different combination of training datasets or add customized datasets as you want;
- run `scripts/run_train.sh`.

## 🤖 Inference

- Modify `evaluate=True` and `pretrained_checkpoint="outputs/ckpt_latest.pth"` in `/scripts/run_eval.sh`;
- Modify `val_tag` in `scripts/run_eval.sh` to specify the datasets used for evaluation as you want;
- run `scripts/run_eval.sh`.		

## 😊 Acknowledgement

We are grateful for the open-source contributions of other projects:

- [Vicuna-v1.5](https://github.com/lm-sys/FastChat)
- [OpenMask3D](https://github.com/OpenMask3D/openmask3d)
- [VideoChat](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat)
- [Chat-Scene](https://github.com/ZzZZCHS/Chat-Scene)

## 🖊️ Citation

If you find our Inst3D-LMM useful for your research, please consider giving this repository a star and citing our paper as follows:
```bibtex
@misc{Inst3D-LMM,
    title={Inst3D-LMM: Instance-Aware 3D Scene Understanding with Multi-modal Instruction Tuning}, 
    author={Hanxun Yu and Wentong Li and Song Wang and Junbo Chen and Jianke Zhu},
    year={2025},
    eprint={2503.00513},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2503.00513}, 
}
```
