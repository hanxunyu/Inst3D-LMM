#!/usr/bin/env bash
set -e

# Install PyTorch with CUDA 12.1 support and Ninja
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 \
  --extra-index-url https://download.pytorch.org/whl/cu121
pip install ninja==1.10.2.3

# Core Python dependencies
pip install pytorch-lightning==1.7.2 fire==0.5.0 imageio==2.23.0 tqdm==4.64.1 wandb==0.13.2
pip install python-dotenv==0.21.0 pyviz3d==0.2.32 scipy==1.9.3 plyfile==0.7.4 scikit-learn==1.2.0 \
  trimesh==3.17.1 loguru==0.6.0 albumentations==1.3.0 volumentations==0.1.8
pip install antlr4-python3-runtime==4.8 black==21.4b2 omegaconf==2.0.6 hydra-core==1.0.5 --no-deps

# Detectron2
pip install 'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf' --no-deps

# OpenBLAS and MinkowskiEngine
sudo apt-get update
sudo apt-get install -y libopenblas-dev
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --config-settings="--blas=openblas"

# Additional utilities
pip install pynvml==11.4.1 gpustat==1.0.0 tabulate==0.9.0 pytest==7.2.0 tensorboardx==2.5.1 yapf==0.32.0 termcolor==2.1.1 addict==2.4.0 blessed==1.19.1
pip install gorilla-core==0.2.7.8
pip install matplotlib==3.7.2
pip install cython

pip install pycocotools==2.0.6
pip install h5py==3.7.0
pip install transforms3d==0.4.1
pip install open3d==0.16.0
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip install torchmetrics==0.11.0
pip install setuptools==68.0.0

pip install git+https://github.com/openai/CLIP.git@a9b1bf5920416aaeaec965c25dd9e8f98c864f16 --no-deps
pip install git+https://github.com/facebookresearch/segment-anything.git@6fdee8f2727f4506cfbbe553e23b895e27956588 --no-deps
pip install ftfy==6.1.1
pip install regex==2023.10.3

pip install fvcore==0.1.5.post20221221
pip install cloudpickle==2.1.0
pip install Pillow==9.3.0
pip install pandas==1.5.3
pip install transformers==4.39.3
pip install einops==0.6.1
pip install plyfile==1.0.1
pip install trimesh==3.23.1
pip install peft==0.9.0
pip install termcolor==2.3.0
pip install scipy==1.12.0
pip install pycocoevalcap==1.2
pip install sentencepiece==0.2.0
pip install protobuf==4.25.3
pip install packaging==24.0
pip install flash_attn==2.5.6
pip install mmengine==0.10.3
pip install wandb==0.16.5
pip install nltk==3.8.1
