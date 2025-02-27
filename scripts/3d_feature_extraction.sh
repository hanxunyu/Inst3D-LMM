# NOTE: SET THESE PARAMETERS!
model=create_uni3d

clip_model="EVA02-E-14-plus" 
pretrained="/path/to/clip_model/open_clip_pytorch_model.bin" # 

pc_model="eva_giant_patch14_560.m30m_ft_in22k_in1k"

ckpt_path="model.pt"

export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

SCANS_PATH="/sens2data_train/"
SCANNET_PROCESSED_DIR="/Mask3D/data/processed/scannet200/"
# model ckpt paths
MASK_MODULE_CKPT_PATH="$(pwd)/resources/scannet200_val.ckpt"
SAM_CKPT_PATH="$(pwd)/resources/sam_vit_h_4b8939.pth"

# output directories to save 3D instances and their features
EXPERIMENT_NAME="scannet200"
OUTPUT_DIRECTORY="$(pwd)/output"
TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
OUTPUT_FOLDER_DIRECTORY="${OUTPUT_DIRECTORY}/${TIMESTAMP}-${EXPERIMENT_NAME}"
MASK_SAVE_DIR="${OUTPUT_FOLDER_DIRECTORY}/masks"
MASK_FEATURE_SAVE_DIR="${OUTPUT_FOLDER_DIRECTORY}/mask_features"
SAVE_VISUALIZATIONS=false #if set to true, saves pyviz3d visualizations

# Paremeters below are AUTOMATICALLY set based on the parameters above:
SCANNET_LABEL_DB_PATH="${SCANNET_PROCESSED_DIR%/}/label_database.yaml"
SCANNET_INSTANCE_GT_DIR="${SCANNET_PROCESSED_DIR%/}/instance_gt/train"
# gpu optimization
OPTIMIZE_GPU_USAGE=true

set -e

# 1.Compute class agnostic 3D proposals and save them
python 3d_feature_extraction/get_3D_proposals.py \
general.experiment_name=${EXPERIMENT_NAME} \
general.project_name="scannet200" \
general.checkpoint=${MASK_MODULE_CKPT_PATH} \
general.train_mode=false \
model.num_queries=150 \
general.use_dbscan=true \
general.dbscan_eps=0.95 \
general.export=true \
general.save_visualizations=${SAVE_VISUALIZATIONS} \
data.test_dataset.data_dir=${SCANNET_PROCESSED_DIR}  \
data.validation_dataset.data_dir=${SCANNET_PROCESSED_DIR} \
data.train_dataset.data_dir=${SCANNET_PROCESSED_DIR} \
data.train_dataset.label_db_filepath=${SCANNET_LABEL_DB_PATH} \
data.validation_dataset.label_db_filepath=${SCANNET_LABEL_DB_PATH} \
data.test_dataset.label_db_filepath=${SCANNET_LABEL_DB_PATH}  \
general.mask_save_dir=${MASK_SAVE_DIR} \

# 2.Extract instance-level features using Uni3D
python 3d_feature_extraction/get_3D_features.py \
    --model $model \
    --batch-size 32 \
    --npoints 10000 \
    --num-group 512 \
    --group-size 64 \
    --pc-encoder-dim 512 \
    --clip-model $clip_model \
    --pretrained $pretrained \
    --pc-model $pc_model \
    --pc-feat-dim 1408 \
    --embed-dim 1024 \
    --validate_dataset_name modelnet40_openshape \
    --validate_dataset_name_lvis objaverse_lvis_openshape \
    --validate_dataset_name_scanobjnn scanobjnn_openshape \
    --evaluate_3d \
    --ckpt_path $ckpt_path
