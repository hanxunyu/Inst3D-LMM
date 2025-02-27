export PYTHONPATH=${PYTHONPATH}:${which_python}:.

evaluate=False
# try different combination of training datasets as you want
train_tag="scanrefer#multi3dref#scan2cap#scanqa#scene_descriptions"
val_tag="scanrefer#multi3dref#scan2cap#scanqa"

epoch=5
batch_size=32
lr=5e-6
train_emb=True
train_img_proj=True
add_2D_token=True
add_scene_token=True
gpu_num=8
do_save=True

pretrained_checkpoint=""

if [ $evaluate = "True" ]; then
  OUTPUT_DIR=outputs/inference/"$(date +"%Y%m%d_%H%M%S")"_lr"$lr"_ep"$epoch"
else
  OUTPUT_DIR=outputs/train/"$(date +"%Y%m%d_%H%M%S")"_lr"$lr"_ep"$epoch"
fi

mkdir -p ${OUTPUT_DIR}

python run/train.py \
    $(dirname $0)/config.py \
    output_dir "$OUTPUT_DIR" \
    scheduler.epochs "$epoch" \
    optimizer.lr "$lr" \
    model.add_scene_token "$add_scene_token" \
    model.add_img_token "$add_2D_token" \
    pretrained_path "$pretrained_checkpoint" \
    evaluate "$evaluate" \
    gpu_num "$gpu_num" \
    do_save "$do_save" \
    batch_size "$batch_size" \
    model.train_emb "$train_emb" \
    model.train_img_proj "$train_img_proj" \
    train_tag "$train_tag" \
    val_tag "$val_tag"