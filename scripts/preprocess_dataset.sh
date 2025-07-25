#!/bin/bash
source "$(dirname "$0")/export_paths.sh"

scannet_dir="$SCANS_PATH"
segment_result_dir="$MASKS_DIR"
inst_seg_dir="$MASKS_DIR/instance/"
processed_data_dir="$SCANNET_PROC"
class_label_file="$SCANS_PATH/scannetv2-labels.combined.tsv"
segmentor=""
train_iou_thres=0.75


python dataset_preprocess/prepare_mask3d_data.py \
    --scannet_dir "$scannet_dir" \
    --output_dir "$processed_data_dir" \
    --segment_dir "$segment_result_dir" \
    --inst_seg_dir "$inst_seg_dir" \
    --class_label_file "$class_label_file" \
    --apply_global_alignment \
    --num_workers 16 \
    --parallel

python dataset_preprocess/prepare_scannet_mask3d_attributes.py \
    --scan_dir "$processed_data_dir" \
    --segmentor "$segmentor" \
    --max_inst_num 100

python dataset_preprocess/prepare_scannet_attributes.py \
    --scannet_dir "$scannet_dir"

python dataset_preprocess/prepare_scanrefer_annos.py \
    --segmentor "$segmentor" \
    --version "$version" \
    --train_iou_thres "$train_iou_thres"

python dataset_preprocess/prepare_scan2cap_annos.py \
    --segmentor "$segmentor" \
    --version "$version" \
    --train_iou_thres "$train_iou_thres"

python dataset_preprocess/prepare_multi3dref_annos.py \
    --segmentor "$segmentor" \
    --version "$version" \
    --train_iou_thres "$train_iou_thres"

python dataset_preprocess/prepare_scanqa_annos.py
