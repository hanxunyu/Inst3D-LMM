#! /bin/bash

scannet_dir="/dataset/scannet/"
segment_result_dir="/dataset/mask3d_scannet_seg_results_wo_dbscan/"
inst_seg_dir="/dataset/mask3d_scannet_seg_results_wo_dbscan/instance/"
processed_data_dir="/dataset/processed_mask3d_ins_data/"
class_label_file="/dataset/scannet/scannetv2-labels.combined.tsv"
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
