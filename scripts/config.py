# ========================= data ==========================
anno_root = "annotations"  # annotation dir
pc_encoder = "uni3d" # or ulip2
segmentor = "mask3d" # or pointgroup


seg_feat_file = f"{anno_root}/scannet_{segmentor}_{pc_encoder}_feats.pt"
seg_img_feat_file = f"{anno_root}/scannet_total_image_feat.pt"

seg_train_attr_file = f"{anno_root}/scannet_{segmentor}_train_attributes.pt"

seg_val_attr_file = f"{anno_root}/scannet_{segmentor}_val_attributes.pt"

train_tag = 'scanrefer#multi3dref#scanqa#scan2cap#scannet_caption'
val_tag = 'scanrefer#multi3dref#scanqa#scan2cap'

train_file_dict = {
    'scanrefer': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/scanrefer_{segmentor}_train.json"
    ],
    'scan2cap': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/scan2cap_{segmentor}_train.json"
    ],
    'scanqa': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/scanqa_train.json"
    ],
    'multi3dref': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/multi3dref_{segmentor}_train.json"
    ],
    'scannet_caption': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/3d_llm_scene_description_train.json"
    ],
}

val_file_dict = {
    'scanqa': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f"{anno_root}/scanqa_val.json"
    ],
    'scanrefer': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f"{anno_root}/scanrefer_{segmentor}_val.json"
    ],
    'scan2cap': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f"{anno_root}/scan2cap_{segmentor}_val.json"
    ],
    'multi3dref': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f"{anno_root}/multi3dref_{segmentor}_val.json"
    ]
}


num_workers = 32
batch_size = 32


# ========================= model ==========================
model = dict(
    llama_model_path="/vicunna-7b-v1.5/",
    clip_path="/CLIP-ViT-L/14-336px",
    sam_path="/SAM-ViT-H/",
    model_cls="Inst3D",
    input_dim=1024 if pc_encoder == "uni3d" else 512,#
    img_input_dim=768, # CLIP embedding space
    attr_dim=512,
    scene_dim=256,
    encoder_num_layers=3,
    low_resource=False,
    system_path="instruction_templates/system.txt",
    instruction_path="instruction_templates/instruction.txt",
    max_txt_len=64,
    end_sym="</s>",
    role=("USER", "ASSISTANT"),
    add_scene_token=True,
    add_img_token=True,
    obj_norm_scale=200,
    scene_norm_scale=50,
    grad_scale=1,
    use_lora=True,
    train_emb=True,
    train_img_proj=True
)

lora = dict(
    lora_target_modules=[
      "q_proj",
      "v_proj",
      "k_proj",
      "o_proj",
      "gate_proj",
      "up_proj",
      "down_proj"
    ],
    lora_r=32,
    lora_alpha=16, 
    lora_dropout=0.1
)

optimizer = dict(
    opt="adamW",
    lr=5e-6,
    opt_betas=[0.9, 0.999],  # default
    weight_decay=0.02,
    scaler_enable=False,
    max_grad_norm=-1,  # requires a positive float, use -1 to disable
    # use a different lr for some modules, e.g., larger lr for new modules
    different_lr=dict(
        enable=False,
        module_names=["module.object_img_proj"],
        lr=[5e-7],
        wd=[0.02]
    ),
)

scheduler = dict(sched="cosine", epochs=3, min_lr_multi=0.01, warmup_epochs=0.1)

evaluate = False

# ========================= others ==========================
output_dir = ""  # output dir
resume = False  # if True, load optimizer and scheduler states as well
debug = False
log_freq = 800
seed = 38

save_latest = False
do_save = True
auto_resume = True
pretrained_path = ""

debug=False
gpu_num=8