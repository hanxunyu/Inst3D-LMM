import datetime
import logging
import time
from os.path import join

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

from dataset import MetaLoader, create_dataset, create_loader, create_sampler
from dataset.dataset_train import train_collate_fn
from dataset.dataset_val import val_collate_fn
from models.inst3d import Inst3D
from tasks.utils_backup import get_media_types, setup_model
from utils.basic_utils import (MetricLogger, SmoothedValue, setup_seed)
from utils.config_utils import setup_main
from utils.distributed import get_rank, get_world_size, is_main_process
from utils.logger import log_dict_to_wandb, setup_wandb
from utils.eval import calc_scanrefer_score, clean_answer, calc_scan2cap_score, calc_scanqa_score, calc_sqa3d_score, calc_multi3dref_score

import pandas as pd
import torch
import torch.distributed as dist
import wandb
from torch.utils.data import ConcatDataset

import numpy as np
from tqdm import tqdm

import json
import os

# Setup logger for this module
logger = logging.getLogger(__name__)
# Initialize the maximum BLEU scores for four different n-gram levels
max_bleus = [0.0] * 4
# Initialize the tokenizer
tokenizer = PTBTokenizer()
# Set the maximum number of global steps
max_global_step = 500,000
# Define the evaluation scorers
scorers = [
    (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
    (Meteor(), "METEOR"),
    (Rouge(), "ROUGE_L"),
    (Cider(), "CIDEr"),
]
def save_checkpoint(model, optimizer, scheduler, scaler, config, epoch, global_step):
    state_dict = {
        k: v for k, v in model.state_dict().items() 
        if model.named_parameters().get(k, torch.nn.Parameter()).requires_grad
    }
    save_obj = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config,
        "epoch": epoch,
        "global_step": global_step,
    }
    save_path = join(config.output_dir, f"ckpt_{epoch:02d}_{global_step}.pth")
    torch.save(save_obj, save_path)


def evaluate_all(
    model,
    val_loaders,
    epoch,
    global_step,
    device,
    config
):
    logger.info("Start evaluating...")
    model.eval()
    val_scores = {}
    for val_loader in val_loaders:
        new_val_scores = evaluate(model, val_loader, epoch, global_step, device, config)
        val_scores.update(new_val_scores)
    
    logger.info(f"[epoch={epoch}, global steps={global_step}] Val Results:")
    for metric, score in val_scores.items():
        logger.info(f"{metric}: {score}")
    
    model.train()
    model.module.llama_model.config.use_cache = False
    return val_scores


def evaluate(model, val_loader, epoch, global_step, device, config):
    eval_name = val_loader.dataset.datasets[0].dataset_name
    logger.info(f"Evaluating {eval_name}...")

    if config.distributed:
        val_loader.sampler.set_epoch(epoch)

    sample_freq = max(1, len(val_loader) // 5)
    cosine_scores, l2_distances, save_preds = [], [], []

    logger.info(f"Batch Size: {val_loader.batch_size}, Number of Batches: {len(val_loader)}")
    
    for i, batch in enumerate(tqdm(val_loader)):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        with torch.no_grad():
            pred = model(**batch, is_eval=True)
        
        if "target_captions" in batch:
            cosine_scores.append(pred["cosine_score"])
            l2_distances.append(pred["l2_dis"])
        
        if "custom_prompt" in batch:
            for bi in range(len(pred)):
                save_preds.append({
                    "scene_id": batch["scene_id"][bi],
                    "gt_id": int(batch["obj_ids"][bi]),
                    "pred_id": int(batch['pred_ids'][bi]),
                    "qid": batch["qid"][bi],
                    "prompt": batch["custom_prompt"][bi],
                    "pred": pred[bi],
                    "ref_captions": batch["ref_captions"][bi],
                    "type_info": batch['type_infos'][bi]
                })
            
            if i % sample_freq == 0:
                print(save_preds[-1])

    if save_preds:
        save_preds = sorted(save_preds, key=lambda x: f"{x['scene_id']}_{x['gt_id']:03}_{x['qid']}")
        output_path = os.path.join(config.output_dir, f"preds_epoch{epoch}_step{global_step}_rank{get_rank()}_{eval_name}.json")
        
        with open(output_path, "w") as f:
            json.dump(save_preds, f, indent=4)

    if is_main_process():
        save_preds = []
        for rank in range(config.gpu_num):
            path = os.path.join(config.output_dir, f"preds_epoch{epoch}_step{global_step}_rank{rank}_{eval_name}.json")
            if os.path.exists(path):
                with open(path, "r") as f:
                    save_preds.extend(json.load(f))
                os.remove(path)

        save_preds = sorted(save_preds, key=lambda x: f"{x['scene_id']}_{x['gt_id']:03}_{x['qid']}")
        final_output_path = os.path.join(config.output_dir, f"preds_epoch{epoch}_step{global_step}_{eval_name}.json")
        
        with open(final_output_path, "w") as f:
            json.dump(save_preds, f, indent=4)
    
                
    val_scores = {}
    if is_main_process() and len(save_preds) > 0:
        if eval_name == 'scanqa':
            val_scores = calc_scanqa_score(save_preds, tokenizer, scorers, config)
        elif eval_name == 'scanrefer':
            val_scores = calc_scanrefer_score(save_preds,config)
        elif eval_name == "scan2cap":
            val_scores = calc_scan2cap_score(save_preds, tokenizer, scorers, config)
        elif eval_name == 'multi3dref':
            val_scores = calc_multi3dref_score(save_preds, config)
        else:
            raise NotImplementedError
        print(json.dumps(val_scores, indent=4))
    return val_scores

def setup_dataloaders(config):
    # train datasets, create a list of data loaders
    train_datasets, val_datasets = create_dataset(config)
    print(f"the length of train_datasets is {len(train_datasets)}")
    if config.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        train_samplers = create_sampler(
            train_datasets, [True] * len(train_datasets), num_tasks, global_rank
        )
        val_samplers = create_sampler(
            val_datasets, [False] * len(val_datasets), num_tasks, global_rank
        )
    else:
        train_samplers = [None] * len(train_datasets)
        val_samplers = [None] * len(val_datasets)

    train_loaders = create_loader(
        train_datasets,
        train_samplers,
        batch_size=[config.batch_size] * len(val_datasets),
        num_workers=[config.num_workers] * len(train_datasets),
        is_trains=[True] * len(train_datasets),
        collate_fns=[train_collate_fn] * len(train_datasets),
    )
    val_loaders = create_loader(
        val_datasets,
        val_samplers,
        batch_size=[config.batch_size] * len(val_datasets),
        num_workers=[config.num_workers] * len(val_datasets),
        is_trains=[False] * len(val_datasets),
        collate_fns=[val_collate_fn] * len(val_datasets),
    )

    return train_loaders, val_loaders


def main(config):

    # Set random seed and device
    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)

    # Prepare data loaders for training and validation
    train_loaders, val_loaders = setup_dataloaders(config)

    # Calculate steps per epoch and configure scheduler
    steps_per_epoch = sum(len(loader) for loader in train_loaders)
    config.scheduler.num_training_steps = steps_per_epoch * config.scheduler.epochs
    config.scheduler.num_warmup_steps = steps_per_epoch * config.scheduler.warmup_epochs

    # Enable benchmark mode in cuDNN for faster training
    torch.backends.cudnn.benchmark = True

    # Initialize model, optimizer, scheduler, scaler, and training state
    model_cls = eval(config.model.get('model_cls', 'Inst3D'))
    model, optimizer, scheduler, scaler, start_epoch, global_step = setup_model(
        config, model_cls=model_cls, find_unused_parameters=True
    )

    save_step_interval = 1
    start_time = time.time()


    if config.evaluate:
        logger.info(f"Global step: {global_step}")
        evaluate_all(model, val_loaders, start_epoch - 1, global_step, device, config)

    # Log total Inference time
    total_time = time.time() - start_time
    logger.info(f"Inference time: {str(datetime.timedelta(seconds=int(total_time)))}")
    logger.info(f"Checkpoints and logs saved at {config.output_dir}")



if __name__ == "__main__":
    cfg = setup_main()
    main(cfg)
