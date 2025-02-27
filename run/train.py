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

def train(
    model,
    train_loaders,
    val_loaders,
    optimizer,
    epoch,
    global_step,
    device,
    scheduler,
    scaler,
    config,
    do_eval=True
):
    model.train()
    
    # Initialize logging and metrics
    metric_logger = MetricLogger(delimiter="  ")
    eval_metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window=1, fmt="{value:.6f}"))
    loss_names = ["loss", "obj_norm", "scene_norm"]
    media_types = get_media_types(train_loaders)

    for name in loss_names:
        metric_logger.add_meter(f"{name}", SmoothedValue(window=1, fmt="{value:.6f}"))

    header = f"Train Epoch: [{epoch}]"
    log_freq = config.log_freq

    # Prepare loaders for distributed training
    if config.distributed:
        for loader in train_loaders:
            loader.sampler.set_epoch(epoch)
    train_loader = MetaLoader(name2loader=dict(zip(media_types, train_loaders)))

    # Training loop variables
    accum_iter = 1
    eval_freq = 800

    optimizer.zero_grad()
    for i, (media_type, batch) in enumerate(metric_logger.log_every(train_loader, log_freq, header)):
        # Move batch to the appropriate device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward pass and loss computation
        loss_dict = model(**batch)
        loss = loss_dict["loss"] / accum_iter
        
        # Backward pass
        scaler.scale(loss).backward()

        # Gradient accumulation and optimization step
        if (i + 1) % accum_iter == 0 or (i + 1) == len(train_loader):
            if config.optimizer.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
            scaler.step(optimizer)
            optimizer.zero_grad()
            scaler.update()
        scheduler.step()

        # Update logging metrics
        for name in loss_names:
            if name in loss_dict:
                metric_logger.update(**{name: loss_dict[name].item() if not isinstance(loss_dict[name], float) else loss_dict[name]})
        metric_logger.update(lr=optimizer.param_groups[-1]["lr"])

        # Log metrics to WandB
        if is_main_process() and config.wandb.enable and global_step % log_freq == 0:
            log_dict_to_wandb(metric_logger.get_avg_dict(), step=global_step, prefix="train/")

        global_step += 1

        # Periodic evaluation
        if do_eval and (i + 1) % eval_freq == 0 or i == len(train_loader) - 1:
            val_metrics = evaluate_all(model, val_loaders, epoch, global_step, device, config)
            if is_main_process():
                for k, v in val_metrics.items():
                    if k not in eval_metric_logger.meters:
                        eval_metric_logger.add_meter(k, SmoothedValue(window=1, fmt="{value:.4f}"))
                eval_metric_logger.update(**val_metrics)
                
                if config.wandb.enable:
                    log_dict_to_wandb(eval_metric_logger.get_avg_dict(), step=global_step, prefix="val/")
                
                # Save intermediate checkpoints
                if config.do_save and not config.debug and i != len(train_loader) - 1:
                    save_checkpoint(model, optimizer, scheduler, scaler, config, epoch, global_step)

        if global_step > config.max_global_step:
            return global_step

    # Synchronize metrics across processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger.global_avg()}")

    return global_step

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

    if not config.evaluate:
        logger.info("Start training")
        for epoch in range(start_epoch, config.scheduler.epochs):
            global_step = train(
                model, train_loaders, val_loaders, optimizer, epoch, global_step,
                device, scheduler, scaler, config
            )

            if is_main_process():
                logger.info(f"Completed Epoch {epoch}")
                # Filter and save model state
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
                # Save checkpoint
                if (
                    (epoch + 1) % save_step_interval == 0 or 
                    epoch == config.scheduler.epochs - 1
                ) and config.do_save and not config.debug:
                    save_path = "ckpt_latest.pth" if config.get("save_latest", False) else f"ckpt_{epoch:02d}_{global_step}.pth"
                    torch.save(save_obj, join(config.output_dir, save_path))

            if global_step > config.max_global_step:
                break

            dist.barrier()

    # Log total training time
    total_time = time.time() - start_time
    logger.info(f"Training time: {str(datetime.timedelta(seconds=int(total_time)))}")
    logger.info(f"Checkpoints and logs saved at {config.output_dir}")



if __name__ == "__main__":
    cfg = setup_main()
    main(cfg)
