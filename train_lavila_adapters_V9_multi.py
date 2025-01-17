import json
import os
import torchvision
import sys
# sys.path.insert(0, './lavila')
sys.path.insert(0, '/scratch/qt2087/DSGA1006_code/lavila')
import gc

from lavila_utils import *
from lavila.lavila.utils.preprocess import generate_tokenizer

# import lavila modles
import lavila_adapters_V2 as lavila_adapters
# suppress warnings
import warnings
warnings.filterwarnings("ignore")
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
# from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.io import read_video
import torchvision.transforms as transforms
import math
import numpy as np

from tqdm import tqdm

from time import time

import pathlib

from transformers import DistilBertTokenizer
from torch.amp import GradScaler, autocast
from time import time

from train_utils_V9 import *
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

# Set up environment variables to assist with NCCL debugging and to handle errors gracefully
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'

def setup(rank, world_size, batch_size, negative_sampling_ratio):
    dist.init_process_group(
        backend='nccl', 
        init_method='env://',
        rank=rank,
        world_size=world_size
    )
    with open(f'/scratch/qt2087/DSGA1006_code/logs/V9_{batch_size}_{negative_sampling_ratio}.log', 'a') as f:
        f.write(f"Process group initialized for rank {rank} with {world_size}.\n")
        # f.write(f"\tMemory Allocated: {allocated:.2f} GB, Memory Reserved: {reserved:.2f} GB, Total Memory: {total_memory:.2f} GB\n")
    f.close()
    print(f"Process group initialized for rank {rank}.")

def cleanup():
    dist.destroy_process_group()

def initialize_model_with_adapters(state_dict, n_frames, old_args, adapter_dim=64):
    # Load checkpoint 
    print('=> creating model with adapters:', old_args.model)
    
    # Initialize the model with all the parameters as in the original function
    model = getattr(lavila_adapters, old_args.model)(
        text_use_cls_token=old_args.use_cls_token,
        project_embed_dim=old_args.project_embed_dim,
        gated_xattn=False if 'gated_xattn' not in old_args else old_args.gated_xattn,
        timesformer_gated_xattn=False if 'timesformer_gated_xattn' not in old_args else old_args.timesformer_gated_xattn,
        timesformer_freeze_space=False if 'timesformer_freeze_space' not in old_args else old_args.timesformer_freeze_space,
        freeze_lm_vclm=False if 'freeze_lm_vclm' not in old_args else old_args.freeze_lm_vclm,
        freeze_visual_vclm=False if 'freeze_visual_vclm' not in old_args else old_args.freeze_visual_vclm,
        num_frames=n_frames,
        drop_path_rate=0,
        adapter_dim=adapter_dim,  # Include adapter dimension
    )

    # If necessary, adjust positional embeddings for TimeSformer models
    if 'TIMESFORMER' in old_args.model:
        print('=> inflating positional embeddings in model due to different frame numbers')
        state_dict = inflate_positional_embeds(
            model.state_dict(), state_dict,
            num_frames=n_frames,
            load_temporal_fix='bilinear',
        )

    # Load the state dictionary with strict=False to account for adapter layers
    print('Loading with strict=True')
    model.load_state_dict(state_dict, strict=True)
    
    # Set the model to evaluation mode if using for inference
    model.eval()
    return model


class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.Inf

    def __call__(self, val_loss):
        """
        if you use other metrics where a higher value is better, e.g. accuracy,
        call this with its corresponding negative value
        """
        if val_loss < self.best_loss:
            early_stop = False
            get_better = True
            self.counter = 0
            self.best_loss = val_loss
        else:
            get_better = False
            self.counter += 1
            if self.counter >= self.patience:
                early_stop = True
            else:
                early_stop = False

        return early_stop, get_better

def latest_checkpoint(directory):
    if not os.path.exists(directory):
        return None
    try:
        # latest_epoch = max([int(x.split('.')[-2].split('_')[-2]) for x in os.listdir(directory)])
        # latest_batch = max([int(str(latest_epoch)+x.split('.')[-2].split('_')[-1]) for x in os.listdir(directory)])
        latest_epoch = 0
        latest_batch = 449
        # if latest_epoch > 0:
        #     latest_batch = int(str(latest_batch)[1:])
    except Exception:
        return None
    return os.path.join(directory, f'alignment_model_{latest_epoch}_{latest_batch}.pth')

def main(rank, world_size, batch_size=sys.argv[1]):

    setup(rank, world_size, batch_size=batch_size, negative_sampling_ratio=1)
    dist.barrier()

    # Set the correct device for each process
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    with open(f'/scratch/qt2087/DSGA1006_code/logs/V9_{batch_size}_{1}.log', 'a') as f:
        f.write(f"Using device: {device}.\n")
        # f.write(f"Environment Variables - RANK: {rank_env}, WORLD_SIZE: {world_size_env}, LOCAL_RANK: {local_rank_env}\n")
    f.close()
    print(f"Using device: {device}")


    video_root_dir = r'/scratch/qt2087/DSGA1006_code/processed_data/mark_code/goalstep_train'  # Path to video embeddings (subfolders per video with .pt files)
    annotation_file = r'/scratch/qt2087/DSGA1006_code/annotations/goalstep_train.json'  # Path to annotation file

    val_video_root_dir = r'/scratch/qt2087/DSGA1006_code/processed_data/mark_code/goalstep_val'
    val_annotation_file = r'/scratch/qt2087/DSGA1006_code/annotations/goalstep_val.json'  # Path to annotation file

    ckpt_path = r'/scratch/qt2087/DSGA1006_code/lavila/downloaded_models/clip_openai_timesformer_large_336px_distilbert_base.pth'
    ckpt, state_dict = load_ckpt_model_weight(ckpt_path)

    num_frames = 4
    model = initialize_model_from_ckpt(ckpt, state_dict, num_frames)
    
    args = ckpt['args']

    # clear unused variables for memory efficiency
    del ckpt
    del state_dict
    gc.collect()

    grad_counter = 0
    for name, param in model.named_parameters():
        if "adapter" not in name:  # Assuming adapters are named with "adapter" in their parameter names
            param.requires_grad = False
        else:
            param.requires_grad = True
            grad_counter += 1
    print(f'# of params requiring grad: {grad_counter}.')

    # Collect trainable parameters (only adapters should be trainable)
    lr = 1e-5
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=lr)
    early_stopping = EarlyStopping()
    
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    num_epochs = 3
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased',force_download=False)

    loading_device = 'cpu'
    if loading_device == 'cpu':
        pin_mem = True
        prefetch_num = 4
        n_worker = 4
    else:
        pin_mem = False
        prefetch_num = 2
        n_worker = 2

    # Create dataset and dataloader
    dataset = VideoTextAlignmentDataset(
        video_root_dir=video_root_dir,
        annotation_file=annotation_file,
        loading_device=loading_device,
        negative_sampling_ratio=1,  # Adjust as needed
        batch_size=batch_size
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    data_loader = DataLoader(dataset, 
                             batch_size=batch_size, 
                            #  shuffle=True, 
                             num_workers=n_worker, 
                             pin_memory=pin_mem,
                             sampler=sampler,  # Use DistributedSampler here
                            #  prefetch_factor = prefetch_num,
                             collate_fn=custom_collate_fn,  # Use the custom collate function here
                             drop_last=True
                             )
    total_batches = math.ceil(len(dataset) / batch_size / 2)
    if rank == 0:
        print(f"Dataset size: {len(dataset)}")
        print(f"Batch size: {batch_size}")
        print(f"Number of bathces: {total_batches}")

    lr_step = total_batches*0.1
    lr_step = round(lr_step / 10) * 10
    lr_scale = 0.8

    num_step_val = total_batches*0.5
    num_step_val = round(num_step_val / 10) * 10

    num_step_save = 20
    if rank == 0:
        with open(f'/scratch/qt2087/DSGA1006_code/logs/V9_{batch_size}_{dataset.negative_sampling_ratio}.log', 'a') as f:
            f.write(f"Dataset size: {len(dataset)}; Batch size: {batch_size}; Number of bathces: {total_batches}; loading on: {loading_device}.\n")
            f.write(f"Initial lr: {lr}; decreasing every {lr_step}; by scale {lr_scale}; validate every {num_step_val}\n")
        f.close()
        print(f"lr will be decreased every {lr_step} batches with scale {lr_scale}.")

    checkpoint_path = latest_checkpoint(f'/scratch/qt2087/DSGA1006_code/lavila_adapter_models/V9/{batch_size}_{dataset.negative_sampling_ratio}')
    if checkpoint_path:
        epoch=checkpoint_path.split('.')[-2].split('_')[-2]
        batch_idx=checkpoint_path.split('.')[-2].split('_')[-1]
        checkpoint = torch.load(checkpoint_path)
        step = checkpoint['step']
        checkpoint_state_dict = checkpoint['model_state_dict']
        # new_state_dict = {}
        # for k, v in checkpoint_state_dict.items():
        #     if "module." not in k:
        #         new_key = "module." + k
        #     else:
        #         new_key = k
        #     new_state_dict[new_key] = v
        new_state_dict = {}
        for k, v in checkpoint_state_dict.items():
            new_key = k.replace("module.", "")
            new_state_dict[new_key] = v
        model = initialize_model_with_adapters(new_state_dict, num_frames, args)
        # model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'total_loss' in checkpoint:
            total_loss = checkpoint['total_loss']
        else:
            total_loss = []
            with open(r'/scratch/qt2087/DSGA1006_code/logs/total_loss_list_V9.txt', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    total_loss.append(float(line))
            f.close()
            with open(f'/scratch/qt2087/DSGA1006_code/logs/V9_{batch_size}_{dataset.negative_sampling_ratio}.log', 'a') as f:
                f.write(f"Total loss loaded from txt.\n")
            f.close()
        if rank == 0:
            print(f'Model loaded from {epoch}-{batch_idx}')
            with open(f'/scratch/qt2087/DSGA1006_code/logs/V9_{batch_size}_{dataset.negative_sampling_ratio}.log', 'a') as f:
                f.write(f"Model loaded from {epoch}-{batch_idx}\n")
            f.close()
    else:
        if rank == 0:
            print(f'Initializing Model')
            with open(f'/scratch/qt2087/DSGA1006_code/logs/V9_{batch_size}_{dataset.negative_sampling_ratio}.log', 'a') as f:
                f.write(f'# of params requiring grad: {grad_counter}; Initializing Model...\n')
            f.close()
        epoch = 0
        batch_idx = 0
        step = 0
        total_loss = []
        
    model = torch.nn.parallel.DistributedDataParallel(model.to(device), device_ids=[rank], output_device=rank)
    model = model.to(device)

    warm_up = False
    warm_up_step = 10
    target_lr = 1e-4
    warm_up_add = 1e-5

    loaded_batch_idx = int(batch_idx)
    loaded_epoch = int(epoch)
    print("Starting training...")
    write_path = f'/scratch/qt2087/DSGA1006_code/logs/V9/{batch_size}_{dataset.negative_sampling_ratio}'
    os.makedirs(write_path, exist_ok=True)
    writer = SummaryWriter(write_path)
    prev_epoch_time = time()

    model.train()

    for epoch in range(num_epochs):
        model.train()
        if epoch < loaded_epoch:
            if rank == 0:
                with open(f'/scratch/qt2087/DSGA1006_code/logs/V9_{batch_size}_{dataset.negative_sampling_ratio}.log', 'a') as f:
                    f.write(f"Skip Trained epoch\n")
                f.close()
            continue
        prev_batch_time = time()
        sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(data_loader):
            if epoch == loaded_epoch and batch_idx <= loaded_batch_idx and loaded_batch_idx > 0:
                if rank == 0:
                    with open(f'/scratch/qt2087/DSGA1006_code/logs/V9_{batch_size}_{dataset.negative_sampling_ratio}.log', 'a') as f:
                        f.write(f"Skip Trained batch\n")
                    f.close()
                continue
            if batch is None:
                print(f'Error loading batch {batch_idx}. Skip')
                with open(f'/scratch/qt2087/DSGA1006_code/logs/V9_{batch_size}_{dataset.negative_sampling_ratio}.log', 'a') as f:
                    f.write(f"Error loading batch {batch_idx}. Skip\n")
                f.close()
                if (batch_idx+1) % lr_step == 0 and rank == 0:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= lr_scale
                        
                        lr_tensor = torch.tensor(optimizer.param_groups[0]['lr'], dtype=torch.float32, device=device)
                        dist.broadcast(lr_tensor, src=0)

                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_tensor.item()

                        with open(f'/scratch/qt2087/DSGA1006_code/logs/V9_{batch_size}_{dataset.negative_sampling_ratio}.log', 'a') as f:
                            f.write(f"\tLr adjusted to: {optimizer.param_groups[0]['lr']}\n")
                        f.close()
                prev_batch_time = time()
                continue
            step += 1
            positive_pairs = batch['positive_pairs']
            negative_pairs = batch['negative_pairs']
            if len(positive_pairs) < batch_size or len(negative_pairs) < batch_size:
                with open(f'/scratch/qt2087/DSGA1006_code/logs/V9_{batch_size}_{dataset.negative_sampling_ratio}.log', 'a') as f:
                    f.write(f"[Rank {rank}] {batch_idx}: truncated batch: {len(positive_pairs)}\n")
                f.close()

            loss = contrastive_loss(positive_pairs, negative_pairs, model, tokenizer = tokenizer, device=device)

            # Backpropagation and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del positive_pairs, negative_pairs
            torch.cuda.empty_cache()
            gc.collect()

            total_loss.append(loss.item())
            avg_loss = np.mean(total_loss)
            if rank == 0:
                with open(f'/scratch/qt2087/DSGA1006_code/logs/V9_{batch_size}_{dataset.negative_sampling_ratio}.log', 'a') as f:
                    f.write(f"[Rank {rank}] Epoch {epoch}, Batch {(batch_idx)}/{total_batches}, Loss: {loss.item()}, Avg Loss: {avg_loss}, Time taken: {time() - prev_batch_time}\n")
                f.close()
                writer.add_scalar(f'Training Loss', loss.item(), step)
                writer.add_scalar(f'Avg Training Loss', avg_loss, step)
            if warm_up:
                for param_group in optimizer.param_groups:
                    param_group['lr'] += warm_up_add
                
                lr_tensor = torch.tensor(optimizer.param_groups[0]['lr'], device=device)
                dist.broadcast(lr_tensor, src=0)

                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_tensor.item()
                warm_up_step -= 1
                try:
                    with open(f'/scratch/qt2087/DSGA1006_code/logs/V9_{batch_size}_{dataset.negative_sampling_ratio}.log', 'a') as f:
                        f.write(f"\ttarget_lr: {target_lr}, current: {optimizer.param_groups[0]['lr']}\n")
                    f.close()
                except Exception:
                    pass
                if warm_up_step <= 0:
                    warm_up = False
            else:
                if (batch_idx+1) % lr_step == 0 and rank == 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= lr_scale
                    
                    lr_tensor = torch.tensor(optimizer.param_groups[0]['lr'], dtype=torch.float32, device=device)
                    dist.broadcast(lr_tensor, src=0)

                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_tensor.item()

                    with open(f'/scratch/qt2087/DSGA1006_code/logs/V9_{batch_size}_{dataset.negative_sampling_ratio}.log', 'a') as f:
                        f.write(f"\tLr adjusted to: {optimizer.param_groups[0]['lr']}\n")
                    f.close()

            prev_batch_time = time()

           
            if (batch_idx+1) % num_step_save == 0 and rank == 0:
                try:
                    model_dir = f'/scratch/qt2087/DSGA1006_code/lavila_adapter_models/V9/{batch_size}_{dataset.negative_sampling_ratio}'
                    os.makedirs(model_dir, exist_ok=True)
                    torch.save(
                                {
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict':
                                    optimizer.state_dict(),
                                    'step':
                                    step,
                                    'total_loss':
                                    total_loss,
                                }, os.path.join(model_dir, f'alignment_model_{epoch}_{batch_idx}.pth'))
                except OSError as error:
                    print(f"OS error: {error}")
        if rank == 0:
            model.eval()
            val_loss = evaluate(model, val_video_root_dir, val_annotation_file, tokenizer, for_val=1.0, loading_device='cpu')
            model.train()
            writer.add_scalar(f'Validation Loss', val_loss, step)
            with open(f'/scratch/qt2087/DSGA1006_code/logs/V9_{batch_size}_{dataset.negative_sampling_ratio}.log', 'a') as f:
                f.write(f"[Rank {rank}] Epoch {epoch}, Batch {(batch_idx)}/{total_batches}, Val Loss: {val_loss}\n")
            f.close()
            try:
                model_dir = f'/scratch/qt2087/DSGA1006_code/lavila_adapter_models/V9/{batch_size}_{dataset.negative_sampling_ratio}'
                os.makedirs(model_dir, exist_ok=True)
                torch.save(
                            {
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict':
                                optimizer.state_dict(),
                                'step':
                                step,
                                'total_loss':
                                total_loss,
                            }, os.path.join(model_dir, f'alignment_model_{epoch}_{batch_idx}.pth'))
            except OSError as error:
                with open(f'/scratch/qt2087/DSGA1006_code/logs/V9_{batch_size}_{dataset.negative_sampling_ratio}.log', 'a') as f:
                    f.write(f"OS error: {error}")
                f.close()
                print(f"OS error: {error}")
        dataset = VideoTextAlignmentDataset(
            video_root_dir=video_root_dir,
            annotation_file=annotation_file,
            loading_device=loading_device,
            negative_sampling_ratio=1,  # Adjust as needed
            batch_size=batch_size
        )
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        data_loader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            # shuffle=True, 
                            num_workers=n_worker, 
                            pin_memory=pin_mem, 
                            #  prefetch_factor = prefetch_num,
                            sampler=sampler,  # Use DistributedSampler here
                            collate_fn=custom_collate_fn,  # Use the custom collate function here
                            drop_last=True
                            )
        with open(f'/scratch/qt2087/DSGA1006_code/logs/V9_{batch_size}_{dataset.negative_sampling_ratio}.log', 'a') as f:
            f.write(f"Finished!!!")
        f.close()

if __name__ == '__main__':
    # call the main function with batch size same as input in the terminal
    world_size = torch.cuda.device_count()
    batch_size = 256
    if len(sys.argv) > 1:
        batch_size = int(sys.argv[1])

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    main(rank, world_size, batch_size)