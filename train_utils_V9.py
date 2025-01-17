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

from time import time
import numpy as np

import pathlib

from transformers import DistilBertTokenizer
from torch.amp import GradScaler, autocast
from time import time

# set cuda device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

class VideoTextAlignmentDataset(Dataset):
    def __init__(self, video_root_dir, annotation_file, loading_device='cpu', negative_sampling_ratio=1, batch_size=64, for_val=None):
        self.video_root_dir = video_root_dir
        self.loading_device = loading_device
        self.for_val = for_val
        self.annotation_data = self.load_annotations(annotation_file)
        self.available_videos = self.get_available_videos()
        self.negative_sampling_ratio = max(1, int(negative_sampling_ratio))
        self.chunk_index = self.create_chunk_index_with_gaps()
        if for_val is not None:
            size = int(for_val * len(self.chunk_index))
            self.chunk_index = self.chunk_index[:size]
        self.batch_size = batch_size

    def load_annotations(self, annotation_file):
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        videos = data['videos']
        annotations = {video['video_uid']: video for video in videos}
        return annotations

    def get_available_videos(self):
        available_videos = []
        for video_uid in self.annotation_data.keys():
            video_path = os.path.join(self.video_root_dir, video_uid, f"{video_uid}_chunk_0.pt")
            if os.path.exists(video_path):
                video = self.annotation_data[video_uid]
                start_time = video['start_time']
                end_time = video['end_time']
                available_videos.append([video_uid, end_time - start_time])
        available_videos.sort(key=lambda x: x[1])
        available_videos = [video[0] for video in available_videos]
        random.seed(1006)
        random.shuffle(available_videos)
        return available_videos

    def create_chunk_index_with_gaps(self):
        chunk_index = []
        for video_uid in self.available_videos:
            segments = self.annotation_data[video_uid]['segments']
            segment_times = [(int(seg['start_time'] // 2), int(seg['end_time'] // 2), seg['step_description']) for seg in segments]
            max_chunk = max([end for _, end, _ in segment_times])

            for chunk_idx in range(max_chunk + 1):
                # Check if chunk matches any segment; if not, it's a gap
                matching_segment = next((desc for start, end, desc in segment_times if start <= chunk_idx <= end), None)
                if matching_segment:
                    # Add labeled chunk as normal
                    chunk_index.append((video_uid, chunk_idx, matching_segment))
                else:
                    # Add gap chunk labeled as "transition of actions"
                    chunk_index.append((video_uid, chunk_idx, "transition of actions"))
        return chunk_index

    def load_video_chunk(self, video_uid, chunk_idx):
        chunk_path = os.path.join(self.video_root_dir, video_uid, f"{video_uid}_chunk_{chunk_idx}.pt")
        if not os.path.exists(chunk_path):
            raise FileNotFoundError(f"Chunk {chunk_idx} for video {video_uid} does not exist!")
        return torch.load(chunk_path, map_location=self.loading_device)

    def __len__(self):
        return len(self.chunk_index)

    def __getitem__(self, idx):
        try:
            video_uid, chunk_idx, step_description = self.chunk_index[idx]
            video_chunk = self.load_video_chunk(video_uid, chunk_idx)
            positive_pairs = [(video_chunk, step_description)]
            negative_pairs = []

            if step_description == "transition of actions":
                # Handle gaps: pair with random step description for negative samples
                random_step_description = random.choice(
                    [seg['step_description'] for vid in self.annotation_data.values() for seg in vid['segments']]
                )
                negative_pairs.append((video_chunk, random_step_description))
            else:
                # Handle labeled chunks as usual
                if idx % 2 == 0:
                    video_side_negative = True
                else:
                    video_side_negative = False
                begin = time()

                # Generate negative pairs with a timeout
                while len(negative_pairs) == 0 and time() - begin < 120:
                    random_video_uid = random.choice(self.available_videos)
                    try:
                        if random_video_uid == video_uid:
                            unique_segments = [segment for segment in self.annotation_data[random_video_uid]['segments']
                                                                if str(segment['step_description']).lower() != str(step_description).lower()]
                            if len(unique_segments) > 0:
                                random_segment = random.choice(unique_segments)
                                random_chunk_idx = random.randint(
                                    int(random_segment['start_time'] // 2), int(random_segment['end_time'] // 2)
                                )
                                random_video_chunk = self.load_video_chunk(random_video_uid, random_chunk_idx)
                                negative_pairs.append((random_video_chunk, step_description))
                        else:
                            random_segment = random.choice(self.annotation_data[random_video_uid]['segments'])
                            random_step_description = random_segment['step_description']
                            negative_pairs.append((video_chunk, random_step_description))
                    except FileNotFoundError:
                        continue

            return {
                'positive_pairs': positive_pairs,
                'negative_pairs': negative_pairs
            }
        except Exception as e:
            with open(f'/scratch/qt2087/DSGA1006_code/logs/V9_{self.batch_size}_{self.negative_sampling_ratio}.log', 'a') as f:
                f.write(f"Error loading batch {idx}: {str(e)}\n")
            return None

def custom_collate_fn(batch):
    if batch is None:
        return None
    positive_pairs = []
    negative_pairs = []

    for item in batch:
        if item is None or item['positive_pairs'] is None or item['negative_pairs'] is None:
            continue
        positive_pairs.extend(item['positive_pairs'])
        negative_pairs.extend(item['negative_pairs'])
    item_size = min(len(positive_pairs),len(negative_pairs))
    assert item_size > 0, "Data loading error!"
    return {
        'positive_pairs': positive_pairs[:item_size],
        'negative_pairs': negative_pairs[:item_size]
    }


def initialize_model_from_ckpt(ckpt, state_dict, n_frames):
    old_args = ckpt['args']
    print('=> creating model: {}'.format(old_args.model))
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
    )

    if 'TIMESFORMER' in old_args.model:
        print('=> inflating PE in models due to different frame numbers')
        state_dict = inflate_positional_embeds(
            model.state_dict(), state_dict,
            num_frames=n_frames,
            load_temporal_fix='bilinear',
        )

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# Define the InfoNCE contrastive loss function
def contrastive_loss(positive_pairs, negative_pairs, model, temperature=0.07, tokenizer = None, device = 'cuda'):
    positive_similarities = []
    negative_similarities = []

    # Compute similarities for positive pairs
    for video_chunk, step_description in positive_pairs:
        if len(video_chunk.shape) > 5:
            video_chunk = video_chunk.squeeze(0)
        elif len(video_chunk.shape) < 5:
            video_chunk = video_chunk.unsqueeze(0)
        video_chunk = video_chunk.to(device)
        # Use gradient checkpointing to save memory
        video_embedding = checkpoint.checkpoint(model.module.encode_image, video_chunk, use_reentrant=False)
        encoded_inputs = tokenizer(step_description, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoded_inputs['input_ids'].to(device)
        attention_mask = encoded_inputs['attention_mask'].to(device)
        step_embedding = model.module.encode_text(input_ids, attention_mask)

        del input_ids
        del attention_mask
        del encoded_inputs
        gc.collect()

        video_embedding = video_embedding.to(device)
        step_embedding = step_embedding.to(device)
        
        similarity = F.cosine_similarity(video_embedding, step_embedding, dim=-1)
        positive_similarities.append(similarity)

    # Compute similarities for negative pairs
    for video_chunk, step_description in negative_pairs:
        if len(video_chunk.shape) > 5:
            video_chunk = video_chunk.squeeze(0)
        elif len(video_chunk.shape) < 5:
            video_chunk = video_chunk.unsqueeze(0)
        video_chunk = video_chunk.to(device)
        # Use gradient checkpointing to save memory
        video_embedding = checkpoint.checkpoint(model.module.encode_image, video_chunk, use_reentrant=False)
        encoded_inputs = tokenizer(step_description, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoded_inputs['input_ids'].to(device)
        attention_mask = encoded_inputs['attention_mask'].to(device)
        step_embedding = model.module.encode_text(input_ids, attention_mask)

        del input_ids
        del attention_mask
        del encoded_inputs
        gc.collect()

        video_embedding = video_embedding.to(device)
        step_embedding = step_embedding.to(device)
        
        similarity = F.cosine_similarity(video_embedding, step_embedding, dim=-1)
        negative_similarities.append(similarity)
    

    # Stack the similarities
    positive_similarities = torch.stack(positive_similarities).to(device)
    negative_similarities = torch.stack(negative_similarities).to(device)
    # Calculate InfoNCE loss
    positives = positive_similarities.to(device)
    # print(positive_similarities.shape, negative_similarities.shape)
    negatives = torch.cat([positive_similarities, negative_similarities], dim=1).to(device)
    loss =  torch.log(torch.exp(positives / temperature) / torch.sum(torch.exp(negatives / temperature), dim=1))
    loss = -1 * loss.mean()

     # Calculate InfoNCE loss
    # loss = 0
    # for i in range(len(positive_similarities)):
    #     positive_score = positive_similarities[i]  # Shape: (1,)
    #     all_scores = torch.cat([positive_similarities[i].unsqueeze(0), negative_similarities], dim=0)  # Shape: (2n + 1, 1)
    #     loss += torch.log(torch.exp(positive_score / temperature) / torch.sum(torch.exp(all_scores / temperature)))
    # loss = -1 * loss / len(positive_similarities)

    # all_similarities = torch.cat([positive_similarities, negative_similarities.view(1, -1).expand(N, -1)], dim=1)  # Shape: (N, k + 1)
    # # Calculate the exponential of all similarities divided by temperature
    # exp_all_similarities = torch.exp(all_similarities / temperature)  # Shape: (N, k + 1)
    # # Calculate the InfoNCE loss
    # positive_exp_scores = exp_all_similarities[:, 0]  # Shape: (N,)
    # loss = -torch.log(positive_exp_scores / torch.sum(exp_all_similarities, dim=1))  # Shape: (N,)
    # # Take the mean of the loss
    # loss = loss.mean()

    return loss

# @torch.no_grad()
def evaluate(model, video_root_dir, annotation_file, tokenizer, loading_device = 'cuda', for_val = 0.01, sampler = None):

    val_dataset = VideoTextAlignmentDataset(
        video_root_dir = video_root_dir,
        annotation_file = annotation_file,
        # device='cpu',  # Use 'cuda' if your GPU memory can handle the validation batch
        loading_device = loading_device,
        negative_sampling_ratio = 1,  # Same sampling ratio as the training data
        for_val = for_val
    )

    pin_mem = loading_device == 'cpu'

    val_loader = DataLoader(
        val_dataset,
        batch_size=128,  # Adjust based on available GPU memory
        shuffle=False,
        num_workers=2,
        pin_memory=pin_mem,
        sampler=sampler,  # Use DistributedSampler here
        collate_fn=custom_collate_fn,
    )

    model.eval()
    
    val_loss = []

    with torch.no_grad():  # Disable gradient calculations for validation
        for batch in val_loader:
            positive_pairs = batch['positive_pairs']
            negative_pairs = batch['negative_pairs']
            with torch.no_grad():
                loss = contrastive_loss(positive_pairs, negative_pairs, model, tokenizer = tokenizer)
            
            val_loss.append(loss.item())

    return np.mean(val_loss)