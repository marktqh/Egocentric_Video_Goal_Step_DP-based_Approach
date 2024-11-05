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
import lavila_adapters
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

import pathlib

from transformers import DistilBertTokenizer
from torch.amp import GradScaler, autocast
from time import time

# set cuda device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class VideoTextAlignmentDataset(Dataset):
    def __init__(self, video_root_dir, annotation_file, device='cpu', negative_sampling_ratio=1):
        self.video_root_dir = video_root_dir
        self.device = device
        self.annotation_data = self.load_annotations(annotation_file)
        self.available_videos = self.get_available_videos()
        self.negative_sampling_ratio = negative_sampling_ratio
        self.segment_index = self.create_segment_index()
    
    def load_annotations(self, annotation_file):
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        f.close()
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
        return available_videos

    def create_segment_index(self):
        segment_index = []
        for video_uid in self.available_videos:
            segments = self.annotation_data[video_uid]['segments']
            for segment_idx, segment in enumerate(segments):
                segment_index.append((video_uid, segment_idx))
        return segment_index

    def load_video_chunk(self, video_uid, chunk_idx):
        chunk_path = os.path.join(self.video_root_dir, video_uid, f"{video_uid}_chunk_{chunk_idx}.pt")
        if not os.path.exists(chunk_path):
            raise FileNotFoundError(f"Chunk {chunk_idx} for video {video_uid} does not exist!")
        return torch.load(chunk_path, map_location=self.device)

    def __len__(self):
        return len(self.segment_index)

    def __getitem__(self, idx):
        try:
            # Get video_uid and segment index
            video_uid, segment_idx = self.segment_index[idx]
            video_annotations = self.annotation_data[video_uid]
            segment = video_annotations['segments'][segment_idx]

            start_time, end_time = segment['start_time'], segment['end_time']
            step_description = segment['step_description']

            # Find video chunks that correspond to the given step
            start_chunk = int(start_time // 2)
            end_chunk = int(end_time // 2)
            video_chunks = []

            for chunk_idx in range(start_chunk, end_chunk + 1):
                try:
                    video_chunk = self.load_video_chunk(video_uid, chunk_idx)
                    video_chunks.append(video_chunk)
                except FileNotFoundError as e:
                    print(e)
                    continue

            # Create positive pairs
            positive_pairs = [(video_chunk, step_description) for video_chunk in video_chunks]

            # Generate negative pairs
            video_side_negative = int(len(positive_pairs) / 2)
            video_side_negative_count = 0
            text_side_negative = len(positive_pairs) - video_side_negative
            text_side_negative_count = 0

            negative_pairs = []
            available_videos_shuffled = random.sample(self.available_videos, len(self.available_videos))
            max_attempts = 100  # Limit the number of attempts to avoid infinite loops
            attempt_count = 0

            while video_side_negative_count + text_side_negative_count < self.negative_sampling_ratio * len(positive_pairs) and attempt_count < max_attempts:
                attempt_count += 1
                random_video_uid = random.choice(available_videos_shuffled)

                # Generate text-side negative pairs
                if text_side_negative_count < text_side_negative:
                    if random_video_uid == video_uid:
                        if len(self.annotation_data[random_video_uid]['segments']) > 1:
                            random_segment = random.choice([segment for segment in self.annotation_data[random_video_uid]['segments']
                                                            if segment['step_description'] != step_description])
                            random_step_description = random_segment['step_description']
                            if len(video_chunks) > 0:
                                random_chunk_idx = random.randint(0, len(video_chunks) - 1)
                                negative_pairs.append((video_chunks[random_chunk_idx], random_step_description))
                                text_side_negative_count += 1
                    else:
                        random_segment = random.choice(self.annotation_data[random_video_uid]['segments'])
                        random_step_description = random_segment['step_description']
                        if len(video_chunks) > 0:
                            random_chunk_idx = random.randint(0, len(video_chunks) - 1)
                            negative_pairs.append((video_chunks[random_chunk_idx], random_step_description))
                            text_side_negative_count += 1

                # Generate video-side negative pairs
                if video_side_negative_count < video_side_negative:
                    random_video_uid = random.choice(available_videos_shuffled)
                    if random_video_uid == video_uid:
                        if len(self.annotation_data[random_video_uid]['segments']) > 1:
                            random_segment = random.choice([segment for segment in self.annotation_data[random_video_uid]['segments']
                                                            if segment['step_description'] != step_description])
                            start_time, end_time = random_segment['start_time'], random_segment['end_time']
                            start_chunk = int(start_time // 2)
                            end_chunk = int(end_time // 2)
                            random_chunk_idx = random.randint(start_chunk, end_chunk)
                            try:
                                random_video_chunk = self.load_video_chunk(random_video_uid, random_chunk_idx)
                                negative_pairs.append((random_video_chunk, step_description))
                                video_side_negative_count += 1
                            except FileNotFoundError:
                                continue
                    else:
                        random_chunk_idx = random.randint(0, len(os.listdir(os.path.join(self.video_root_dir, random_video_uid))) - 1)
                        try:
                            random_video_chunk = self.load_video_chunk(random_video_uid, random_chunk_idx)
                            negative_pairs.append((random_video_chunk, step_description))
                            video_side_negative_count += 1
                        except FileNotFoundError:
                            continue
        except Exception as e:
            print(str(e))
            return None
        return {
            'positive_pairs': positive_pairs,
            'negative_pairs': negative_pairs
        }

def custom_collate_fn(batch):
    if batch is None:
        return None
    positive_pairs = []
    negative_pairs = []

    for item in batch:
        positive_pairs.extend(item['positive_pairs'])
        negative_pairs.extend(item['negative_pairs'])

    # # Pad the video chunks so that all elements in the batch have the same size
    # video_chunks = [pair[0] for pair in positive_pairs]
    # max_chunk_size = max([chunk.size(0) for chunk in video_chunks])
    # padded_video_chunks = [F.pad(chunk, (0, 0, 0, 0, 0, max_chunk_size - chunk.size(0))) for chunk in video_chunks]

    # positive_pairs = [(padded_chunk, pair[1]) for padded_chunk, pair in zip(padded_video_chunks, positive_pairs)]

    return {
        'positive_pairs': positive_pairs,
        'negative_pairs': negative_pairs
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
# def contrastive_loss(positive_pairs, negative_pairs, model, temperature=0.1, tokenizer=None):
#     video_chunks, step_descriptions = [], []

#     batch_size = len(positive_pairs)
#     positive_indices = torch.arange(batch_size)

#     # Gather embeddings for all pairs
#     for video_chunk, step_description in positive_pairs:
#         if len(video_chunk.shape) > 5:
#             video_chunk = video_chunk.squeeze(0)
#         elif len(video_chunk.shape) < 5:
#             video_chunk = video_chunk.unsqueeze(0)
#         video_chunks.append(checkpoint.checkpoint(model.encode_image, video_chunk, use_reentrant=False))  # Encode video chunk
#         # video_chunks.append(model.encode_image(video_chunk))  # Encode video chunk
#         encoded_inputs = tokenizer(step_description, return_tensors='pt', padding=True, truncation=True)
#         input_ids = encoded_inputs['input_ids'].to(device)
#         attention_mask = encoded_inputs['attention_mask'].to(device)
#         step_descriptions.append(model.encode_text(input_ids, attention_mask))  # Encode step description
#     del input_ids
#     del attention_mask
#     del encoded_inputs
#     gc.collect()


#     for video_chunk, step_description in negative_pairs:
#         if len(video_chunk.shape) > 5:
#             video_chunk = video_chunk.squeeze(0)
#         elif len(video_chunk.shape) < 5:
#             video_chunk = video_chunk.unsqueeze(0)
#         video_chunks.append(checkpoint.checkpoint(model.encode_image, video_chunk, use_reentrant=False))  # Encode video chunk
#         # video_chunks.append(model.encode_image(video_chunk))  # Encode video chunk
#         encoded_inputs = tokenizer(step_description, return_tensors='pt', padding=True, truncation=True)
#         input_ids = encoded_inputs['input_ids'].to(device)
#         attention_mask = encoded_inputs['attention_mask'].to(device)
#         step_descriptions.append(model.encode_text(input_ids, attention_mask))  # Encode step description
#     del input_ids
#     del attention_mask
#     del encoded_inputs
#     gc.collect()

#     # Stack the embeddings
#     video_chunks = torch.stack(video_chunks).to(device)  # Shape: (batch_size, embedding_dim)
#     step_descriptions = torch.stack(step_descriptions).to(device)  # Shape: (batch_size, embedding_dim)

#     # Compute similarities
#     similarity_matrix = F.cosine_similarity(video_chunks.unsqueeze(1), step_descriptions.unsqueeze(0), dim=2)

#     # Positive pairs are on the diagonal
#     positives = similarity_matrix[positive_indices, positive_indices]

#     # Calculate InfoNCE loss
#     loss = -torch.log(torch.exp(positives / temperature) / torch.sum(torch.exp(similarity_matrix / temperature), dim=1))
#     loss = loss.mean()

#     return loss

# Define the InfoNCE contrastive loss function
def contrastive_loss(positive_pairs, negative_pairs, model, temperature=0.1, tokenizer = None):
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
        video_embedding = checkpoint.checkpoint(model.encode_image, video_chunk, use_reentrant=False)
        encoded_inputs = tokenizer(step_description, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoded_inputs['input_ids'].to(device)
        attention_mask = encoded_inputs['attention_mask'].to(device)
        step_embedding = model.encode_text(input_ids, attention_mask)

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
        video_embedding = checkpoint.checkpoint(model.encode_image, video_chunk, use_reentrant=False)
        encoded_inputs = tokenizer(step_description, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoded_inputs['input_ids'].to(device)
        attention_mask = encoded_inputs['attention_mask'].to(device)
        step_embedding = model.encode_text(input_ids, attention_mask)

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
    print(positive_similarities.shape, negative_similarities.shape)
    negatives = torch.cat([positive_similarities, negative_similarities], dim=1).to(device)
    loss = -torch.log(torch.exp(positives / temperature) / torch.sum(torch.exp(negatives / temperature), dim=1))
    loss = loss.mean()

     # Calculate InfoNCE loss
    # loss = 0
    # for i in range(len(positive_similarities)):
    #     positive_score = positive_similarities[i]  # Shape: (1,)
    #     all_scores = torch.cat([positive_similarities[i].unsqueeze(0), negative_similarities], dim=0)  # Shape: (2n + 1, 1)
    #     loss += -torch.log(torch.exp(positive_score / temperature) / torch.sum(torch.exp(all_scores / temperature)))
    # loss = loss / len(positive_similarities)

    return loss

def main(batch_size=8):
    # video_root_dir = r'.\processed_data\kevin_code\goalstep_val'  # Path to video embeddings (subfolders per video with .pt files)
    # annotation_file = r'.\ego4d_data\V4\annotations\goalstep_val.json'  # Path to annotation file

    video_root_dir = r'/scratch/qt2087/DSGA1006_code/processed_data/mark_code/goalstep_val'  # Path to video embeddings (subfolders per video with .pt files)
    annotation_file = r'/scratch/qt2087/DSGA1006_code/annotations/goalstep_val.json'  # Path to annotation file

    temp = pathlib.PosixPath
    # pathlib.PosixPath = pathlib.WindowsPath
    # ckpt_path = r'./lavila/downloaded_models/clip_openai_timesformer_large_336px_distilbert_base.pth'
    ckpt_path = r'/scratch/qt2087/DSGA1006_code/lavila/downloaded_models/clip_openai_timesformer_large_336px_distilbert_base.pth'
    ckpt, state_dict = load_ckpt_model_weight(ckpt_path)
    # pathlib.PosixPath = temp
    

    segment_duration = 2

    num_frames = 4
    model = initialize_model_from_ckpt(ckpt, state_dict, num_frames)

    model.to(device)

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
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=1e-4)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    num_epochs = 2

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased',force_download=False)

    # Create dataset and dataloader
    dataset = VideoTextAlignmentDataset(
        video_root_dir=video_root_dir,
        annotation_file=annotation_file,
        device='cpu',
        negative_sampling_ratio=1,  # Adjust as needed
    )
    data_loader = DataLoader(dataset, 
                             batch_size=batch_size, 
                             shuffle=False, 
                             num_workers=4, 
                             pin_memory=True, 
                             collate_fn=custom_collate_fn  # Use the custom collate function here
                             )
    total_batches = math.ceil(len(dataset) / batch_size)
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Number of bathces: {total_batches}")
    with open(f'/scratch/qt2087/DSGA1006_code/logs/V4_{batch_size}_{dataset.negative_sampling_ratio}.log', 'a') as f:
        f.write(f"Dataset size: {len(dataset)}; Batch size: {batch_size}.\n")
    f.close()
    print("Starting training...")

    writer = SummaryWriter(f'/scratch/qt2087/DSGA1006_code/logs/V4')
    scaler = GradScaler()
    prev_epoch_time = time()
    batch_count = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        counter = 0
        prev_batch_time = time()
        # for batch in dataset:
        for batch_idx, batch in enumerate(data_loader):
            if batch is None:
                print(f'Error loading batch {batch_idx}. Skip')
                with open(f'/scratch/qt2087/DSGA1006_code/logs/V4_{batch_size}_{dataset.negative_sampling_ratio}.log', 'a') as f:
                    f.write(f"Error loading batch {batch_idx}. Skip\n")
                f.close()
                if (batch_idx+1) % 10 == 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.9
                prev_batch_time = time()
                continue
            positive_pairs = batch['positive_pairs']
            negative_pairs = batch['negative_pairs']
            # print(len(positive_pairs), len(negative_pairs))
            # print(f"Batch {batch_idx+1} loaded...")
            # raise Exception("Stop for debugging")
            # Compute contrastive loss (InfoNCE)
            # with autocast(device_type=device.type, dtype=torch.float16):
            loss = contrastive_loss(positive_pairs, negative_pairs, model, tokenizer = tokenizer)
            # print(f"Loss computed...")

            # Backpropagation and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            del positive_pairs, negative_pairs
            torch.cuda.empty_cache()
            gc.collect()

            total_loss += loss.item()
            counter += 1
            avg_loss = total_loss / counter
            with open(f'/scratch/qt2087/DSGA1006_code/logs/V4_{batch_size}_{dataset.negative_sampling_ratio}.log', 'a') as f:
                f.write(f"Epoch {epoch+1}, Batch {(batch_idx+1)}/{total_batches}, Loss: {loss.item()}, Avg Loss: {avg_loss}, Time taken: {time() - prev_batch_time}\n")
            f.close()
            writer.add_scalar(f'Training Loss - {(batch_idx+1)}_{dataset.negative_sampling_ratio}', loss.item(), (batch_idx+1) + batch_count)
            if (batch_idx+1) == 1:
                print(f"Epoch {epoch+1}, Batch {(batch_idx+1)}/{total_batches}, Loss: {loss.item()}, Avg Loss: {avg_loss}, Time taken: {time() - prev_batch_time}")
                # also log the message to a file
            if (batch_idx+1) % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {(batch_idx+1)}/{total_batches}, Loss: {loss.item()}, Avg Loss: {avg_loss}, Time taken: {time() - prev_batch_time}")
            if (batch_idx+1) % 10 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.9
            prev_batch_time = time()
                

        avg_loss = total_loss / counter
        print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss}, Time taken: {time() - prev_epoch_time}")
        with open(f'/scratch/qt2087/DSGA1006_code/logs/V4_{batch_size}_{dataset.negative_sampling_ratio}.log', 'a') as f:
            f.write(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss}, Time taken: {time() - prev_epoch_time}\n")
        f.close()
        prev_epoch_time = time()
        print("Saving model...")
        model_dir = f'/scratch/qt2087/DSGA1006_code/lavila_adapter_models/{batch_size}_{dataset.negative_sampling_ratio}'
        os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(model_dir, f'video_text_alignment_model_V4_{epoch}.pth'))
        batch_count = batch_idx
    
    # save the model
    print("Saving model...")
    model_dir = f'/scratch/qt2087/DSGA1006_code/lavila_adapter_models/{batch_size}_{dataset.negative_sampling_ratio}'
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, f'video_text_alignment_model_V4.pth'))

if __name__ == '__main__':
    # call the main function with batch size same as input in the terminal
    main(int(sys.argv[1]))

    # @TODO: update V5: instead of loading one step as a batch, load one chunk as a batch