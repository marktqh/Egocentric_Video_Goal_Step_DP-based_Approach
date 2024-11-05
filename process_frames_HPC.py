#!/bin/env python

import torchvision
import torch
import os
import sys
sys.path.insert(0, '/scratch/qt2087/DSGA1006_code/lavila')
from lavila.lavila.data.video_transforms import Permute
from collections import OrderedDict
from lavila.lavila.models import models
from lavila.lavila.models.utils import inflate_positional_embeds
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
from datetime import datetime

import json
import pathlib
import fire
from glob import glob

def load_ckpt_model_weight(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v
    return ckpt, state_dict

def split_videos_by_duration(videos: list[tuple], T):

    groups = []
    current_group = []
    current_total = 0

    for video_id, split, duration in videos:
        if duration > T:
            if current_group:
                groups.append(current_group)
                current_group = []
                current_total = 0
            groups.append([(video_id, split)])
        else:
            if current_total + duration <= T:
                current_group.append((video_id, split))
                current_total += duration
            else:
                groups.append(current_group)
                current_group = [(video_id, split)]
                current_total = duration

    if current_group:
        groups.append(current_group)

    return groups


def transform_input(input, args):
    crop_size = 224 if '336PX' not in args.model else 336
    val_transform = transforms.Compose([
        Permute([1, 0, 2, 3]), 
        transforms.Resize(crop_size),
        transforms.CenterCrop(crop_size),
        (transforms_video.NormalizeVideo(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]) if ('OPENAI' not in args.model) else
            transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305])),
    ])
    transformed_frames = val_transform(input) # [C, T, H, W]
    output = transformed_frames.unsqueeze(0)
    return output

def save_video_features_in_chunks(video_path, split, num_samples, clip_length, saving_threshold, output_dir, device, args):
    video_id = os.path.basename(video_path).split('.')[0]
    output_path = os.path.join(output_dir, split, video_id)
    os.makedirs(output_path, exist_ok = True)
    vobject = torchvision.io.VideoReader(video_path)
    vobject.set_current_stream('video')
    fps = 30
    frames_per_clip = int(fps * clip_length)  
    frame_chunk = []
    # all_features = []
    count = 0
    save_count = 0
    accumulated_time = 0
    chunk_index = 0
    for frame in vobject:
        frame_chunk.append(frame['data'])
        count += 1
        accumulated_time += 1 / fps
        if count == frames_per_clip:
            if len(frame_chunk) >= num_samples:
                indices = torch.linspace(0, len(frame_chunk) - 1, num_samples).long() 
                sampled_frames = [frame_chunk[i] for i in indices]  
                frames_tensor = torch.stack(sampled_frames).float()
            else:
                frames_tensor = torch.stack(frame_chunk).float()  
            frames_tensor = transform_input(frames_tensor, args).to(device)

            save_path = os.path.join(output_path, f"{video_id}_chunk_{chunk_index}.pt")
            torch.save(frames_tensor, save_path)
            
            # all_features.append(frames_tensor)
            frame_chunk = []
            count = 0
            chunk_index += 1

        # if accumulated_time >= saving_threshold:
        #     save_path = os.path.join(output_path, f"{video_id}_part_{save_count}.pt")
        #     torch.save(torch.cat(all_features, dim=0), save_path)
        #     all_features.clear()  
        #     save_count += 1
        #     accumulated_time = 0 

    if frame_chunk:
        if len(frame_chunk) >= num_samples:
            indices = torch.linspace(0, len(frame_chunk) - 1, num_samples).long()
            sampled_frames = [frame_chunk[i] for i in indices]
            frames_tensor = torch.stack(sampled_frames).float()
        else:
            frames_tensor = torch.stack(frame_chunk).float()
        frames_tensor = transform_input(frames_tensor, args).to(device)
        # check for dimension 2 and make sure it matches num_samples
        # if not, pad with last frame
        if frames_tensor.shape[2] < num_samples:
            # print(frames_tensor.shape)
            # pad = frames_tensor.unsqueeze(2).repeat(1, 1, num_samples - frames_tensor.shape[2], 1, 1).to(device)
            # frames_tensor = torch.cat([frames_tensor, pad], dim=2)
            num_padding_frames = num_samples - frames_tensor.shape[2]
            last_frame = frames_tensor[:, :, -1, :, :].unsqueeze(2)  
            padding = last_frame.repeat(1, 1, num_padding_frames, 1, 1) 
            # Concatenate the padding to the original frames_tensor along dimension 2 (frame dimension)
            frames_tensor = torch.cat([frames_tensor, padding], dim=2)  # Shape becomes [1, 3, 16, 336, 336]

        save_path = os.path.join(output_path, f"{video_id}_chunk_{chunk_index}.pt")
        torch.save(frames_tensor, save_path)
    
def save_video_features(video_paths: list[tuple[str, str]], num_samples, clip_length, saving_threthhold, output_dir, device, args):
    for video_path, split in video_paths:
        if os.path.exists(video_path):
            print(f"Processing video {video_path}")
            # check if the video is already processed by checking if the output directory exists
            # and check if the output directory is empty
            video_id = os.path.basename(video_path).split('.')[0]
            output_path = os.path.join(output_dir, split, video_id)
            # if os.path.exists(check_path):
            #     print(f'Video {video_id} already processed. Skip!')
            #     continue
            # if video_id in processed_videos:
            #     print(f"Video {video_path} already processed.")
            #     continue
            if os.path.exists(output_path) and len(os.listdir(output_path)) > 0:
                print(f"Video {video_path} already processed.")
                continue

            start = datetime.now()
            try:
                save_video_features_in_chunks(video_path, split, num_samples, clip_length, saving_threthhold, output_dir, 'cuda', args)
            except RuntimeError as e:
                print(e)
                print(f"Video {video_path} failed to process using cuda. Trying cpu.")
                try:
                    save_video_features_in_chunks(video_path, split, num_samples, clip_length, saving_threthhold, output_dir, 'cpu', args)
                except Exception as e:
                    print(e)
                    print(f"Video {video_path} failed to process.")
                    continue
            # save_video_features_in_chunks(video_path, split, num_samples, clip_length, saving_threthhold, output_dir, device, args)
            end = datetime.now()
            print(f'Finished, used {end - start}.')
        else:
            print(f"Video path {video_path} does not exist.")

# main
def main(array_index):
    videos_dir  = '/scratch/work/public/ml-datasets/ego4d/v2/v2/full_scale'
    segment_duration = 2
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"use: {device}")

    ckpt_path = './lavila/downloaded_models/clip_openai_timesformer_large_336px_distilbert_base.pth'
    # temp = pathlib.PosixPath
    # pathlib.PosixPath = pathlib.WindowsPath
    ckpt, state_dict = load_ckpt_model_weight(ckpt_path)
    # pathlib.PosixPath = temp

    num_frames = 4

    output_dir = f'/scratch/qt2087/DSGA1006_code/processed_data/mark_code'
    os.makedirs(output_dir, exist_ok = True)


    clip_ids_duration = []
    # json_path = '/scratch/qt2087/DSGA1006_code/goalstep_val.json'
    json_base_dir = '/scratch/qt2087/DSGA1006_code/annotations'
    json_files = glob(os.path.join(json_base_dir, '*.json'))
    for json_path in json_files:
        split = os.path.basename(json_path).split('.')[0]
        with open(json_path, 'r') as file:
            data = json.load(file)
        videos = data['videos']
        for i in videos:
            uid = i['video_uid']
            video_path = os.path.join(videos_dir, f'{uid}.mp4')
            try:
                start = i['start_time']
                end = i['end_time'] 
            except Exception:
                start = 0
                end = 0
            clip_ids_duration.append([video_path, split, end - start])
    sorted_list = sorted(clip_ids_duration, key=lambda x: x[-1], reverse=False)

    splited_videos = split_videos_by_duration(sorted_list, 3600) # 423 before removing non exist videos, 316 after 
    print(array_index, len(splited_videos))
    # counter = 0
    save_video_features(splited_videos[array_index], num_frames, segment_duration, 3600, output_dir, 'cuda', ckpt['args'])
    # for splited_video in splited_videos:
    #     save_video_features(splited_video, num_frames, segment_duration, 3600, output_dir, 'cuda', ckpt['args'])
    #     counter += 1
    #     print(f"Processed {counter}/{len(splited_videos)} of splitted chunks.")
    # # try use cuda first
    # # if cuda memory is not enough, use cpu
    # try:
    #     save_video_features(splited_videos[0], num_frames, segment_duration, 3600, output_dir, 'cuda', ckpt['args'])
    # except RuntimeError as e:
    #     print(e)
    #     save_video_features(splited_videos[0], num_frames, segment_duration, 3600, output_dir, 'cpu', ckpt['args'])
        
if __name__ == '__main__':
    fire.Fire(main)