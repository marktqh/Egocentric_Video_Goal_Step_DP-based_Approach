import torchvision
import torch
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lavila')))
from lavila.lavila.data.video_transforms import Permute
from collections import OrderedDict
from lavila.lavila.models import models
from lavila.lavila.models.utils import inflate_positional_embeds
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
from datetime import datetime



def load_ckpt_model_weight(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v
    return ckpt, state_dict

def initialize_model_from_ckpt(ckpt, state_dict, n_frames):
    old_args = ckpt['args']
    print('=> creating model: {}'.format(old_args.model))
    model = getattr(models, old_args.model)(
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

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

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

def save_video_features_in_chunks(video_path, split, num_samples, clip_length, saving_threshold, output_dir, model, device, args):
    video_id = os.path.basename(video_path).split('.')[0]
    output_path = os.path.join(output_dir, split, video_id)
    os.makedirs(output_path, exist_ok = True)
    vobject = torchvision.io.VideoReader(video_path)
    vobject.set_current_stream('video')
    fps = 30
    frames_per_clip = int(fps * clip_length)  
    frame_chunk = []
    all_features = []
    count = 0
    save_count = 0
    accumulated_time = 0
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
            with torch.no_grad():
                video_embedding = model.encode_image(frames_tensor)
            
            all_features.append(video_embedding)
            frame_chunk = []
            count = 0

        if accumulated_time >= saving_threshold:
            save_path = os.path.join(output_path, f"{video_id}_part_{save_count}.pt")
            torch.save(torch.cat(all_features, dim=0), save_path)
            all_features.clear()  
            save_count += 1
            accumulated_time = 0 

    if frame_chunk:
        if len(frame_chunk) >= num_samples:
            indices = torch.linspace(0, len(frame_chunk) - 1, num_samples).long()
            sampled_frames = [frame_chunk[i] for i in indices]
            frames_tensor = torch.stack(sampled_frames).float()
        else:
            frames_tensor = torch.stack(frame_chunk).float()
        frames_tensor = transform_input(frames_tensor, args).to(device)
        with torch.no_grad():
            video_embedding = model.encode_image(frames_tensor)
        all_features.append(video_embedding)

    if all_features:
        save_path = os.path.join(output_path, f"{video_id}_part_{save_count}.pt")
        torch.save(torch.cat(all_features, dim=0), save_path)
    





def save_video_features(video_paths: list[tuple[str, str]], num_samples, clip_length, saving_threthhold, output_dir, model, device, args):
    for video_path, split in video_paths:
        if os.path.exists(video_path):
            print(f"Processing video {video_path}")
            start = datetime.now()
            save_video_features_in_chunks(video_path, split, num_samples, clip_length, saving_threthhold, output_dir, model, device, args)
            end = datetime.now()
            print(f'Finished, used {end - start}')
        else:
            print(f"Video path {video_path} does not exist.")