import torch
import urllib
import os
import torchvision
from collections import OrderedDict, defaultdict
from lavila.lavila.models import models
from lavila.lavila.models.utils import inflate_positional_embeds
from lavila.lavila.data.video_transforms import Permute
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
import torch.nn.functional as F

def load_ckpt_model_weight(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v
    return ckpt, state_dict


# def get_segments_with_frame_duration(labels, num_frames_per_segment, time_per_frame, num_total_frames):
#     segments = []
#     start_idx = 0  
#     current_label = labels[0]

#     for i in range(1, len(labels)):
#         if labels[i] != current_label:
#             start_time = start_idx * num_frames_per_segment * time_per_frame
#             end_time = i * num_frames_per_segment * time_per_frame
#             segments.append((current_label, start_time, end_time))

#             start_idx = i
#             current_label = labels[i]
    

#     start_time = start_idx * num_frames_per_segment * time_per_frame
#     end_time = num_total_frames * time_per_frame
#     segments.append((current_label, start_time, end_time))

#     return segments





def get_start_end_description(video_dict: dict, gap_label = None) -> list[tuple]:
    segments = video_dict['segments']
    labeled_intervals = [(step['start_time'], step['end_time'], step['step_description']) for step in segments]
    if gap_label is not None:
        labeled_intervals.append((0,0, gap_label))
    return labeled_intervals


def download_and_load_model_weight(download_dir, url):
    model_name = url.split('/')[-1].split('.')[0]
    ckpt_path = os.path.join(download_dir, f'{model_name}.pth')
    if not os.path.exists(ckpt_path):
        try:
            urllib.request.urlretrieve(url, ckpt_path)
            print("Download completed successfully.")
        except Exception as e:
            print(f"Failed to download the file. Error: {e}")
            return None, None
    else:
        print("Model already exists. Skipping download.")

    return ckpt_path




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


def transform_input(ckpt, input):
    old_args = ckpt['args']
    crop_size = 224 if '336PX' not in old_args.model else 336
    val_transform = transforms.Compose([
        Permute([1, 0, 2, 3]), 
        transforms.Resize(crop_size),
        transforms.CenterCrop(crop_size),
        (transforms_video.NormalizeVideo(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]) if ('OPENAI' not in old_args.model) else
            transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305])),
    ])
    transformed_frames = val_transform(input)
    output = transformed_frames.unsqueeze(0)
    return output



def read_frames_between_intervals(video_dict, video_path):
    video_object = torchvision.io.VideoReader(video_path)
    video_object.set_current_stream("video")
    segments = video_dict['segments']
    labeled_intervals = [(step['start_time'], step['end_time'], step['step_description']) for step in segments]
    interval_frames = []
    frame_iterator = iter(video_object)  
    current_frame = next(frame_iterator, None)
    for start_time, end_time, _ in labeled_intervals:
        frames_in_interval = []
        
        while current_frame is not None:
            timestamp = current_frame['pts']
            if start_time <= timestamp <= end_time:
                frames_in_interval.append(current_frame['data'].to(torch.float32))
            if timestamp > end_time:
                break
            current_frame = next(frame_iterator, None)

        interval_frames.append(torch.stack(frames_in_interval))

    return labeled_intervals, interval_frames




def calculate_iou(pred_window, gt_window):
    pred_start, pred_end = pred_window
    gt_start, gt_end = gt_window


    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    intersection = max(0, intersection_end - intersection_start)

    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection

    iou = intersection / union if union > 0 else 0
    return iou


# def evaluate_r_at_1(predictions, ground_truths, iou_threshold):
#     correct_count = 0
#     total_count = len(ground_truths)

#     for pred_window, gt_window in zip(predictions, ground_truths):
#         iou = calculate_iou(pred_window, gt_window)
#         if iou >= iou_threshold:
#             correct_count += 1

#     return correct_count / total_count


# def evaluate_r_at_1(predictions:list[tuple[float, float, str]], ground_truth:list[tuple[float, float, str]], iou_threshold):
#     total_count = len(ground_truth)
#     correct = 0
#     for idx, (start, end, des) in enumerate(ground_truth):
#         if start is not None:
#             best_iou = 0  
#             for p_start, p_end, pred_idx in predictions:
#                 if idx == pred_idx:
#                     iou = calculate_iou((p_start, p_end), (start, end))
#                     best_iou = max(best_iou, iou)
#             if best_iou >= iou_threshold:
#                 correct += 1
        
#         else:
#             total_count -= 1


#     recall = correct / total_count if ground_truth else 0
#     return recall


# def count_truth_label(truths: list[tuple]):


def group_predictions_per_label(predictions:list[tuple[float, float, int, tuple[int, int]]]):
    grouped = defaultdict(list)
    for i in predictions:
        key = i[2]
        grouped[key].append(i)
    return dict(grouped)

def select_best_pred_per_lable(predictions: list[tuple], sim_scores):
    highest_mean = -float('inf')  
    best_index = -1
    for idx, (_, _, _, (start, end)) in enumerate(predictions):
        mean_score = torch.mean(sim_scores[start:end])
        if mean_score > highest_mean:
            highest_mean = mean_score
            best_index = idx
    
    return predictions[best_index]



def evaluate_r_at_1(predictions:list[tuple[float, float, int,  tuple[int, int]]], ground_truth:list[tuple[float, float, str]], sim_scores, iou_threshold):
    total_count = len(ground_truth)
    correct = 0
    predictions_dict = group_predictions_per_label(predictions)
    for idx, (start, end, des) in enumerate(ground_truth):
        if start is not None:
            prediction_list = predictions_dict.get(idx, None)
            if prediction_list is not None:
                pred_start, pred_end, _, (_, _)  = select_best_pred_per_lable(prediction_list, sim_scores)
                iou = calculate_iou((pred_start, pred_end), (start, end))
                if iou >= iou_threshold:
                    correct += 1
            else:
                total_count -= 1
        
        else:
            total_count -= 1


    recall = correct / total_count if ground_truth else 0
    return recall





def encode_images(model, device, num_frames, input):
    embeddings = []
    for i in range(0, input.size(2), num_frames):
        frame_chunk = input[:, :, i:i + num_frames, :, :].to(device)
        with torch.no_grad():
            video_embedding = model.encode_image(frame_chunk)
            embeddings.append(video_embedding)
    video_embeddings = torch.cat(embeddings, dim=0).to(device)
    return video_embeddings
    

def encode_texts(model, tokenizer, device, text_labels:[list[tuple]]):
    text_embeddings = []
    for _, _, description in text_labels:
        texts = tokenizer(description)
        if isinstance(texts, tuple):
            texts, masks = texts
        texts = texts.view(-1, 77).contiguous().to(device)
        masks = masks.view(-1, 77).contiguous().to(device) if masks is not None else None
        
        with torch.no_grad():
            if masks is not None:
                class_embeddings = model.encode_text(texts, attention_mask=masks)
            else:
                class_embeddings = model.encode_text(texts)
            text_embeddings.append(class_embeddings)
    text_embeddings = torch.cat(text_embeddings, dim=0).to(device) 
    return text_embeddings


def get_segments_with_frame_duration(labels, segment_duration, total_time):
    segments = []
    current_index = 0
    start_time = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[current_index]:
            n = i - current_index
            end_time = start_time + segment_duration * n
            segments.append((start_time, end_time, labels[current_index].item(), (current_index, i)))
            current_index = i
            start_time = end_time  
    segments.append((start_time, total_time, labels[current_index].item(),  (current_index, len(labels))))

    return segments


def create_predictions_truths(labels, start_end_label, num_frames, total_frames):
    ground_truths = []
    prediction = []
    segments = get_segments_with_frame_duration(labels, num_frames, 1/30, total_frames)

    for idx, start, end in segments:
        prediction.append((start, end))
        truth = start_end_label[idx]
        ground_truths.append((truth[0], truth[1]))
    return prediction, ground_truths

def get_predicted_labels(video_embeddings, text_embeddings):
    video_embeddings = F.normalize(video_embeddings, p=2, dim=1)  
    text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
    similarities = torch.mm(video_embeddings, text_embeddings.T) 
    similarities_smooth = F.avg_pool1d(similarities.T.unsqueeze(0), kernel_size=7, stride=1, padding=2).squeeze(0).T
    labels = torch.argmax(similarities_smooth, dim=1)
    actual_max_value = torch.max(similarities_smooth, dim=1)
    return labels, similarities


def split_videos_by_duration(videos: list[dict], T):

    groups = []
    current_group = []
    current_total = 0

    for video_dict in videos:
        duration = video_dict['end_time'] - video_dict['start_time']
        if duration > T:
            if current_group:
                groups.append(current_group)
                current_group = []
                current_total = 0
            groups.append([video_dict])
        else:
            if current_total + duration <= T:
                current_group.append(video_dict)
                current_total += duration
            else:
                groups.append(current_group)
                current_group = [video_dict]
                current_total = duration

    if current_group:
        groups.append(current_group)

    return groups