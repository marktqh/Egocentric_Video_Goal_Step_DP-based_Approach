import torch
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import threading

class PredictionTimeoutException(Exception):
    """Custom exception for handling timeouts in prediction."""
    pass

def compute_IoU(pred, gt):

    assert isinstance(pred, list) and isinstance(gt, list)
    pred_is_list = isinstance(pred[0], list)
    gt_is_list = isinstance(gt[0], list)
    if not pred_is_list:
        pred = [pred]
    if not gt_is_list:
        gt = [gt]
    pred, gt = np.array(pred), np.array(gt)
    inter_left = np.maximum(pred[:, 0, None], gt[None, :, 0])
    inter_right = np.minimum(pred[:, 1, None], gt[None, :, 1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:, 0, None], gt[None, :, 0])
    union_right = np.maximum(pred[:, 1, None], gt[None, :, 1])
    union = np.maximum(0.0, union_right - union_left)
    overlap = 1.0 * inter / union
    if not gt_is_list:
        overlap = overlap[:, 0]
    if not pred_is_list:
        overlap = overlap[0]
    return overlap

def evaluate_performance(predictions, ground_truth, thresholds, topK):
    """
    Evaluates performance by comparing predictions against ground truth.
    Evaluates each chunk independently and calculates overall scores across all chunks.

    Parameters:
    - predictions: List[dict], each entry contains `video_uid`, `step_description`, and `predicted_times`
    - ground_truth: Ground truth data read from the JSON file, with start and end times
    - thresholds: List of IoU thresholds
    - topK: List of top-K values to evaluate

    Returns:
    - mean_results: Array of mean results over thresholds and topK values
    - mIoU: Mean IoU across all instances
    """
    gt_dict = {}
    num_gt_queries = 0

    # Build ground truth dictionary with keys as (video_uid, step_description)
    for video_datum in ground_truth["videos"]:
        video_uid = video_datum["video_uid"]
        for segment in video_datum["segments"]:
            key = (video_uid, segment["step_description"])
            if key not in gt_dict:
                gt_dict[key] = segment
                num_gt_queries += 1

    # Results storage
    results = [[[] for _ in topK] for _ in thresholds]
    iou_scores = []
    num_instances = 0

    # Evaluate each prediction independently
    for pred_datum in predictions:
        video_uid = pred_datum["video_uid"]
        step_description = pred_datum["step_description"]
        key = (video_uid, step_description)

        # Get the ground truth entry for the video and step description
        if key not in gt_dict:
            print(f"No ground truth found for video {video_uid}, step: {step_description}")
            continue
        gt_datum = gt_dict[key]

        # Ground truth time range for the current step description
        gt_time = [[gt_datum["start_time"], gt_datum["end_time"]]]

        # Compute IoU between prediction and ground truth
        overlap = compute_IoU(
            pred_datum["predicted_times"],  # [[s1, e1], [s2, e2], ...]
            gt_time  # [st, et]
        )
        iou_scores.append(np.mean(np.sort(overlap[0])[-3:]))  # Save the IoU score
        for tt, threshold in enumerate(thresholds):
            for rr, KK in enumerate(topK):
                results[tt][rr].append((overlap > threshold)[:KK].any())
        num_instances += 1

    # Calculate overall mean results and IoU
    mean_results = np.array(results).mean(axis=-1)  # Mean over all evaluated chunks
    mIoU = np.mean(iou_scores)  # Mean IoU across all chunks
    print(f"Evaluated: {num_instances} / {num_gt_queries} instances")

    return mean_results, mIoU

def dfs(frame_index, action_index, log_matrix, index_mark, repeated_actions, similarity = None):
    
    if frame_index + 1 == cols and index_mark + 1 >= rows and not repeated_actions:
        return log_matrix[action_index, frame_index], [action_index]

    if frame_index + 1 == cols and (index_mark + 1 < rows or repeated_actions):
        return float('-inf'), []

    # Key for caching
    key = (frame_index, action_index, index_mark, tuple(repeated_actions.items()))
    if key in cache:
        return cache[key]

    best_score = float('-inf')
    best_path = []

    # Option 1: Stay at current action
    if frame_index + 1 < cols:
        score, path = dfs(frame_index + 1, action_index, log_matrix, index_mark, repeated_actions)
        stay_score = log_matrix[action_index, frame_index] + score
        stay_path = [action_index] + path
        if stay_score > best_score:
            best_score = stay_score
            best_path = stay_path

    # Option 2: Move to next action
    if frame_index + 1 < cols and index_mark + 1 < rows:
        # Handle repeated actions
        if (index_mark + 1) in repeated_actions:
            local_repeated_actions = repeated_actions.copy()
            local_repeated_actions[index_mark + 1] -= 1
            if local_repeated_actions[index_mark + 1] == 0:
                del local_repeated_actions[index_mark + 1]
            score, path = dfs(frame_index + 1, index_mark + 1, log_matrix, index_mark + 1, local_repeated_actions)
        else:
            score, path = dfs(frame_index + 1, index_mark + 1, log_matrix, index_mark + 1, repeated_actions)
        move_score = log_matrix[action_index, frame_index] + score
        move_path = [action_index] + path
        if move_score > best_score:
            best_score = move_score
            best_path = move_path

    # Option 3: Move to a repeated action
    if frame_index + 1 < cols and repeated_actions:
        for repeated_action_index in repeated_actions:
            if index_mark >= repeated_action_index and action_index != repeated_action_index:
                local_repeated_actions = repeated_actions.copy()
                local_repeated_actions[repeated_action_index] -= 1
                if local_repeated_actions[repeated_action_index] == 0:
                    del local_repeated_actions[repeated_action_index]
                score, path = dfs(frame_index + 1, repeated_action_index, log_matrix, index_mark, local_repeated_actions)
                repeated_score = log_matrix[action_index, frame_index] + score
                repeated_path = [action_index] + path
                if repeated_score > best_score:
                    best_score = repeated_score
                    best_path = repeated_path
            else:
                break

    # Choose the best option and cache the result
    cache[key] = best_score, best_path
    # col_indices = [num for num in range(50)]
    # best_50_sum = similarity[optimal_path[:50], col_indices].sum()
    # all_50_sum = []
    # for key in cache.
    return cache[key]

def convert_raw_predictions_to_evaluation_format(raw_predictions, step_descriptions, video_uid):
    """
    Converts raw predictions into the format required by the evaluate_performance function.
    Handles same step descriptions within the same video by appending intervals to the same entry,
    but treats step descriptions from different videos as separate entries.
    Ensures the last action is captured correctly.

    Parameters:
    - raw_predictions: List[int], each element represents the label for a 2-second chunk
    - step_descriptions: List[str], list of step descriptions in the correct order
    - video_uid: String, unique identifier for the video

    Returns:
    - List[dict], where each entry contains `video_uid`, `step_description`, and `predicted_times`.
    """
    predictions = []
    chunk_duration = 2

    # Create a mapping for step descriptions within the same video
    video_predictions = {}

    for step_idx, step_description in enumerate(step_descriptions):
        if (video_uid, step_description) not in video_predictions:
            video_predictions[(video_uid, step_description)] = []

        predicted_times = []
        start_time = None

        # Read through chunks and map each continuous label appearance
        for chunk_idx, label in enumerate(raw_predictions):
            if label == step_idx:
                if start_time is None:
                    # Start a new interval when encountering a label
                    start_time = chunk_idx * chunk_duration
                end_time = (chunk_idx + 1) * chunk_duration
            else:
                if start_time is not None:
                    # End the current interval and reset start_time
                    predicted_times.append([start_time, end_time])
                    start_time = None

        # Capture the last interval if still active
        if start_time is not None:
            predicted_times.append([start_time, end_time])

        # Append to existing predictions for this step description in this video
        video_predictions[(video_uid, step_description)].extend(predicted_times)

    # Ensure all predicted_times have exactly 5 intervals by padding or truncating
    for (video_uid, step_description), predicted_times in video_predictions.items():
        if len(predicted_times) < 5:
            last_interval = predicted_times[-1] if predicted_times else [0, 2]
            while len(predicted_times) < 5:
                predicted_times.append(last_interval)
        elif len(predicted_times) > 5:
            predicted_times = predicted_times[:5]

        # Add to the final predictions list
        predictions.append({
            "video_uid": video_uid,
            "step_description": step_description,
            "predicted_times": predicted_times
        })

    return predictions

def predict(video_uid, videos, timeout=5*60):
    try:
        similarity = torch.load(f'D:\DGSA1006\output\{video_uid}\similarity.pt')
    except Exception:
        print(f'Error loading similarity matrix for {video_uid}')
        return None
    similarity = similarity.T

    seen = set()
    unique_indices = []

    # Iterate through rows and collect indices of the first occurrence of each unique row
    for i, row in enumerate(similarity):
        row_tuple = tuple(row.tolist())  # Convert to tuple to make it hashable
        if row_tuple not in seen:
            seen.add(row_tuple)
            unique_indices.append(i)

    # Use the collected indices to get the unique rows in the original order
    similarity = similarity[unique_indices]

    actions = [v for v in videos if v['video_uid'] == video_uid][0]['segments'] 
    descriptions = [step['step_description'] for step in actions]

    action_count = {}
    for des in descriptions:
        action_count[des] = action_count.get(des, 0) + 1
    action_count = {index: value for index, (key, value) in enumerate(action_count.items())}

    repeated_actions = {}
    for key, value in action_count.items():
        if value > 1:
            repeated_actions[key] = value
            if key == 0:
                repeated_actions[key] -= 1

    # take the log of the similarity matrix
    similarity = torch.clamp(similarity, min=1e-10)
    similarity_log = torch.log(similarity)

    # initialize the cache as global variable
    global rows, cols, cache
    rows, cols = similarity_log.shape
    cache = {}

    optimal_alignment_score, optimal_path = dfs(0, 0, similarity_log, 0, repeated_actions)

    seen = set()
    unique_step_descriptions = []
    for step in descriptions:
        if step not in seen:
            seen.add(step)
            unique_step_descriptions.append(step)

    return convert_raw_predictions_to_evaluation_format(optimal_path, unique_step_descriptions, video_uid)

def predict_with_timeout(predict_function, video_uid, videos, timeout=5*60):
    """
    Runs the prediction function with a timeout using threading.

    Parameters:
    - predict_function: Callable, the prediction function to execute
    - video_uid: String, unique identifier for the video
    - videos: List or dict containing video data
    - timeout: Time limit in seconds for the prediction function

    Returns:
    - predictions: The result of the prediction function if successful
    - error: None if successful, or an error message if it times out
    """
    # Container for the result
    result = []

    # Wrapper function to capture the result or exception
    def wrapper():
        try:
            result.append(predict_function(video_uid, videos))
        except Exception as e:
            result.append(e)

    # Create and start the thread
    thread = threading.Thread(target=wrapper)
    thread.start()
    thread.join(timeout)  # Wait for the thread to finish or timeout

    if thread.is_alive():
        # If the thread is still running, it exceeded the timeout
        thread.join(0)  # Clean up the thread
        return None, f"Prediction for video {video_uid} timed out"

    # Check the result
    if isinstance(result[0], Exception):
        return None, f"Error during prediction for video {video_uid}: {result[0]}"
    return result[0], None

def main():
    val_annotation_file = r'D:\DGSA1006\ego4d_data\v2\annotations\goalstep_val.json'
    with open(val_annotation_file, 'r') as file:
        data = json.load(file)

    videos = data['videos']

    # video_lt = ['0e0d6704-1c6c-4a62-bc97-cc55658cf8ac',
    #             '0e6fb738-05fc-4dd5-9746-a8e10efe8c20',
    #             '0fcf23a0-fc53-4378-9a99-18c4f109f659',
    #             '1a327b0c-b78b-4ae2-add0-49334bccddcd',
    #             '2f23b607-f2e6-4f58-85d3-004c840bead2',
    #             '3a158275-c4c2-4bde-a186-788341e43ed4',
    #             '4d770048-cfde-411f-ad53-e4d57a7dd0d1',
    #             '04fe8f4d-081e-437e-a56a-2d53b6233fc9',
    #             '6bb4a311-a05b-413e-879c-1c30809e62d4',
    #             '13c76616-f168-4af0-8d2a-fe82ce232d6a']

    # get all video uids
    # timed_out_videos = [
    #     "864fa3d8-9b18-44cb-a8e9-9b40765e2d0c",
    #     "grp-a8ce8831-58e4-4c84-926e-8f948fc782a7",
    #     "e4d961f5-176f-4dde-864a-bb94523312a1",
    #     "grp-9c5c9efc-608f-4fdf-9c29-2251a451c8f9",
    #     "e8657b65-be92-401d-a8d8-2fa32cb861c0",
    #     "26f1c77e-814e-4609-ad67-12447d1627e1",
    #     "grp-79f47a60-f1e9-4232-88b8-a1836e7dfd30",
    #     "grp-a9c519a7-4776-42d6-bcf1-270f0d302843",
    #     "grp-5b24c19c-0bde-46ce-a32e-418b5ffaa8a3",
    #     "grp-304735ba-6bf5-4d39-bcb5-0dabddb11d68",
    #     "bd970f5b-3fd9-4ae9-9f2b-738e8ca54c1e",
    #     "grp-ebce88dd-4852-4506-9dcc-5f5798ce1cbf",
    #     "623bc0af-7d4f-4b85-9263-2d04e016283d",
    #     "grp-719d9e89-4eb2-49ea-be14-dc2637dc303f",
    #     "grp-1a9bb53f-01ae-4bd7-afa3-a92570679a7a",
    #     "90e4acc5-28a2-4bd4-972c-c6f7e18e1cde",
    #     "228eb02a-6c89-4aa0-9cd4-0cdab1550c83",
    #     "grp-2bccee1b-0ade-47ad-8e15-ad6c00861540",
    #     "cc575a16-64fd-4cda-9248-5d85f506fdfd",
    #     "6af04762-9ccf-41aa-bbfb-48a443c7cec3",
    #     "2c0c6508-397f-4c48-aeb7-abc7a3cae8d1",
    #     "0a01978c-e16d-4587-95f1-49efa3ab15d9",
    #     "grp-c56e7e04-8787-4df1-98c6-352076f61e53",
    #     "grp-7fb63b81-0a4f-4e9e-906b-60d1935d53c7",
    #     "grp-51fc62f8-00f4-44e3-af9c-7ebb63da6c3d",
    #     "002c3b5c-ed86-4af3-99a1-4b497b7c8a86",
    #     "46d00bf5-ed73-4e5f-84eb-9c880eec10d8",
    #     "f40e0f92-2250-46c9-98a0-8ccf23d164e0",
    #     "afbab8ce-797a-4c6e-800b-5acd9bf1653e",
    #     "3bd5bf35-d6ac-43b2-ab75-1558a37c8550",
    #     "grp-3d7ccb44-b05d-4b67-baae-4e0f55d8307b",
    #     "478d7fa4-b174-4266-b0ff-bd180ba0b806",
    #     "c2d06df7-5d3a-4116-9edb-f1c81a4f669b",
    #     "04ac4c40-22fd-42aa-a7f0-ee597ffb7058",
    #     "b2a1b8ca-99d6-4f26-953f-426e89649e90",
    #     "7ddbf8a2-5b3e-44cd-b9dc-db17ec06831b"
    # ]

    video_lt = [[video['video_uid'], video['end_time']] for video in videos]
    # sort from shortest to longest video
    video_lt = sorted(video_lt, key=lambda x: x[1])
    video_lt = [video[0] for video in video_lt]
    # remove timed out videos
    # video_lt = [video for video in video_lt if video not in timed_out_videos]
    
    predictions = []
    # timed_out = timed_out_videos
    timed_out = []
    # append the last two videos to the timed out list
    timed_out.extend(video_lt[-4:])
    video_lt = video_lt[:-4]
    counter = 0
    for video_uid in tqdm(video_lt):
        try:
            # try:
            #     similarity = torch.load(f'D:\DGSA1006\output\{video_uid}\similarity.pt')
            #     # print(f'Loaded similarity matrix for {video_uid}')
            # except Exception:
            #     print(f'Error loading similarity matrix for {video_uid}')
            #     # predictions.append(None)
            #     continue
            # similarity = similarity.T

            # seen = set()
            # unique_indices = []

            # # Iterate through rows and collect indices of the first occurrence of each unique row
            # for i, row in enumerate(similarity):
            #     row_tuple = tuple(row.tolist())  # Convert to tuple to make it hashable
            #     if row_tuple not in seen:
            #         seen.add(row_tuple)
            #         unique_indices.append(i)

            # # Use the collected indices to get the unique rows in the original order
            # similarity = similarity[unique_indices]

            # actions = [v for v in videos if v['video_uid'] == video_uid][0]['segments'] 
            # descriptions = [step['step_description'] for step in actions]

            # action_count = {}
            # for des in descriptions:
            #     action_count[des] = action_count.get(des, 0) + 1
            # action_count = {index: value for index, (key, value) in enumerate(action_count.items())}

            # repeated_actions = {}
            # for key, value in action_count.items():
            #     if value > 1:
            #         repeated_actions[key] = value
            #         if key == 0:
            #             repeated_actions[key] -= 1

            # # take the log of the similarity matrix
            # similarity = torch.clamp(similarity, min=1e-10)
            # similarity_log = torch.log(similarity)

            # # initialize the cache as global variable
            # global rows, cols, cache
            # rows, cols = similarity_log.shape
            # cache = {}

            # optimal_alignment_score, optimal_path = dfs(0, 0, similarity_log, 0, repeated_actions)

            # seen = set()
            # unique_step_descriptions = []
            # for step in descriptions:
            #     if step not in seen:
            #         seen.add(step)
            #         unique_step_descriptions.append(step)

            # predictions.extend(convert_raw_predictions_to_evaluation_format(optimal_path, unique_step_descriptions, video_uid))
            result = predict_with_timeout(predict, video_uid, videos, timeout=10*60)
            if result[0] is not None:
                predictions.extend(result[0])
            else:
                print(result[1])
                timed_out.append(video_uid)
        # except max recursion depth error
        except RecursionError:
            print(f"Recursion error for video {video_uid}")
            # predictions.append(None)
            timed_out.append(video_uid)
            continue
        counter += 1
        if counter > 10:
            break

    mean_results, mIoU = evaluate_performance(predictions, data, [0.3, 0.5], [1, 3, 5])
    df = pd.DataFrame(mean_results, index=[0.3, 0.5], columns=['top 1', 'top 3', 'top 5'])
    df.to_csv('evaluation_results_V8_Val.csv')
    # write Mean IoU to the last row of the output csv file
    with open('evaluation_results_V8_Val.csv', 'a') as f:
        f.write(f"Mean IoU: {mIoU}\n")
        f.write(f"Number of timed out videos: {len(timed_out)}\n")
        f.write(f"Timed out videos: {timed_out}\n")
    f.close()

    print(f"Mean IoU: {mIoU}")
    print(f"Number of timed out videos: {len(timed_out)}")
    print(f"Timed out videos: {timed_out}")
    print(df)

if __name__ == '__main__':
    sys.setrecursionlimit(5000)
    # print recursion limit
    print(f'Recursion limit: {sys.getrecursionlimit()}')
    main()