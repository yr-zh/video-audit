import os
import cv2
import shutil
import json
from tqdm import tqdm
import pandas as pd

def extract_frames(video_path, output_dir, num_frames=3, duration=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return []
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 获取视频帧率
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取总帧数
    if total_frames == 0:
        print("视频没有帧")
        return []
    
    start_frame = 0
    end_frame = min(duration * fps, total_frames)  # 确保不超过视频总帧数
    
    frame_indices = [
        start_frame + (end_frame - start_frame) // (num_frames - 1) * i 
        for i in range(num_frames)
    ]

    cover_paths = []
    video_filename = os.path.basename(video_path).split(".")[0]
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            cover_filename = f"{video_filename}_frame_{frame_idx}.jpg"
            cover_path = os.path.join(output_dir, cover_filename)
            cv2.imwrite(cover_path, frame)
            cover_paths.append(cover_path)
        else:
            print(f"无法读取帧{frame_idx}")

    cap.release()
    return cover_paths

def augment_positive_samples(train_data_path, scale=4):
    positive_samples = []
    with open(os.path.join(train_data_path, 'index.jsonl')) as fp:
        for line in tqdm(fp):
            case = json.loads(line.strip())
            cover_filename = case["content"]["cover"]["value"]
            video_filename = case["content"]["video"]["value"]
            if not case["label"] in ["Q-1", "Q0"] or case["content"]["meta"]["value"]["categoryLevel1"] == "体育":
                continue
            # else:
            #     tgt_cover_path = f"positive_samples/covers/{os.path.basename(cover_filename)}"
            #     shutil.copyfile(cover_filename, tgt_cover_path)
            #     positive_samples.append(
            #         {
            #             "categoryLevel1": case["content"]["meta"]["value"]["categoryLevel1"],
            #             "cover_path": tgt_cover_path,
            #             "label": 1
            #         }
            #     )
            frame_paths = extract_frames(video_filename, "positive_samples/frames", scale)
            for frame_path in frame_paths:
                positive_samples.append(
                    {
                        "categoryLevel1": case["content"]["meta"]["value"]["categoryLevel1"],
                        "cover_path": frame_path,
                        "label": 1
                    }
                )
    positive_samples_df = pd.DataFrame(positive_samples)
    return positive_samples_df