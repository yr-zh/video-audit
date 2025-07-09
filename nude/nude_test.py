# from nudenet import NudeDetector
# detector = NudeDetector()
# # the 320n model included with the package will be used

# res = detector.detect('test.jpg') # Returns list of detections
# print(res)

import cv2
import numpy as np
from nudenet import NudeDetector
from concurrent.futures import ThreadPoolExecutor
import os
import time

class VideoPornDetector:
    def __init__(self, threshold=0.5, frame_interval=0.1):
        """
        :param threshold: 判定为色情的置信度阈值 (0-1)
        :param frame_interval: 抽帧间隔（每秒抽几帧）
        """
        self.detector = NudeDetector()  # 加载模型
        self.threshold = threshold
        self.frame_interval = frame_interval

    def extract_frames(self, video_path, output_dir="frames"):
        """抽帧并返回帧路径列表"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_paths = []

        # 计算实际抽帧间隔（按帧数）
        interval = int(fps * self.frame_interval)
        
        for i in range(0, frame_count, interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break

            frame_path = os.path.join(output_dir, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)

        cap.release()
        return frame_paths

    def detect_frame(self, frame_path):
        """检测单帧"""
        try:
            results = self.detector.detect(frame_path)
            for obj in results:
                if obj["class"] in ["FEMALE_GENITALIA_EXPOSED", "FEMALE_BREAST_EXPOSED", "ANUS_EXPOSED", "MALE_GENITALIA_EXPOSED", "BUTTOCKS_EXPOSED"] and obj["score"] > self.threshold:
                    return True, frame_path, obj
            return False, None, None
        except Exception as e:
            print(f"检测失败 {frame_path}: {str(e)}")
            return False, None, None

    def process_video(self, video_path, max_workers=4):
        """多线程处理视频"""
        start_time = time.time()
        print("开始抽帧...")
        frame_paths = self.extract_frames(video_path)
        print(f"抽帧完成，共 {len(frame_paths)} 帧")

        suspicious_frames = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.detect_frame, path) for path in frame_paths]
            for future in futures:
                is_porn, path, obj = future.result()
                if is_porn:
                    suspicious_frames.append((path, obj))

        # 清理临时帧文件
        for path in frame_paths:
            os.remove(path)

        print(f"检测完成，耗时 {time.time()-start_time:.2f}s")
        print(f"可疑帧数量: {len(suspicious_frames)}")
        return suspicious_frames

if __name__ == "__main__":

    video_path = "/mnt/data/leaderboard/56da1986c432e0fff72fb039491c3548/videos/63/7479906070043954495s_720_1282.mp4"
    detector = VideoPornDetector(threshold=0.7, frame_interval=5)
    results = detector.process_video(video_path)
    
    # 输出结果示例
    for path, obj in results:
        print(f"可疑帧: {path}, 检测到: {obj['class']} (置信度: {obj['score']:.2f})")