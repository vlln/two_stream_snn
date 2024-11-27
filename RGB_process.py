import os
import glob
import cv2
import torch
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from multiprocessing import Pool, Manager


# 提取光流函数
def extract_rgb_frames(vid_path, out_path):
    video = cv2.VideoCapture(vid_path)
    vid_name = vid_path.split('/')[-1].split('.')[0]
    out_full_path = os.path.join(out_path, vid_name, 'rgb')

    os.makedirs(out_full_path, exist_ok=True)

    frame_idx = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        cv2.imwrite(f"{out_full_path}/frame_{frame_idx:06d}.jpg", frame)
        frame_idx += 1

    video.release()
    return out_full_path


# 准备数据函数
def prepare_data_for_model_rgb(rgb_path, target_frame_count=4, target_size=(32, 32)):
    rgb_frames = sorted(glob.glob(os.path.join(rgb_path, "frame_*.jpg")))

    total_frames = len(rgb_frames)

    # 如果帧数少于目标帧数，补充帧
    if total_frames < target_frame_count:
        while len(rgb_frames) < target_frame_count:
            rgb_frames.append(rgb_frames[-1])  # 重复最后一帧

    # 如果帧数多于目标帧数，进行均匀采样
    elif total_frames > target_frame_count:
        indices = np.linspace(0, total_frames - 1, target_frame_count, dtype=int)
        rgb_frames = [rgb_frames[i] for i in indices]

    # 将 RGB 帧转换为 Tensor，调整大小到 target_size
    rgb_tensors = []
    for frame_path in rgb_frames:
        frame = cv2.imread(frame_path)  # 读取 RGB 图像
        frame = cv2.resize(frame, target_size)  # 调整大小
        frame = frame.transpose((2, 0, 1))  # 转换为 (C, H, W)
        # 对单帧应用 transform
        frame_tensor = torch.tensor(frame, dtype=torch.float32) / 255.0  # 归一化到 [0, 1]
        frame_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(frame_tensor)
        rgb_tensors.append(frame_tensor)

    # 堆叠帧，生成最终的张量
    rgb_tensors = torch.stack(rgb_tensors, dim=0)  # (time_step, c, target_size[0], target_size[1])
    return rgb_tensors


# 处理单个视频
def process_video_rgb(args):
    video_file, out_path, target_frame_count, target_size, class_name, class_to_idx = args
    rgb_path = extract_rgb_frames(video_file, out_path)
    if rgb_path:
        rgb_tensors = prepare_data_for_model_rgb(rgb_path, target_frame_count, target_size)
        label = class_to_idx[class_name]
        return rgb_tensors, label
    return None

if __name__ == '__main__':
    root_dir = '/data1/home/sunyi/large-files/Python_Objects/spiking_neural_network/UCF-101'
    out_path = '/data1/home/sunyi/large-files/Python_Objects/spiking_neural_network/RGB' 
    target_frame_count = 4
    target_size = (32, 32)

    # 类别映射为标签
    classes = sorted(os.listdir(root_dir))
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    # 创建参数列表
    video_args = []
    for cls_name in classes:
        cls_dir = os.path.join(root_dir, cls_name)
        video_files = glob.glob(os.path.join(cls_dir, '*.avi'))
        for video_file in video_files:
            video_args.append((video_file, out_path, target_frame_count, target_size, cls_name, class_to_idx))

    # 使用多进程处理
    with Pool(processes=128) as pool:  # 根据CPU核心数调整processes
        results = list(tqdm(pool.imap(process_video_rgb, video_args), total=len(video_args), desc="Processing videos"))

    # 汇总数据和标签
    data = [result[0] for result in results if result is not None]
    labels = [result[1] for result in results if result is not None]

    # 将数据和标签堆叠为张量
    data_tensor = torch.stack(data)  # (num_samples, time_step, c, h, w)
    labels_tensor = torch.tensor(labels)  # (num_samples,)

    # 检查输出维度
    print(f"Data tensor shape: {data_tensor.shape}")  # (num_samples, time_step, c, h, w)
    print(f"Labels tensor shape: {labels_tensor.shape}")  # (num_samples,)

    # 保存到 .pth 文件
    torch.save({'data': data_tensor, 'labels': labels_tensor}, 'UCF-101-RGB-data.pth')
    print("Data processing complete. ")
