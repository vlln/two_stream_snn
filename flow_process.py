import os
import glob
import cv2
import torch
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from multiprocessing import Pool, Manager

# 定义光流预处理 transform
flow_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5], std=[0.226, 0.226])
])

# 提取光流函数
def extract_optical_flow(vid_path, out_path):
    """
    使用 TV-L1 算法从视频中提取光流。
    """
    video = cv2.VideoCapture(vid_path)
    vid_name = vid_path.split('/')[-1].split('.')[0]
    out_full_path = os.path.join(out_path, vid_name, 'flow')

    os.makedirs(out_full_path, exist_ok=True)

    ret, prev_frame = video.read()
    if not ret:
        print(f"Failed to read video {vid_path}")
        return None

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_idx = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        flow_x = flow[..., 0]
        flow_y = flow[..., 1]

        flow_x = cv2.normalize(flow_x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        flow_y = cv2.normalize(flow_y, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        cv2.imwrite(f"{out_full_path}/flow_x_{frame_idx:06d}.jpg", flow_x)
        cv2.imwrite(f"{out_full_path}/flow_y_{frame_idx:06d}.jpg", flow_y)

        prev_gray = gray
        frame_idx += 1

    video.release()
    return out_full_path

# 准备数据函数
def prepare_data_for_model(flow_path, target_frame_count=4, target_size=(32, 32)):
    """
    加载提取的光流帧，并统一帧数量和大小。
    """
    flow_x_frames = sorted(glob.glob(os.path.join(flow_path, "flow_x_*.jpg")))
    flow_y_frames = sorted(glob.glob(os.path.join(flow_path, "flow_y_*.jpg")))

    total_frames = len(flow_x_frames)

    # 如果帧数少于目标帧数，补充帧
    if total_frames < target_frame_count:
        while len(flow_x_frames) < target_frame_count:
            flow_x_frames.append(flow_x_frames[-1])  # 重复最后一帧
            flow_y_frames.append(flow_y_frames[-1])

    # 如果帧数多于目标帧数，进行均匀采样
    elif total_frames > target_frame_count:
        indices = np.linspace(0, total_frames - 1, target_frame_count, dtype=int)
        flow_x_frames = [flow_x_frames[i] for i in indices]
        flow_y_frames = [flow_y_frames[i] for i in indices]

    # 将光流帧转换为 Tensor，调整大小到 target_size
    flow_tensors = []
    for i in range(target_frame_count):
        flow_x = cv2.imread(flow_x_frames[i], 0)  # 读取为灰度图
        flow_y = cv2.imread(flow_y_frames[i], 0)  # 读取为灰度图
        flow_x = cv2.resize(flow_x, target_size)  # 调整大小
        flow_y = cv2.resize(flow_y, target_size)  # 调整大小
        flow_frame = np.stack([flow_x, flow_y], axis=0)  # (2, target_size[0], target_size[1])
        # 对单帧应用 transform
        flow_frame = torch.tensor(flow_frame, dtype=torch.float32) / 255.0  # 手动归一化到 [0, 1]
        flow_frame = transforms.Normalize(mean=[0.5, 0.5], std=[0.226, 0.226])(flow_frame)
        flow_tensors.append(flow_frame)

    # 堆叠帧，生成最终的张量
    flow_tensors = torch.stack(flow_tensors, dim=0)  # (time_step, c, target_size[0], target_size[1])
    return flow_tensors

# 处理单个视频的函数
def process_video(args):
    video_file, out_path, target_frame_count, target_size, class_name, class_to_idx = args
    flow_path = extract_optical_flow(video_file, out_path)
    if flow_path:
        flow_tensors = prepare_data_for_model(flow_path, target_frame_count, target_size)
        label = class_to_idx[class_name]
        return flow_tensors, label
    return None

if __name__ == '__main__':
    root_dir = '/data0/home/wenyaozhi/wyz/snn/UCF-101/UCF-101'
    out_path = '/data0/home/wenyaozhi/wyz/snn/flow_output'  # 光流输出路径
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
        results = list(tqdm(pool.imap(process_video, video_args), total=len(video_args), desc="Processing videos"))

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
    torch.save({'data': data_tensor, 'labels': labels_tensor}, 'ucf101_flow_data.pth')
    print("Data processing complete. Saved to ucf101_flow_data.pth")
