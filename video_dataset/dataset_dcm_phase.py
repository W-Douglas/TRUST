import os
import io
import numpy as np
import torch
import pydicom
from torchvision import transforms

# 复用原 dataset.py 的 VideoDataset 类
from .dataset import VideoDataset

# 导入原有的 transform 工具
from .transform import (
    create_random_augment,
    random_resized_crop,
    random_short_side_scale_jitter,
    random_crop,
)

# 尝试导入二进制读取函数
try:
    from .load_binary_internal import load_binary
except ImportError:
    from .load_binary import load_binary

def _read_dcm_file_to_frames(binary_data):
    """
    辅助函数：从二进制流中读取 DICOM 并转换为 List[np.ndarray(H, W, 3)]
    保持原逻辑不变
    """
    with io.BytesIO(binary_data) as f:
        ds = pydicom.dcmread(f)
        pixel_array = ds.pixel_array.astype(np.float32)

    frames_list = []
    
    if pixel_array.ndim == 2:
        pixel_array = pixel_array[np.newaxis, ...]

    if pixel_array.ndim == 3:
        # (T, H, W) -> (T, H, W, 3)
        frames_temp = np.stack([pixel_array] * 3, axis=-1)
        frames_list = [frames_temp[i] for i in range(frames_temp.shape[0])]

    elif pixel_array.ndim == 4:
        if pixel_array.shape[-1] == 3:
            frames_list = [pixel_array[i] for i in range(pixel_array.shape[0])]
        elif pixel_array.shape[-1] == 1:
            temp = np.concatenate([pixel_array] * 3, axis=-1)
            frames_list = [temp[i] for i in range(temp.shape[0])]
    
    return frames_list

class VideoDatasetDCMPhase(VideoDataset):
    """
    专门用于读取 DCM 文件并适配手术阶段识别（Phase Recognition）任务的 Dataset
    """

    def _load_annotations(self, annotation_path):
        """
        读取手术阶段的逐帧标注文件
        假设标注文件格式为每一行: <frame_id> <phase_label>
        返回: {frame_index: label}
        """
        labels = {}
        if not os.path.exists(annotation_path):
            # 如果找不到标注文件，返回空字典，后续处理会报错或跳过
            print(f"Warning: Annotation file not found: {annotation_path}")
            return labels
            
        with open(annotation_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    parts = line.split() 
                    # 兼容不同分隔符，取前两个值
                    frame_idx = int(parts[0])
                    label = int(parts[1])
                    labels[frame_idx] = label
                except ValueError:
                    continue
        return labels

    def __getitem__(self, idx):
        line = self.data_list[idx]
        
        # --- 解析行数据 ---
        # 新格式支持: video_path annotation_path target_frame_idx
        parts = line.split()
        path = parts[0]
        annotation_path = parts[1] if len(parts) > 1 else None
        
        # 尝试读取指定的中心帧索引 (如果列表里指定了)
        target_center_idx = int(parts[2]) if len(parts) > 2 else None

        path = os.path.join(self.data_root, path)
        if annotation_path:
            annotation_path = os.path.join(self.data_root, annotation_path)

        # 1. 读取数据
        raw_data = load_binary(path)
        frames = _read_dcm_file_to_frames(raw_data)
        total_video_frames = len(frames)

        # 2. 读取标注 (如果是测试且没有标注文件，可以跳过，但在 Phase Reco 中通常有 GT 用于计算 Metric)
        frame_labels_dict = self._load_annotations(annotation_path) if annotation_path else {}

        # --- 3. 确定中心帧 ---
        if target_center_idx is not None:
            # [推理模式]: 列表明确指定了要预测哪一帧
            center_frame_idx = target_center_idx
            # 边界保护
            center_frame_idx = max(0, min(center_frame_idx, total_video_frames - 1))
        else:
            # [训练/旧模式]: 自动选择
            valid_centers = list(frame_labels_dict.keys())
            if not valid_centers:
                center_frame_idx = total_video_frames // 2
            elif self.random_sample:
                center_frame_idx = np.random.choice(valid_centers)
            else:
                center_frame_idx = valid_centers[len(valid_centers) // 2]

        # 获取标签 (如果没有标注，默认为 -1 或 0)
        label = frame_labels_dict.get(center_frame_idx, 0)

        # --- 4. 构建 Clip (滑动窗口) ---
        actual_sampling_rate = self.sampling_rate if self.sampling_rate > 0 else 1
        half_window = (self.num_frames * actual_sampling_rate) // 2
        start_idx = center_frame_idx - half_window
        
        frame_indices = []
        for i in range(self.num_frames):
            idx = start_idx + i * actual_sampling_rate
            idx = max(0, min(idx, total_video_frames - 1))
            frame_indices.append(idx)

        # 提取帧
        frames_clip = [frames[x] for x in frame_indices]
        frames_clip = torch.as_tensor(np.stack(frames_clip)).float() / 255.

        # --- 数据增强与标准化 (保持不变) ---
        if self.random_sample:
            # ... (训练时的增强逻辑，保持不变) ...
             if self.auto_augment is not None:
                aug_transform = create_random_augment(
                    input_size=(frames_clip.size(1), frames_clip.size(2)),
                    auto_augment=self.auto_augment,
                    interpolation=self.interpolation,
                )
                frames_clip = frames_clip.permute(0, 3, 1, 2)
                frames_clip = [transforms.ToPILImage()(frames_clip[i]) for i in range(frames_clip.size(0))]
                frames_clip = aug_transform(frames_clip)
                frames_clip = torch.stack([transforms.ToTensor()(img) for img in frames_clip])
                frames_clip = frames_clip.permute(0, 2, 3, 1)
        
        # 标准化
        frames_clip = (frames_clip - self.mean) / self.std
        frames_clip = frames_clip.permute(3, 0, 1, 2) # C, T, H, W
        
        # Resize
        if self.random_sample:
             # ... (训练 Resize 逻辑保持不变)
             if self.resize_type == 'random_resized_crop':
                frames_clip = random_resized_crop(
                    frames_clip, self.spatial_size, self.spatial_size,
                    scale=self.scale_range,
                    interpolation=self.interpolation,
                )
             elif self.resize_type == 'random_short_side_scale_jitter':
                frames_clip, _ = random_short_side_scale_jitter(
                    frames_clip,
                    min_size=round(self.spatial_size * self.scale_range[0]),
                    max_size=round(self.spatial_size * self.scale_range[1]),
                    interpolation=self.interpolation,
                    )
                frames_clip, _ = random_crop(frames_clip, self.spatial_size)
             
             if self.mirror and torch.rand(1).item() < 0.5:
                frames_clip = frames_clip.flip(dims=(-1,))
        else:
            # [推理 Resize 逻辑]: 保持 Center Crop
            if frames_clip.size(-2) < frames_clip.size(-1):
                new_width = frames_clip.size(-1) * self.spatial_size // frames_clip.size(-2)
                new_height = self.spatial_size
            else:
                new_height = frames_clip.size(-2) * self.spatial_size // frames_clip.size(-1)
                new_width = self.spatial_size
            
            frames_clip = torch.nn.functional.interpolate(
                frames_clip, size=(new_height, new_width),
                mode=self.interpolation, align_corners=False,
            )
            
            # Center Crop
            frames_clip = self._generate_spatial_crops(frames_clip)[0] # 形状: [C, T, H, W]

            # --- 修复点：增加 View 维度 ---
            # main.py 期望数据包含 View 维度，即使只有 1 个 View
            # 形状变为: [1, C, T, H, W]
            frames_clip = frames_clip.unsqueeze(0)

        return frames_clip, label