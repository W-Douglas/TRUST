import os
import io
import numpy as np
import torch
import pydicom
from torchvision import transforms

# 复用原 dataset.py 的 VideoDataset 类，方便继承或参考
# 假设原文件名为 dataset.py
from .dataset import VideoDataset

# 导入原有的 transform 工具 (必须保证 dataset.py 同级目录下有 transform.py)
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
    """
    # 使用 BytesIO 将二进制数据包装成文件流供 pydicom 读取
    with io.BytesIO(binary_data) as f:
        ds = pydicom.dcmread(f)
        # 获取像素矩阵 (通常是 T, H, W 或 T, H, W, C)
        # astype(float32) 是为了后续归一化计算
        pixel_array = ds.pixel_array.astype(np.float32)

    frames_list = []
    
    # 情况 A: 单帧图像 (H, W) -> 扩展为 (1, H, W)
    if pixel_array.ndim == 2:
        pixel_array = pixel_array[np.newaxis, ...]

    # 处理数据维度，目标是生成 list of (H, W, 3)
    if pixel_array.ndim == 3:
        # 情况 B: 灰度视频 (T, H, W) -> 复制为伪彩色 (T, H, W, 3)
        # 或者 RGB 单帧 (H, W, 3) -> 此时 T=H (误判)，但在 DICOM 视频语境下通常是 T,H,W
        # 我们可以通过判断第一个维度是否等于 num_frames 来辅助，但这里假设是 (T, H, W)
        
        # 复制通道：(T, H, W) -> (T, H, W, 3)
        frames_temp = np.stack([pixel_array] * 3, axis=-1)
        
        # 将 numpy 数组切片放入 list
        frames_list = [frames_temp[i] for i in range(frames_temp.shape[0])]

    elif pixel_array.ndim == 4:
        # 情况 C: RGB 视频 (T, H, W, 3)
        if pixel_array.shape[-1] == 3:
            frames_list = [pixel_array[i] for i in range(pixel_array.shape[0])]
        # 情况 D: (T, H, W, 1) -> (T, H, W, 3)
        elif pixel_array.shape[-1] == 1:
            temp = np.concatenate([pixel_array] * 3, axis=-1)
            frames_list = [temp[i] for i in range(temp.shape[0])]
    
    return frames_list

class VideoDatasetDCM(VideoDataset):
    """
    专门用于读取 DCM 文件的 Dataset，继承自原 VideoDataset 以复用参数逻辑
    """

    def __getitem__(self, idx):
        line = self.data_list[idx]
        if self.load_labels:
            path, label = line.split(' ')
            label = int(label)
        else:
            path = line.split(' ')[0]
            label = None
        path = os.path.join(self.data_root, path)

        # 1. 读取二进制数据
        raw_data = load_binary(path)
        
        # 2. 解析 DICOM 获取帧列表
        # frames 结构: List[np.ndarray], 每个元素 shape 为 (H, W, 3)
        frames = _read_dcm_file_to_frames(raw_data)

        # 3. 数据处理流程 (完全复用原 dataset.py 的逻辑)
        if self.random_sample:
            # 随机采样帧索引
            frame_idx = self._random_sample_frame_idx(len(frames))
            
            # 提取选中的帧
            frames = [frames[x] for x in frame_idx]
            
            # 转为 Tensor 并归一化到 [0, 1]
            # np.stack(frames) -> (T, H, W, 3)
            frames = torch.as_tensor(np.stack(frames)).float() / 255.

            # --- AutoAugment (完全照搬原文件) ---
            if self.auto_augment is not None:
                aug_transform = create_random_augment(
                    input_size=(frames.size(1), frames.size(2)),
                    auto_augment=self.auto_augment,
                    interpolation=self.interpolation,
                )
                frames = frames.permute(0, 3, 1, 2) # T, C, H, W
                frames = [transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))]
                frames = aug_transform(frames)
                frames = torch.stack([transforms.ToTensor()(img) for img in frames])
                frames = frames.permute(0, 2, 3, 1) # 回到 T, H, W, C

            # 标准化
            frames = (frames - self.mean) / self.std
            
            # --- 关键维度变换 ---
            # 原文件此处将 (T, H, W, C) 变为 (C, T, H, W)
            frames = frames.permute(3, 0, 1, 2) 

            # --- 空间裁剪与 Resize ---
            if self.resize_type == 'random_resized_crop':
                frames = random_resized_crop(
                    frames, self.spatial_size, self.spatial_size,
                    scale=self.scale_range,
                    interpolation=self.interpolation,
                )
            elif self.resize_type == 'random_short_side_scale_jitter':
                frames, _ = random_short_side_scale_jitter(
                    frames,
                    min_size=round(self.spatial_size * self.scale_range[0]),
                    max_size=round(self.spatial_size * self.scale_range[1]),
                    interpolation=self.interpolation,
                    )
                frames, _ = random_crop(frames, self.spatial_size)
            else:
                raise NotImplementedError()

            if self.random_erasing is not None:
                frames = self.random_erasing(frames.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)

            if self.mirror and torch.rand(1).item() < 0.5:
                frames = frames.flip(dims=(-1,))
            
        else:
            # --- 验证/测试模式的处理逻辑 ---
            frames = [frames[x] for x in range(len(frames))] # 保持全部帧，或者根据逻辑筛选
            frames = torch.as_tensor(np.stack(frames))
            frames = frames.float() / 255.

            frames = (frames - self.mean) / self.std
            frames = frames.permute(3, 0, 1, 2) # (T, H, W, C) -> (C, T, H, W)
            
            # Resize
            if frames.size(-2) < frames.size(-1):
                new_width = frames.size(-1) * self.spatial_size // frames.size(-2)
                new_height = self.spatial_size
            else:
                new_height = frames.size(-2) * self.spatial_size // frames.size(-1)
                new_width = self.spatial_size
            
            frames = torch.nn.functional.interpolate(
                frames, size=(new_height, new_width),
                mode=self.interpolation, align_corners=False,
            )

            # 生成多个 View (Crop)
            frames = self._generate_spatial_crops(frames)
            frames = sum([self._generate_temporal_crops(x) for x in frames], [])
            if len(frames) > 1:
                frames = torch.stack(frames)

        if label is None:
            return frames
        else:
            return frames, label