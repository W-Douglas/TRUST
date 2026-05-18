import torch
import torch.nn.functional as F
import torchvision
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # <--- 新增这行

class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # 递归寻找目标层
        target_layer = dict([*self.model.named_modules()])[self.target_layer_name]
        
        self.hooks.append(target_layer.register_forward_hook(forward_hook))
        self.hooks.append(target_layer.register_backward_hook(backward_hook)) # 注意：PyTorch版本差异可能需要 register_full_backward_hook

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def generate_cam(self, class_idx=None):
        # 1. Global Average Pooling of Gradients -> Weights
        # gradients shape: [B, C, T, H, W] or [B, C, H, W] depending on layer
        # Assuming [B, C, T, H, W] for 3D or (B*T, C, H, W) for 2D
        
        if self.gradients is None or self.activations is None:
            return None

        # 处理维度: 假设是 [BT, C, H, W] (ViT Patch特征重组后) 或 [B, C, T, H, W]
        # 这里针对你的 ViT 架构，通常 Hook 的是 Patch Embedding 后的 ResBlock 输出
        # Shape 可能是 [BT, L, C] -> 需要重塑
        
        # 简化处理：假设我们拿到的是 [N, C, H, W] 格式
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        cam = F.relu(cam) # ReLU is important

        N, C, H, W = cam.shape
        cam_flat = cam.view(N, -1) # [N, H*W]

        # 针对每一张图计算 min 和 max
        min_val, _ = torch.min(cam_flat, dim=1, keepdim=True) # [N, 1]
        max_val, _ = torch.max(cam_flat, dim=1, keepdim=True) # [N, 1]

        # 广播机制进行归一化
        # 恢复维度 [N, 1, 1, 1] 用于广播
        min_val = min_val.view(N, 1, 1, 1)
        max_val = max_val.view(N, 1, 1, 1)

        cam = (cam - min_val) / (max_val - min_val + 1e-8)
            
        # 归一化
        # cam = cam - cam.min()
        # cam = cam / (cam.max() + 1e-8)
        return cam

class FeatureVisualizer:
    def __init__(self, model, save_root):
        self.model = model
        self.save_root = save_root
        self.raw_dir = os.path.join(save_root, 'raw_images')
        self.heatmap_dir = os.path.join(save_root, 'heatmaps')
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.heatmap_dir, exist_ok=True)
        
    def process_vit_feature(self, feat, H_img, W_img):
        """
        处理 ViT 特征 [BT, L, C] -> Heatmap [BT, H, W]
        """
        # 假设 feat 是 [BT, 197, 768] (含CLS)
        if len(feat.shape) == 3:
            # 去掉 CLS token
            if feat.shape[1] > (14*14): # 假设 14x14 patches
                feat = feat[:, 1:, :] 
            
            # Reshape: [BT, 196, 768] -> [BT, 14, 14, 768] -> [BT, 768, 14, 14]
            L = feat.shape[1]
            H_feat = int(np.sqrt(L))
            feat = feat.permute(0, 2, 1).view(-1, feat.shape[-1], H_feat, H_feat)
        
        # 简单 Feature Map 可视化: 对 Channel 取平均
        heatmap = torch.mean(feat, dim=1) # [BT, 14, 14]
        
        # 归一化
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # 上采样到原图尺寸
        heatmap = F.interpolate(heatmap.unsqueeze(1), size=(H_img, W_img), mode='bilinear', align_corners=False)
        return heatmap.squeeze(1)

    def save_batch(self, images, heatmaps, batch_idx, prefix=''):
        """
        images: [B, C, T, H, W] 原始图像 tensor (normalized)
        heatmaps: [BT, H, W] 热力图 tensor
        """
        # 反归一化 (根据你的数据集 mean/std 调整)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1).to(images.device)
        images = images * std + mean
        
        B, C, T, H, W = images.shape
        images = images.permute(0, 2, 3, 4, 1).cpu().numpy() # [B, T, H, W, 3]
        heatmaps = heatmaps.view(B, T, H, W).detach().cpu().numpy()
        
        for b in range(B):
            for t in range(T):
                # 1. 保存原图
                img_uint8 = (np.clip(images[b, t], 0, 1) * 255).astype(np.uint8)
                img_path = os.path.join(self.raw_dir, f'{prefix}b{batch_idx}_s{b}_t{t}.png')
                img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_path, img_bgr)
                
                # 2. 保存热力图叠加
                hm = heatmaps[b, t]
                hm[hm < 0.2] = 0
                
                hm_uint8 = (hm * 255).astype(np.uint8)
                hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
                
                overlay = cv2.addWeighted(img_bgr, 0.6, hm_color, 0.4, 0)
                hm_path = os.path.join(self.heatmap_dir, f'{prefix}b{batch_idx}_s{b}_t{t}_cam.png')
                cv2.imwrite(hm_path, overlay)

class AttentionVisualizer:
    def __init__(self, model, save_root):
        self.model = model
        self.save_root = os.path.join(save_root, 'attn_maps')
        os.makedirs(self.save_root, exist_ok=True)

    def save_attention_map(self, raw_images, batch_idx):
        """
        raw_images: [Batch, C, T, H, W] (原始图像数据，用于叠加)
        """
        # 1. 从模型最后一层 Transformer Block 获取保存的权重
        # 路径: model -> module -> visual -> transformer -> resblocks -> [-1]
        try:
            # 假设使用 DDP，所以有 .module
            last_block = self.model.module.visual.transformer.resblocks[-1]
        except AttributeError:
            # 单卡情况
            last_block = self.model.visual.transformer.resblocks[-1]
            
        if not hasattr(last_block, 'last_attn_weights'):
            print("Warning: No attention weights found. Did you modify ResidualAttentionBlock?")
            return

        # attn_weights shape: [Batch, Heads, L, L] 或 [Batch, L, L]
        attn_weights = last_block.last_attn_weights
        
        # 2. 处理多头 (Average over Heads)
        if attn_weights.dim() == 4: # [B, H, L, L]
            attn_weights = attn_weights.mean(dim=1) # -> [B, L, L]
        
        # 3. 提取 CLS Token 对 Spatial Tokens 的关注
        # Matrix 结构: [B, Target_Len, Source_Len]
        # index 0 是 CLS，1: 是 Patch
        # 我们看 CLS (Row 0) 关注了哪些 Source (Cols 1:)
        cls_attn = attn_weights[:, 0, 1:] # [Batch, 196] (假设 14x14)
        
        # 4. Reshape 成 2D 图片
        B_attn, L_spatial = cls_attn.shape
        H_feat = int(np.sqrt(L_spatial)) # 14
        
        # [Batch, 14, 14]
        attn_map = cls_attn.view(B_attn, H_feat, H_feat)

        # 5. 归一化 (Min-Max) 以便可视化
        # 注意：Attention 数值通常很小，直接归一化看相对强度
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

        # 6. 处理图像并保存
        self._plot_and_save(raw_images, attn_map, batch_idx)

    def _plot_and_save(self, raw_images, attn_map_tensor, batch_idx):
        """
        将 Attention Map 叠加到原图并保存
        """
        # 确保 raw_images 是 5维 [B, C, T, H, W]
        # 如果之前在 main.py 里 flatten 过了，这里可能是 [BT, C, T, H, W]
        # 我们只取第一帧 T=0 或者中间帧来展示
        
        # 反归一化图片 (假设 ImageNet Mean/Std)
        # 如果您没做 Normalization，请注释下面两行
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cpu()
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cpu()
        
        B = raw_images.shape[0]
        
        for b in range(B):
            # 取第一帧 (T=0)
            img_tensor = raw_images[b, :, 0, :, :].cpu() # [C, H, W]
            # img_tensor = img_tensor * std + mean # 反归一化
            
            img_np = img_tensor.permute(1, 2, 0).numpy()
            img_np = np.clip(img_np, 0, 1)
            img_uint8 = (img_np * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

            # 处理 Attention Map
            # 插值放大到原图尺寸
            att = attn_map_tensor[b].unsqueeze(0).unsqueeze(0) # [1, 1, 14, 14]
            att = F.interpolate(att, size=(img_np.shape[0], img_np.shape[1]), mode='bilinear')
            att = att.squeeze().numpy()
            
            att_uint8 = (att * 255).astype(np.uint8)
            att_heatmap = cv2.applyColorMap(att_uint8, cv2.COLORMAP_JET)
            
            # 叠加
            overlay = cv2.addWeighted(img_bgr, 0.6, att_heatmap, 0.4, 0)
            
            # 保存
            save_path = os.path.join(self.save_root, f'batch{batch_idx}_sample{b}_attn.png')
            cv2.imwrite(save_path, overlay)


class SingleFeatureVisualizer:
    def __init__(self, model, save_root):
        self.model = model
        self.save_root = os.path.join(save_root, 'feature_maps_pixelated')
        os.makedirs(self.save_root, exist_ok=True)
        self.activations = {}
        self.hooks = []

    def register_hook(self, layer_name):
        """注册 Hook 到指定层"""
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, tuple):
                    output = output[0] # ViT 输出通常是 tuple
                self.activations[name] = output.detach()
            return hook

        module = self.model
        try:
            if hasattr(module, 'module'):
                module = module.module
            for part in layer_name.split('.'):
                module = getattr(module, part)
            self.hooks.append(module.register_forward_hook(get_activation(layer_name)))
            print(f"Hook registered: {layer_name}")
        except AttributeError:
            print(f"Error: Layer {layer_name} not found")

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.activations = {}

    def save_feature_map(self, layer_name, raw_image_tensor, batch_idx, channel_idx=None, alpha=0.5):
        """
        将特征图平滑后叠加在原图上
        Args:
            layer_name: 目标层名称
            raw_image_tensor: [3, 224, 224] 的原始图像 Tensor (单个样本)
            batch_idx: 当前 batch 索引
            channel_idx: None(平均所有通道) 或 int(指定通道)
            alpha: 热力图的透明度 (0.0~1.0)，越大越红，越小越能看清底图
        """
        if layer_name not in self.activations:
            return

        # 1. 获取特征 [Batch, 197, 768]
        feat = self.activations[layer_name]
        
        # 取 Batch 中的第 0 个样本，并去掉 CLS token
        feat = feat[0] 
        if feat.shape[0] == 197:
            feat = feat[1:, :] # [196, 768]
            
        # Reshape: [196, 768] -> [14, 14, 768]
        L, C = feat.shape
        H = int(np.sqrt(L)) # 14
        feat_spatial = feat.view(H, H, C)

        # 2. 处理通道 (平均 或 单选)
        if channel_idx is None:
            # 平均所有通道，代表“总体关注度”
            heatmap_data = torch.mean(feat_spatial, dim=2) # [14, 14]
            suffix = "avg"
        else:
            if channel_idx >= C: return
            heatmap_data = feat_spatial[:, :, channel_idx] # [14, 14]
            suffix = f"ch{channel_idx}"

        # 3. 上采样 (Upsample) - 关键步骤！
        # 将 14x14 插值放大到 224x224
        # mode='bilinear' 会让格子消失，变得平滑
        heatmap_tensor = heatmap_data.unsqueeze(0).unsqueeze(0) # [1, 1, 14, 14]
        heatmap_resized = F.interpolate(
            heatmap_tensor, size=(224, 224), mode='bilinear', align_corners=False
        )
        heatmap_np = heatmap_resized.squeeze().cpu().numpy() # [224, 224]

        # 4. 热力图归一化与着色
        # 归一化到 0-255
        heatmap_norm = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)
        heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)
        # 应用伪彩色 (JET 是经典的红蓝热力图)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # 5. 处理原始底图
        # raw_image_tensor: [3, 224, 224]
        # 假设已经反归一化到 [0, 1] 范围，如果没有，这里需要自行处理 Mean/Std
        if raw_image_tensor.dim() == 4:
            # raw_image_tensor shape: [C, T, H, W]
            # 取 T=0 (第一帧) 或者 T=raw_image_tensor.shape[1]//2 (中间帧)
            raw_image_tensor = raw_image_tensor[:, 0, :, :]
        img_np = raw_image_tensor.permute(1, 2, 0).cpu().numpy() # [224, 224, 3]
        
        # 简单鲁棒性处理：强制拉伸到 0-1 之间显示，防止过暗或过亮
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        
        img_uint8 = (img_np * 255).astype(np.uint8)
        # RGB 转 BGR (OpenCV 格式)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

        # 6. 叠加 (Overlay)
        # formula: src1 * alpha + src2 * beta + gamma
        overlay = cv2.addWeighted(heatmap_color, alpha, img_bgr, 1 - alpha, 0)

        # 7. 保存
        save_name = f"{layer_name.replace('.','_')}_b{batch_idx}_{suffix}_overlay.png"
        save_path = os.path.join(self.save_root, save_name)
        cv2.imwrite(save_path, overlay)
        # print(f"Saved overlay: {save_path}")

class CLIPSimilarityMap:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """
        修改点：
        不能 Hook 在 ln_post，因为那里的 Spatial Token 梯度为 0。
        必须 Hook 在最后一个 Block 的 ln_1 (Attention 之前)，
        这里的 Spatial Token 正在通过 Attention 机制向 CLS Token 传递信息。
        """
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # 获取 visual 模型
        if hasattr(self.model, 'module'):
            visual = self.model.module.visual
        else:
            visual = self.model.visual

        # --- 关键修改 ---
        # 目标层：Transformer 最后一层 ResBlock 的 ln_1
        target_layer = visual.transformer.resblocks[-1].ln_1
        # ----------------

        self.hooks.append(target_layer.register_forward_hook(forward_hook))
        self.hooks.append(target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def get_text_embedding(self, text_prompts):
        """
        获取特定文本的 Embedding
        """
        # 使用模型自带的 tokenizer
        tokenizer = self.model.module.tokenizer if hasattr(self.model, 'module') else self.model.tokenizer
        device = next(self.model.parameters()).device
        
        tokenized = []
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
            
        for s in text_prompts:
            # 这里的 context_length 需要与模型定义一致，通常是 77
            context_length = 77 
            ids = [tokenizer.encoder["<|startoftext|>"]] + \
                  tokenizer.encode(s)[:context_length - 2] + \
                  [tokenizer.encoder["<|endoftext|>"]]
            pad_len = context_length - len(ids)
            ids = ids + [0] * pad_len
            tokenized.append(ids)
            
        text_input = torch.tensor(tokenized).long().to(device)
        
        with torch.no_grad():
            text_features, _ = self.model.module.encode_text(text_input) if hasattr(self.model, 'module') else self.model.encode_text(text_input)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        return text_features

    def generate_heatmap(self, image_tensor, text_prompt_embedding):
        """
        生成热力图
        image_tensor: [B, C, T, H, W] (Video) 或 [B, C, H, W] (Image)
        text_prompt_embedding: [1, D] 目标文本的特征向量
        """
        self.model.eval()
        self.model.zero_grad()
        
        # 1. 前向传播图像
        # 确保输入需要梯度的
        image_tensor.requires_grad = True
        
        # 调用 model.encode_image 获取图像特征
        # 注意：这里我们只关心输出的 image_features (Global CLS Embedding)
        # 根据你的代码: output = x, x1.view(B,T,-1), x2, x1
        # 我们需要最后的 x1 (Projected CLS Token) 用于计算相似度
        if hasattr(self.model, 'module'):
            _, _, _, image_features_raw = self.model.module.encode_image(image_tensor)
        else:
            _, _, _, image_features_raw = self.model.encode_image(image_tensor)

        # 归一化图像特征 (CLIP 必须步骤)
        image_features = image_features_raw / image_features_raw.norm(dim=-1, keepdim=True)

        # 2. 计算相似度 (Dot Product)
        # [B*T, D] @ [D, 1] -> [B*T, 1]
        # 这里假设 image_tensor 已经被 flatten 或者是 batch 处理过的
        # 如果是视频 [B, C, T, H, W]，通常 ViT 内部会 flatten 成 [B*T, ...]
        similarity = image_features @ text_prompt_embedding.t()
        
        # 3. 反向传播
        # 我们最大化这个相似度，求梯度
        target_score = similarity.sum()
        target_score.backward()

        # 4. 生成 Grad-CAM
        # Gradients: [BT, L, D]
        # Activations: [BT, L, D]
        grads = self.gradients
        acts = self.activations
        
        # 处理序列长度，去除 CLS Token (Index 0)
        # L = H*W + 1
        grads_spatial = grads[:, 1:, :] 
        acts_spatial = acts[:, 1:, :]
        
        # Global Average Pooling on Gradients -> Weights
        # [BT, N_patches, D] -> [BT, 1, D]
        # weights = torch.mean(grads_spatial, dim=1, keepdim=True)
        # cam = torch.sum(weights * acts_spatial, dim=-1)

        # 新逻辑，使用layer cam公式
        cam = (grads_spatial * acts_spatial).sum(dim=-1) # [BT, N_patches]
        # ReLU (只保留正向贡献)
        cam = F.relu(cam)
        
        # Reshape to 2D
        # N_patches = H * W
        BT, N_patches = cam.shape
        H_feat = int(np.sqrt(N_patches))
        cam = cam.view(BT, H_feat, H_feat)
        
        # 归一化到 0-1 (Per sample normalization)
        cam_min = cam.view(BT, -1).min(dim=-1, keepdim=True)[0].view(BT, 1, 1)
        cam_max = cam.view(BT, -1).max(dim=-1, keepdim=True)[0].view(BT, 1, 1)

        valid_mask = (cam_max > 1e-7).float()
        denominator = cam_max - cam_min + 1e-8
        cam = (cam - cam_min) / denominator
        
        # 应用掩码：如果原本数值极小，归一化后变成全黑，而不是全噪点
        cam = cam * valid_mask

        # cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        
        return cam.detach() # [BT, 14, 14]

    def save_visualization(self, image_tensor, text_prompt, save_path, alpha=0.5):
        """
        端到端：输入图片和文字，保存热力图叠加结果
        """
        # 1. 准备文本特征
        text_emb = self.get_text_embedding(text_prompt)
        
        # 2. 生成 CAM
        heatmap_low_res = self.generate_heatmap(image_tensor, text_emb)
        
        # 3. 处理图像用于保存
        # 假设 image_tensor 是归一化过的，需要反归一化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(image_tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(image_tensor.device)
        
        # 处理 Video Tensor [B, C, T, H, W] -> 取第一帧 或 Flatten
        if image_tensor.dim() == 5:
            # 取 Batch 0, Frame 0 演示
            img_disp = image_tensor[0, :, 0, :, :] # [C, H, W]
            heatmap_disp = heatmap_low_res[0] # [14, 14] (对应 Frame 0)
        elif image_tensor.dim() == 4:
            img_disp = image_tensor[0]
            heatmap_disp = heatmap_low_res[0]
            
        img_disp = img_disp * std[0,:,0,0].view(3,1,1) + mean[0,:,0,0].view(3,1,1)
        
        # 4. 上采样 Heatmap
        H, W = img_disp.shape[1], img_disp.shape[2]
        heatmap_disp = heatmap_disp.unsqueeze(0).unsqueeze(0) # [1, 1, 14, 14]
        heatmap_high_res = F.interpolate(heatmap_disp, size=(H, W), mode='bilinear', align_corners=False)
        heatmap_np = heatmap_high_res.squeeze().cpu().numpy()
        
        # 5. 叠加显示
        img_np = img_disp.permute(1, 2, 0).detach().cpu().numpy()
        img_np = np.clip(img_np, 0, 1)
        img_uint8 = (img_np * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        
        heatmap_uint8 = (heatmap_np * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        overlay = cv2.addWeighted(heatmap_color, alpha, img_bgr, 1 - alpha, 0)
        
        cv2.imwrite(save_path, overlay)
        # print(f"Saved CLIP-GradCAM to {save_path}")

    


class TemporalDynamicsVisualizer:
    def __init__(self, model, save_root):
        self.model = model
        self.save_root = os.path.join(save_root, 'temporal_dynamics_video_level')
        os.makedirs(self.save_root, exist_ok=True)
        
        # 缓存
        self.all_coords_clips = [] 
        self.all_pe_clips = []      # [新增] 缓存 Position Embeddings
        self.hooks = []

    def register_hooks(self):
        def hook_fn(module, input, output):
            x = input[0] 
            T = module.T
            BT = x.shape[0]
            B = BT // T
            
            # 1. 计算物理坐标 [B, T]
            coords = module.get_physics_coordinate(x, B, T)
            self.all_coords_clips.append(coords.detach().cpu().numpy()) # 用于画曲线
            
            # 2. 【关键】缩放坐标 (Rescaling) 防止相似度全为 1
            # 假设该 Clip 的物理时长应当撑满整个序列长度 T
            # 加上 1e-6 防止除以零
            max_val = coords.max(dim=1, keepdim=True)[0]
            scale_factor = T / (max_val + 1e-6)
            coords_scaled = coords * scale_factor
            
            # 3. 生成动态位置编码 [B, T, C]
            Ca = module.fc1.out_features
            pos_emb = module.get_physics_pos_embed(coords_scaled, Ca)
            
            # 4. 缓存 PE
            self.all_pe_clips.append(pos_emb.detach().cpu()) # 存 Tensor 以便后续 GPU 计算

        if hasattr(self.model, 'module'):
            target_module = self.model.module.visual.t_adapter
        else:
            target_module = self.model.visual.t_adapter
            
        self.hooks.append(target_module.register_forward_hook(hook_fn))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.all_coords_clips = []
        self.all_pe_clips = []
    
    def plot_full_video_analysis(self, video_name="video_01", window_size=20):
        """
        同时绘制【运动曲线】和【相似度矩阵】
        window_size: 矩阵展示的帧数窗口，避免全视频太长导致矩阵像素过小看不清
        """
        if not self.all_coords_clips:
            return

        # --- 1. 处理曲线数据 ---
        full_coords = np.concatenate(self.all_coords_clips, axis=0) # [Total_Clips, T]
        # 计算瞬时运动代价 (Motion Cost)
        costs = np.diff(full_coords, axis=1, prepend=0)
        avg_motion_cost = np.mean(costs, axis=1) # [Total_Clips]
        
        # 平滑曲线
        smooth_curve = avg_motion_cost
        if len(avg_motion_cost) > 5:
            win = 5
            smooth_curve = np.convolve(avg_motion_cost, np.ones(win)/win, mode='same')

        # --- 2. 处理相似度矩阵数据 ---
        # 拼接所有 PE: [Total_Clips, T, C] -> Flatten -> [Total_Frames, C]
        # 注意：这里简单的 Flatten 会导致重叠帧重复出现。
        # 如果只是为了定性展示，我们取每个 Clip 的第一帧或中心帧代表该时刻
        # 假设我们取每个 Clip 的平均 PE 向量作为该 Clip 的代表
        
        all_pe = torch.cat(self.all_pe_clips, dim=0) # [Total_Clips, T, C]
        # 取每个 Clip 的平均特征作为该时刻的 Embedding
        clip_pe = torch.mean(all_pe, dim=1) # [Total_Clips, C]
        
        # 截取前 window_size 个时刻进行展示 (保证矩阵清晰度)
        display_len = min(len(clip_pe), window_size)
        pe_subset = clip_pe[:display_len]          # [display_len, C]
        curve_subset = smooth_curve[:display_len]  # [display_len] (用于对齐)

        # 计算余弦相似度矩阵
        # Normalize
        pe_subset = F.normalize(pe_subset, p=2, dim=1)
        sim_matrix = torch.matmul(pe_subset, pe_subset.T).numpy() # [L, L]

        # --- 3. 联合绘图 (上下对齐) ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True,
                                       gridspec_kw={'height_ratios': [1, 2]})
        
        # 上图：运动曲线
        frames = np.arange(display_len)
        ax1.plot(frames, curve_subset, 'r-', linewidth=2, label='Motion Cost (Smoothed)')
        ax1.set_ylabel("Motion Cost\n(High=Fast, Low=Static)")
        ax1.set_title(f"Joint Analysis: Motion Curve vs PE Similarity ({video_name})")
        ax1.legend(loc='upper right')
        ax1.grid(True, linestyle='--', alpha=0.5)
        # 填充背景色以辅助看图：波峰位置标红，波谷位置标蓝（可选）
        
        # 下图：相似度矩阵
        sns.heatmap(sim_matrix, ax=ax2, cmap='viridis', square=False, vmin=0.0, vmax=1.0, cbar_kws={'label': 'Cosine Similarity'})
        ax2.set_xlabel("Timeline (Clip Index)")
        ax2.set_ylabel("Timeline (Clip Index)")
        
        plt.tight_layout()
        save_path = os.path.join(self.save_root, f'{video_name}_joint_analysis.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved joint analysis to {save_path}")
        
        self.all_coords_clips = []
        self.all_pe_clips = []

class TSNEVisualizer:
    def __init__(self, model, save_root):
        self.model = model
        # 保存到专门的文件夹
        self.save_root = os.path.join(save_root, 'tsne_results')
        os.makedirs(self.save_root, exist_ok=True)
        
        # 数据缓存
        self.features = []      # 存特征 [N, D]
        self.motion_costs = []  # 存运动代价 [N]
        self.labels = []        # 存类别标签 [N]
        self.row_indices = []   # 存行号 [N] (用于代替 Frame Index)

    def extract_features(self, dataloader, max_batches=50):
        """
        核心函数：从 DataLoader 中提取特征并处理维度
        """
        self.model.eval()
        self.features = []
        self.motion_costs = []
        self.labels = []
        self.row_indices = []
        
        global_row_idx = 0 
        
        print(f"Start extracting features for T-SNE (Max batches: {max_batches})...")
        
        # --- 【修改点1】开启 Autocast 上下文 ---
        # 这会自动处理 Half(CLIP) 和 Float(Adapter) 之间的类型转换
        with torch.no_grad(), torch.cuda.amp.autocast():
            for i, (data, label) in enumerate(dataloader):
                if i >= max_batches: break
                
                data = data.cuda()
                
                # 处理 6 维数据
                if data.dim() == 6:
                    data = data.flatten(0, 1) 
                
                # 检查数据形状
                try:
                    B, C, T, H, W = data.shape
                except ValueError:
                    continue

                if hasattr(self.model, 'module'):
                    visual = self.model.module.visual
                else:
                    visual = self.model.visual
                
                # Flatten B and T: [B*T, C, H, W]
                x = data.permute(0, 2, 1, 3, 4).flatten(0, 1) 
                
                # --- 【修改点2】删除手动类型转换 ---
                # x = x.type(visual.conv1.weight.dtype) <--- 删除这行
                # Autocast 会自动处理 conv1 (Half) 遇到 Float 输入的情况
                
                x = visual.conv1(x)
                x = x.flatten(-2).permute(0, 2, 1)
                
                # 添加 Class Token 和 Pos Embed
                # 确保 class_embedding 类型匹配 (Autocast 环境下通常不需要手动 to，但保留也无妨)
                cls_token = visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
                x = torch.cat([cls_token, x], dim=1)
                x = x + visual.positional_embedding.to(x.dtype)
                
                x = visual.ln_pre(x)
                
                # 这里进入 Transformer -> Adapter
                # Autocast 会在这里发挥作用：
                # 输入 x 是 Half (来自 ln_pre), Adapter 权重是 Float
                # Autocast 会临时将 x 转为 Float 传给 Adapter
                x = visual.transformer(x)
                
                x = visual.ln_post(x)
                
                if visual.proj is not None:
                    x = x[:, 0, :] @ visual.proj 
                else:
                    x = x[:, 0, :]

                # --- 2. 提取 Adapter 特征 (Drift-Aware) ---
                adapter = visual.t_adapter
                
                # 恢复 B, T 维度
                x_temporal = x.view(B, T, -1)
                
                # 计算 Motion Cost
                # 注意：get_physics_coordinate 内部可能有 detach，autocast 依然有效
                coords = adapter.get_physics_coordinate(x_temporal.flatten(0, 1), B, T)
                row_cost = torch.diff(coords, dim=1).mean(dim=1) 
                
                # 获取最终特征
                x_adapted = adapter(x) 
                
                # 聚合
                x_adapted = x_adapted.view(B, T, -1).mean(dim=1) 
                
                # 存入缓存
                self.features.append(x_adapted.cpu().numpy().astype(np.float32)) # 显式转 float32 存numpy
                self.motion_costs.append(row_cost.cpu().numpy().astype(np.float32))
                self.labels.append(label.cpu().numpy())
                self.row_indices.append(np.arange(global_row_idx, global_row_idx + B))
                
                global_row_idx += B

        # 拼接
        self.features = np.concatenate(self.features, axis=0)        
        self.motion_costs = np.concatenate(self.motion_costs, axis=0) 
        self.labels = np.concatenate(self.labels, axis=0)           
        self.row_indices = np.concatenate(self.row_indices, axis=0)  
        
        print(f"Extraction Done. Total rows: {len(self.features)}")

    def plot_comet_tsne(self, video_id="video_sample"):
        """
        绘制 T-SNE 彗星图 (大小=稳定性, 颜色=时间)
        """
        if len(self.features) == 0: return
        print("Running T-SNE fitting...")
        
        # 1. 降维
        tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42)
        X_2d = tsne.fit_transform(self.features)
        
        # 2. 准备绘图数据
        # 按行号排序 (确保时间顺序)
        sort_idx = np.argsort(self.row_indices)
        X_sorted = X_2d[sort_idx]
        costs_sorted = self.motion_costs[sort_idx]
        
        # 计算点的大小 (Motion Cost 越小 -> 越稳 -> 点越大)
        # 归一化 cost
        cost_norm = (costs_sorted - costs_sorted.min()) / (costs_sorted.max() - costs_sorted.min() + 1e-6)
        sizes = 150 * (1 - cost_norm) + 30 # 范围 30~180
        
        plt.figure(figsize=(10, 8))
        
        # 3. 画轨迹线 (淡色背景)
        plt.plot(X_sorted[:, 0], X_sorted[:, 1], 'k-', alpha=0.15, linewidth=1)
        
        # 4. 画彗星点
        scatter = plt.scatter(X_sorted[:, 0], X_sorted[:, 1], 
                              c=np.arange(len(X_sorted)), # 颜色=时间进度
                              s=sizes,                    # 大小=物理稳定性
                              cmap='viridis', alpha=0.8, edgecolors='none')
        
        # 标注
        plt.text(X_sorted[0,0], X_sorted[0,1], "START", color='green', fontweight='bold')
        plt.text(X_sorted[-1,0], X_sorted[-1,1], "END", color='red', fontweight='bold')
        
        cbar = plt.colorbar(scatter)
        cbar.set_label('Timeline (Row Index)')
        
        plt.title(f"T-SNE Comet Plot: {video_id}\n(Node Size = Stability)")
        plt.savefig(os.path.join(self.save_root, f'{video_id}_comet.png'))
        plt.close()
        print(f"Saved Comet Plot to {self.save_root}")

    def plot_barcode(self, video_id="video_sample"):
        """
        绘制时序条形码
        """
        if len(self.motion_costs) == 0: return
        
        # 按时间排序
        sort_idx = np.argsort(self.row_indices)
        costs = self.motion_costs[sort_idx]
        
        # 归一化
        costs_norm = (costs - costs.min()) / (costs.max() - costs.min() + 1e-6)
        
        # 扩展成图片 [50, N]
        barcode = np.tile(costs_norm, (50, 1))
        
        plt.figure(figsize=(12, 3))
        plt.imshow(barcode, cmap='plasma', aspect='auto', interpolation='nearest')
        plt.axis('off') # 关掉坐标轴
        plt.title(f"Scanning Rhythm Barcode: {video_id}")
        
        # 手动加个 colorbar
        cbar = plt.colorbar(orientation='horizontal', pad=0.1)
        cbar.set_label('Motion Cost (Blue=Static, Yellow=Fast)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_root, f'{video_id}_barcode.png'))
        plt.close()
        print(f"Saved Barcode to {self.save_root}")

    def plot_motion_manifold(self, video_id="video_sample"):
        """
        方案 C：运动状态流形图 (Motion State Manifold Clustering)
        
        侧重点：不关注时间顺序，只关注【物理状态】对特征分布的影响。
        核心逻辑：用 Motion Cost 为点着色。
        预期效果：
          - 蓝色/紫色点（低 Cost，静止观察）：聚集成紧密的核心簇 (Core Clusters)。
          - 黄色/亮色点（高 Cost，快速漂移）：散落在 T-SNE 图的边缘或呈稀疏带状 (Periphery/Outliers)。
        """
        if len(self.features) == 0: return
        print("Running Motion Manifold T-SNE...")

        # 1. T-SNE 降维
        # perplexity 稍微调大一点 (e.g., 40)，有助于让核心簇更紧凑，离群点更明显
        tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42, perplexity=40)
        X_2d = tsne.fit_transform(self.features)
        
        # 2. 准备颜色数据 (Motion Cost)
        # 无需排序，直接对应
        costs = self.motion_costs
        
        # 归一化 (Min-Max Normalization) 以便映射颜色
        # 加上 1e-6 防止除以0
        norm_costs = (costs - costs.min()) / (costs.max() - costs.min() + 1e-6)
        
        plt.figure(figsize=(10, 8))
        
        # 3. 绘制散点图
        # cmap='plasma': 蓝色(低Cost) -> 红色 -> 黄色(高Cost)
        # alpha=0.7: 让重叠的核心区域颜色更深，体现密度
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], 
                              c=norm_costs, 
                              cmap='plasma', 
                              s=80,          # 固定大小，只看颜色分布
                              alpha=0.7, 
                              edgecolors='none')
        
        # 4. 添加标注与解释
        cbar = plt.colorbar(scatter)
        cbar.set_label('Motion Cost / Drift Magnitude\n(Blue=Stable Fixation, Yellow=Rapid Drift)')
        
        plt.title(f"Motion State Manifold: {video_id}\n(Drift Frames Pushed to Periphery)")
        plt.xlabel("Feature Dimension 1")
        plt.ylabel("Feature Dimension 2")
        
        # 去掉坐标刻度，只看拓扑结构
        plt.xticks([])
        plt.yticks([])
        
        plt.tight_layout()
        save_path = os.path.join(self.save_root, f'{video_id}_motion_manifold.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved Motion Manifold Plot to {save_path}")

    def plot_lda_separation(self, video_id="video_sample", threshold=0.5):
        """
        [美化版] 方案 D: LDA 线性判别分析 KDE 密度图
        使用 Seaborn 绘制平滑的概率密度分布，更适合 Paper 展示。
        """
        if len(self.features) == 0: return
        print("Running LDA for Drift Separation...")
        
        # 1. 准备数据 (同前)
        costs = self.motion_costs
        norm_costs = (costs - costs.min()) / (costs.max() - costs.min() + 1e-6)
        drift_labels = (norm_costs > threshold).astype(int)
        
        if len(np.unique(drift_labels)) < 2:
            print("Skipping LDA: Not enough variance.")
            return

        # 2. LDA 降维
        lda = LinearDiscriminantAnalysis(n_components=1)
        X_lda = lda.fit_transform(self.features, drift_labels).flatten() # 展平为 1D 数组

        # 3. 绘制美化版 KDE 图
        plt.figure(figsize=(10, 6))
        
        # 使用 seaborn 的 kdeplot (Kernel Density Estimate)
        # fill=True: 填充颜色
        # common_norm=False: 各自归一化，方便对比形状
        # palette: 颜色盘
        import seaborn as sns
        
        # 准备 DataFrame 方便 seaborn 绘图 (可选，但直接传数组也行)
        data = {
            'Discriminant Value': X_lda,
            'State': ['Stable Phase' if l == 0 else 'Drift Event' for l in drift_labels]
        }
        
        # 绘图风格设置
        sns.set_style("whitegrid") # 白色网格背景，很学术
        
        # 绘制
        ax = sns.kdeplot(
            data=data, 
            x='Discriminant Value', 
            hue='State', 
            fill=True, 
            common_norm=False, 
            palette={'Stable Phase': '#4361EE', 'Drift Event': '#F72585'}, # 赛博朋克风配色或经典蓝红
            alpha=0.5, 
            linewidth=2
        )
        
        # 4. 细节装饰
        plt.title(f"Linear Separability of Motion States\n(Projected by LDA)", fontsize=14, fontweight='bold')
        plt.xlabel("Discriminant Feature Space (LDA Axis)", fontsize=12)
        plt.ylabel("Probability Density", fontsize=12)
        
        # 去掉上方和右侧的边框 (Spines)
        sns.despine()
        
        # 添加显著性标记 (Optional)
        # 计算两个峰值的中心距离
        mean_stable = X_lda[drift_labels==0].mean()
        mean_drift = X_lda[drift_labels==1].mean()
        
        # 画个箭头标注 Gap
        plt.annotate(
            '', xy=(mean_drift, 0.05), xytext=(mean_stable, 0.05),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5)
        )
        plt.text((mean_stable + mean_drift)/2, 0.06, "Clear Margin", ha='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
        save_path = os.path.join(self.save_root, f'{video_id}_lda_kde_beautiful.png')
        plt.savefig(save_path, dpi=300) # 300 DPI 高清保存
        plt.close()
        print(f"Saved Beautiful LDA Plot to {save_path}")
    
    def plot_lda_ridge_by_patient(self, num_patients=5, frames_per_patient=None, threshold=0.5):
        """
        方案 E: Multi-Patient Ridge Plot (山脊图)
        
        核心逻辑：
        1. 将巨大的特征集合切分为多个子集 (代表不同病人/视频)。
        2. 对每个子集分别画 LDA 分布图。
        3. 垂直堆叠这些图，形成"峰峦叠嶂"的效果。
        
        参数:
            num_patients: 想要展示多少个病人(层数)。
            frames_per_patient: 每个病人大概多少帧？如果不填，自动平分。
        """
        if len(self.features) == 0: return
        print("Generating Ridge Plot (Joyplot) for multiple patients...")

        # 1. 准备数据
        # 假设数据是按顺序提取的，我们将其切分为 num_patients 段
        total_frames = len(self.features)
        if frames_per_patient is None:
            chunk_size = total_frames // num_patients
        else:
            chunk_size = frames_per_patient

        # 2. 全局 LDA 训练 (Global Projector)
        # 关键点：我们要证明同一个 Adapter (同一套投影参数) 对所有病人都有效
        # 所以用所有数据训练 LDA
        costs = self.motion_costs
        norm_costs = (costs - costs.min()) / (costs.max() - costs.min() + 1e-6)
        global_labels = (norm_costs > threshold).astype(int)
        
        lda = LinearDiscriminantAnalysis(n_components=1)
        X_lda_global = lda.fit_transform(self.features, global_labels).flatten()

        # 3. 准备绘图数据 DataFrame
        import pandas as pd
        import seaborn as sns
        
        plot_data = []
        
        for i in range(num_patients):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, total_frames)
            if start >= total_frames: break
            
            # 取出该病人的数据
            sub_X = X_lda_global[start:end]
            sub_labels = global_labels[start:end]
            
            # 只有当该病人既有稳定又有漂移时，画出来才有意义
            # 为了美观，我们只存数据，让 seaborn 处理
            for val, label in zip(sub_X, sub_labels):
                state = 'Stable' if label == 0 else 'Drift'
                # 记录数据：值，状态，病人ID
                plot_data.append({
                    'LDA Value': val,
                    'State': state,
                    'Patient': f'Patient {i+1}'
                })
        
        df = pd.DataFrame(plot_data)

        # 4. 绘制 Ridge Plot
        # 使用 Seaborn 的 FacetGrid 实现层叠效果
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        
        # 创建网格：每一行是一个 Patient
        g = sns.FacetGrid(df, row="Patient", hue="State", aspect=6, height=1.2, 
                          palette={'Stable': '#4361EE', 'Drift': '#F72585'})

        # 画密度图 (KDE)
        # fill=True 填充, alpha=0.8 不透明度
        g.map_dataframe(sns.kdeplot, x="LDA Value", fill=True, alpha=0.6, linewidth=1.5)
        
        # 画轮廓线 (让山峰更清晰)
        g.map_dataframe(sns.kdeplot, x="LDA Value", color="white", linewidth=2)

        # 5. 美化细节
        # 让子图之间稍微重叠，产生 3D 遮挡感
        g.figure.subplots_adjust(hspace=-0.4)

        # 移除子图的标题，把 Patient ID 写在左边
        g.set_titles("")
        g.set(yticks=[], ylabel="")
        g.despine(bottom=True, left=True)

        # 在每一行左侧添加 Patient 标签
        for ax, label in zip(g.axes.flat, g.row_names):
            # 将文字写在坐标轴坐标系中 (0, 0) 的左侧
            ax.text(-0.02, 0.1, label, fontweight="bold", color="black",
                    ha="right", va="center", transform=ax.transAxes)
            
            # 画一条基准线
            ax.axhline(0, lw=1, clip_on=False, color="gray")

        # 添加总标题和图例
        plt.suptitle("Cross-Patient Robustness: Motion State Separability", 
                     fontsize=14, fontweight='bold', y=0.98)
        
        # 自定义图例 (因为 FacetGrid 的图例有时候不好调)
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='#4361EE', lw=4, alpha=0.6),
                        Line2D([0], [0], color='#F72585', lw=4, alpha=0.6)]
        plt.legend(custom_lines, ['Stable Phase', 'Drift Event'], 
                   loc='upper right', bbox_to_anchor=(1, 1.5))

        # 保存
        save_path = os.path.join(self.save_root, 'lda_ridge_plot_multipatient.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved Ridge Plot to {save_path}")
    

    def plot_lda_raincloud(self, video_id="video_sample", threshold=0.5, max_points=2000):
        """
        方案 G: Raincloud Plot (雨云图) - 顶级论文可视化标准
        
        组成元素：
        1. Cloud (Upper): 半小提琴图，展示分布形状。
        2. Umbrella (Middle): 箱线图，展示统计四分位。
        3. Rain (Lower): 抖动散点图，展示微观数据。
        
        解决痛点：
        通过 Jitter (抖动) 和下采样，让底部的"条形"变成清晰的"雨滴"，避免视觉混乱。
        """
        if len(self.features) == 0: return
        print("Generating Raincloud Plot...")
        
        # 1. 准备数据
        costs = self.motion_costs
        norm_costs = (costs - costs.min()) / (costs.max() - costs.min() + 1e-6)
        drift_labels = (norm_costs > threshold).astype(int)
        
        if len(np.unique(drift_labels)) < 2:
            print("Skipping: Not enough variance.")
            return

        # LDA 投影
        lda = LinearDiscriminantAnalysis(n_components=1)
        X_lda = lda.fit_transform(self.features, drift_labels).flatten()
        
        # 2. 构建 DataFrame 便于绘图
        import pandas as pd
        import seaborn as sns
        
        df = pd.DataFrame({
            'LDA Score': X_lda,
            'State': ['Stable Phase' if l == 0 else 'Drift Event' for l in drift_labels]
        })
        
        # 【关键优化】数据下采样 (Subsampling)
        # 如果数据点超过几万个，画散点依然会糊。
        # 我们随机抽取 max_points 个点来画"雨"，但用全部数据画"云"和"伞"。
        if len(df) > max_points:
            df_rain = df.sample(n=max_points, random_state=42)
        else:
            df_rain = df

        # 3. 绘图设置
        # 创建画布，使用 GridSpec 来精确控制 Cloud/Rain 的比例
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        
        # 定义配色 (Nature/Science 风格)
        palette = {'Stable Phase': '#4361EE', 'Drift Event': '#F72585'}
        
        # --- Layer 1: Cloud (Violin Plot) ---
        # inner=None 去掉内部原本的箱线图，因为我们要自己画更好看的
        # cut=0 让分布图在数据范围内截断，不无限延伸
        sns.violinplot(data=df, x="LDA Score", y="State", hue="State",
                       palette=palette, inner=None, orient="h", 
                       alpha=0.4, linewidth=0, scale="width", ax=ax, zorder=1)
        
        # --- Layer 2: Umbrella (Box Plot) ---
        # 放在中间，窄一点 (width)，深色边框
        sns.boxplot(data=df, x="LDA Score", y="State", 
                    width=0.15, color="black", # 黑色箱体
                    boxprops={'facecolor':'none', 'edgecolor':'#333333', 'linewidth':1.5},
                    whiskerprops={'color':'#333333', 'linewidth':1.5},
                    medianprops={'color':'white', 'linewidth':2}, # 白色中位数线
                    showfliers=False, # 不显示离群点，由散点图展示
                    ax=ax, zorder=2)
        
        # --- Layer 3: Rain (Strip Plot with Jitter) ---
        # 使用下采样后的数据 df_rain
        # jitter=True 让点上下抖动，避免重叠
        sns.stripplot(data=df_rain, x="LDA Score", y="State", hue="State", 
                      palette=palette, size=2, alpha=0.6, jitter=0.2, 
                      orient="h", ax=ax, zorder=0)

        # 4. 美化与细节
        # 调整 Y 轴标签位置，让图表看起来不像传统的 Violin Plot
        ax.set_ylabel("")
        ax.set_xlabel("Discriminant Feature Space (LDA Projection)", fontsize=12, fontweight='bold')
        
        # 移除四周的边框
        sns.despine(left=True, bottom=False)
        
        # 添加一些文字说明
        # 计算统计量
        means = df.groupby('State')['LDA Score'].mean()
        for i, state in enumerate(['Stable Phase', 'Drift Event']):
            # 注意：seaborn 的 y 轴顺序可能不同，通常 Stable 是 0, Drift 是 1
            # 我们需要获取 y 坐标。stripplot 默认是 0, 1
            # 简单起见，直接在图上方标注 Margin
            pass

        margin = abs(means['Drift Event'] - means['Stable Phase'])
        plt.suptitle(f"LDA Separation Raincloud Plot\n(Clear Margin $\Delta \\approx {margin:.2f}$)", 
                     fontsize=14, fontweight='bold', y=0.98)
        
        # 自定义图例 (解释图表元素)
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=palette['Stable Phase'], lw=4, alpha=0.5, label='Stable Distribution (KDE)'),
            Line2D([0], [0], color=palette['Drift Event'], lw=4, alpha=0.5, label='Drift Distribution (KDE)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', label='Individual Frames (Subsampled)', markersize=6),
            Line2D([0], [0], color='#333333', lw=2, label='Statistical Quartiles (Box)')
        ]
        ax.legend(handles=legend_elements, loc='lower right', frameon=True, fontsize=9)

        plt.tight_layout()
        save_path = os.path.join(self.save_root, f'{video_id}_lda_raincloud.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved Raincloud Plot to {save_path}")


class TemporalAttentionVisualizer:
    """
    专门用于绘制 UnifiedDriftAwareAdapter 的 T*T 时序注意力热图。
    适用于帧按行组织、采样率为 8帧*2stride 的场景。
    """
    def __init__(self, model, save_root):
        """
        Args:
            model: 你的 CLIP 或 VisionTransformer 模型实例
            save_root: 图片保存的根目录
        """
        self.model = model
        self.save_dir = os.path.join(save_root, 'temporal_attention_maps')
        os.makedirs(self.save_dir, exist_ok=True)

    def _get_adapter(self):
        """
        自动寻找模型中的 UnifiedDriftAwareAdapter 模块。
        兼容 DDP (DistributedDataParallel) 和直接模型。
        """
        # 1. 解开 DDP 包装 (如果存在)
        model_ptr = self.model.module if hasattr(self.model, 'module') else self.model

        # 2. 寻找 t_adapter
        # 路径通常是 model.visual.t_adapter (如果 model 是 CLIP)
        # 或者 model.t_adapter (如果 model 只是 VisionTransformer)
        if hasattr(model_ptr, 'visual') and hasattr(model_ptr.visual, 't_adapter'):
            return model_ptr.visual.t_adapter
        elif hasattr(model_ptr, 't_adapter'):
            return model_ptr.t_adapter
        else:
            return None

    def visualize_batch(self, batch_idx, file_names=None):
        """
        绘制当前 Batch 中所有样本的注意力热图。
        
        Args:
            batch_idx (int): 当前 Batch 的索引，用于生成文件名。
            file_names (list, optional): 当前 Batch 对应的原始文件名列表，用于更精确的命名。
        """
        adapter = self._get_adapter()
        if adapter is None:
            print("[Visualizer] Warning: Could not find 't_adapter' in model.")
            return

        if not hasattr(adapter, 'last_attn_weights'):
            print("[Visualizer] Warning: 'last_attn_weights' not found. Please modify forward().")
            return

        # 获取 Attention Weights: [B, T, T]
        # B: 当前 Batch 有多少个 sample (每个 sample 是一组 8 帧序列)
        # T: 时序长度 (8)
        attn_weights = adapter.last_attn_weights.numpy()
        
        B, T, _ = attn_weights.shape

        # 遍历 Batch 中的每一个 Sample 进行绘图
        for b in range(B):
            self._plot_single_map(attn_weights[b], batch_idx, sample_idx=b, T=T, file_name=file_names[b] if file_names else None)

    def _plot_single_map(self, attn_matrix, batch_idx, sample_idx, T, file_name=None):
        """
        绘制单个样本的 T*T 热力图
        """
        plt.figure(figsize=(8, 6))
        
        # 使用 Seaborn 绘制热力图
        # annot=True 会在格子里显示数值，适合 T 较小 (如 8) 的情况
        # cmap='viridis' (黄蓝) 或 'magma' (紫黄) 对比度高
        ax = sns.heatmap(
            attn_matrix, 
            annot=True, 
            fmt=".2f", 
            cmap='viridis', 
            vmin=0.0, 
            vmax=1.0,
            cbar_kws={'label': 'Attention Weight'},
            square=True
        )

        # 设置坐标轴标签
        # X轴: Key Frame (Source) - 被关注的帧
        # Y轴: Query Frame (Target) - 主动去关注的帧
        ax.set_xlabel(f'Key Frame Index (History -> Target)', fontsize=10, fontweight='bold')
        ax.set_ylabel(f'Query Frame Index (History -> Target)', fontsize=10, fontweight='bold')
        
        # 生成时间刻度标签 (例如: T-7, T-6, ..., Target)
        # 既然数据是按行组织，第 0 行是最早的帧，第 T-1 行是最新的目标帧
        tick_labels = [f"t_{i}" for i in range(T)]
        ax.set_xticklabels(tick_labels, rotation=0)
        ax.set_yticklabels(tick_labels, rotation=0)

        # 设置标题
        title_str = f"Temporal Attention (Batch {batch_idx}, Sample {sample_idx})"
        if file_name:
            title_str += f"\n{os.path.basename(file_name)}"
        plt.title(title_str, fontsize=12)

        # 保存图片
        # 命名格式: attn_b{batch}_s{sample}.png
        save_name = f"attn_b{batch_idx}_s{sample_idx}.png"
        if file_name:
            # 如果提供了文件名，使用原始文件名作为前缀，方便对应
            clean_name = os.path.splitext(os.path.basename(file_name))[0]
            save_name = f"{clean_name}_attn.png"
            
        save_path = os.path.join(self.save_dir, save_name)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close() # 关闭画布释放内存
        
        # print(f"Saved attention map: {save_path}")
