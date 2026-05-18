
from typing import Tuple
from collections import OrderedDict
import math
import functools
import pdb
from typing import Tuple, Union
import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import pywt
from simple_tokenizer import SimpleTokenizer

from configs import (
    CLIP_VIT_B16_PATH,
    CLIP_VIT_B32_PATH,
    CLIP_VIT_L14_PATH,
    DWCONV3D_DISABLE_CUDNN,
    )

from typing import Dict, List, Optional, Tuple

class VisionGatedTextAggregator(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        temperature: float = 1.0,
        use_topk: Optional[int] = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.text_dim = text_dim

        self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))

        if vision_dim != text_dim:
            self.proj_v = nn.Linear(vision_dim, text_dim, bias=False)
            nn.init.xavier_uniform_(self.proj_v.weight)
        else:
            self.proj_v = nn.Identity()

        self.use_topk = use_topk
        self.eps = eps

    def forward(self, x_visual: torch.Tensor, category_text_embeddings: torch.Tensor, mode: str = 'gated',num_desc: Optional[int] = 8):
        category_text_embeddings = category_text_embeddings.to(x_visual.device)
        BT, C = x_visual.size()

        if mode == 'mean':

            T_C_fixed = category_text_embeddings.mean(dim=1)  # [C, D_txt]
            return T_C_fixed.unsqueeze(0).expand(BT, -1, -1)
        
        N_class, N_desc, D_text = category_text_embeddings.shape

        v_anchor_text = self.proj_v(x_visual)

        v_anchor_text_norm = F.normalize(v_anchor_text, dim=-1)
        text_embeddings_norm = F.normalize(category_text_embeddings, dim=-1)

        similarity = torch.einsum('bd,cnd->bcn', v_anchor_text_norm, text_embeddings_norm) 
        similarity = similarity / self.temperature.exp()

        if self.use_topk is not None and self.use_topk < N_desc:
            topk_values, topk_indices = torch.topk(similarity, self.use_topk, dim=-1)
            mask = torch.ones_like(similarity) * self.eps 
            mask.scatter_(dim=-1, index=topk_indices, value=1.0)
            similarity = similarity.masked_fill(mask == self.eps, float('-inf'))

        alpha = F.softmax(similarity, dim=-1)

        T_C_dynamic = torch.einsum('bcn,cnd->bcd', alpha, category_text_embeddings)

        return T_C_dynamic

class DWT2d(nn.Module):
    """
    """
    def __init__(self, in_channels, wavelet='haar'):
        super().__init__()
        w = pywt.Wavelet(wavelet)
        dec_lo = torch.tensor(w.dec_lo[::-1], dtype=torch.float32)
        dec_hi = torch.tensor(w.dec_hi[::-1], dtype=torch.float32)

        ll = torch.outer(dec_lo, dec_lo)
        lh = torch.outer(dec_lo, dec_hi)
        hl = torch.outer(dec_hi, dec_lo)
        hh = torch.outer(dec_hi, dec_hi)

        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        
        self.filters = filters.repeat(in_channels, 1, 1, 1)
        

        self.register_buffer('weight', self.filters)
        self.stride = 2

        self.pad = (filters.shape[-1] // 2) - 1 

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        res = F.conv2d(x, self.weight, stride=self.stride, padding=self.pad, groups=C)
        
        res = res.view(B, C, 4, res.shape[2], res.shape[3])
        
        ll = res[:, :, 0, :, :]
        lh = res[:, :, 1, :, :]
        hl = res[:, :, 2, :, :]
        hh = res[:, :, 3, :, :]
        
        return ll, (lh, hl, hh)

class IDWT2d(nn.Module):
    def __init__(self, in_channels, wavelet='haar'):
        super().__init__()
        w = pywt.Wavelet(wavelet)

        rec_lo = torch.tensor(w.rec_lo[::-1], dtype=torch.float32)
        rec_hi = torch.tensor(w.rec_hi[::-1], dtype=torch.float32)

        ll = torch.outer(rec_lo, rec_lo)
        lh = torch.outer(rec_lo, rec_hi)
        hl = torch.outer(rec_hi, rec_lo)
        hh = torch.outer(rec_hi, rec_hi)

        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)

        self.filters = filters.repeat(in_channels, 1, 1, 1)
        
        self.register_buffer('weight', self.filters)
        self.stride = 2
        self.pad = (filters.shape[-1] // 2) - 1

    def forward(self, ll, highs):
        lh, hl, hh = highs
        B, C, H, W = ll.shape

        combined = torch.stack([ll, lh, hl, hh], dim=2).reshape(B, C * 4, H, W)
        
        res = F.conv_transpose2d(combined, self.weight, stride=self.stride, padding=self.pad, groups=C)
        
        return res

class DWTInteractiveAdapter(nn.Module):
    def __init__(self, in_channels, adapter_channels, wavelet='haar'):
        super().__init__()
        self.in_channels = in_channels
        
        self.dwt = DWT2d(adapter_channels, wavelet=wavelet)
        self.idwt = IDWT2d(adapter_channels, wavelet=wavelet)

        self.input_proj = nn.Linear(in_channels, adapter_channels)
        self.output_proj = nn.Linear(adapter_channels, in_channels)
        self.lf_context_conv = nn.Sequential(
            nn.Conv2d(adapter_channels, adapter_channels, kernel_size=3, padding=1, groups=adapter_channels),
            nn.Conv2d(adapter_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.hf_modulation_conv = nn.Sequential(
            nn.Conv2d(adapter_channels * 3, adapter_channels, kernel_size=3, padding=1, groups=adapter_channels),
            nn.GELU(),
            nn.Conv2d(adapter_channels, adapter_channels * 2, kernel_size=1)
        )

        self.act = nn.GELU()
        
        nn.init.constant_(self.input_proj.bias, 0.)
        nn.init.constant_(self.output_proj.bias, 0.)

    def forward(self, x, mode='full'):

        if mode == 'identity':
            return x
        BT, L, C = x.size()
        H = W = int(math.sqrt(L - 1))
        
        cls_token = x[:, :1, :]
        patch_tokens = x[:, 1:, :]
        
        x_mid = self.act(self.input_proj(patch_tokens))
        Ca = x_mid.shape[-1]
        
        x_spatial = x_mid.permute(0, 2, 1).view(BT, Ca, H, W)
        
        pad_h = H % 2
        pad_w = W % 2
        if pad_h > 0 or pad_w > 0:
            x_spatial = F.pad(x_spatial, (0, pad_w, 0, pad_h), mode='reflect')

        F_LL, (F_LH, F_HL, F_HH) = self.dwt(x_spatial)

        if mode == 'lf_only':
            F_LH_new = torch.zeros_like(F_LH)
            F_HL_new = torch.zeros_like(F_HL)
            F_HH_new = torch.zeros_like(F_HH)
            F_LL_new = F_LL  # 不进行高频调制

        elif mode == 'hf_only':
            F_LH_new, F_HL_new, F_HH_new = F_LH, F_HL, F_HH
            F_LL_new = F_LL

        else: # mode == 'full' (Ours)
            F_High_concat = torch.cat([F_LH, F_HL, F_HH], dim=1)
            mask_semantic = self.lf_context_conv(F_LL)
            F_High_new = F_High_concat * mask_semantic
            
            modulation_params = self.hf_modulation_conv(F_High_new)
            gamma, beta = torch.chunk(modulation_params, 2, dim=1)
            F_LL_new = F_LL * (1 + gamma) + beta
            
            F_LH_new, F_HL_new, F_HH_new = torch.chunk(F_High_new, 3, dim=1)
        
        # F_High_concat = torch.cat([F_LH, F_HL, F_HH], dim=1)

        # mask_semantic = self.lf_context_conv(F_LL)
        # F_High_new = F_High_concat * mask_semantic
        
        # modulation_params = self.hf_modulation_conv(F_High_new)
        # gamma, beta = torch.chunk(modulation_params, 2, dim=1)
        # F_LL_new = F_LL * (1 + gamma) + beta
        
        # F_LH_new, F_HL_new, F_HH_new = torch.chunk(F_High_new, 3, dim=1)
        
        x_reconstructed = self.idwt(F_LL_new, (F_LH_new, F_HL_new, F_HH_new))
        
        if pad_h > 0 or pad_w > 0:
            x_reconstructed = x_reconstructed[:, :, :H, :W]
            
        out = x_reconstructed.flatten(2).permute(0, 2, 1)
        out = self.output_proj(out)
        
        return torch.cat([cls_token, patch_tokens + out], dim=1)


class DriftAwareTemporalAdapter(nn.Module):
    
    def __init__(self, in_channels, adapter_channels, kernel_size=None, T=8):
        super().__init__()
        self.T = T
        self.scale = adapter_channels ** -0.5
        
        self.fc1 = nn.Linear(in_channels, adapter_channels)
    
        self.to_qkv = nn.Linear(adapter_channels, adapter_channels * 3, bias=False)
        
        self.pos_emb = nn.Parameter(torch.randn(1, T, adapter_channels) * 0.02)
        
        self.fc2 = nn.Linear(adapter_channels, in_channels)
        self.act = nn.GELU()
       
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x):
        BT, C = x.size()
        T = self.T
        B = BT // T
        
        x_res = x 

        x_mid = self.act(self.fc1(x)) 

        x_temporal = x_mid.view(B, T, -1)

        x_temporal = x_temporal + self.pos_emb

        qkv = self.to_qkv(x_temporal).chunk(3, dim=-1)
        q, k, v = map(lambda t: t, qkv)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x_out = attn @ v # [B, T, Ca]

        x_out = x_out.view(BT, -1)
        x_out = self.fc2(x_out)
        
        return x_res + x_out


class UnifiedDriftAwareAdapter(nn.Module):

    def __init__(self, in_channels, adapter_channels, T, kernel_size=3,):
        super().__init__()
        self.T = T
        Ca = adapter_channels

        self.fc1 = nn.Linear(in_channels, Ca)

        self.conv = nn.Conv1d(
            Ca, Ca,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=Ca
        )

        self.scale = Ca ** -0.5
        self.to_qkv = nn.Linear(Ca, Ca * 3, bias=False)

        self.register_buffer('inv_freq', 1.0 / (10000 ** (torch.arange(0, Ca, 2).float() / Ca)))
        self.learnable_pe = nn.Parameter(torch.randn(1, T, Ca) * 0.02)

        self.fc2 = nn.Linear(Ca, in_channels)
        self.act = nn.GELU()

        nn.init.constant_(self.conv.weight, 0.)
        nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def get_physics_coordinate(self, x, B, T):

        x_detach = x.detach().view(B, T, -1)
        x_norm = F.normalize(x_detach, p=2, dim=-1)
        x_prev = torch.cat([x_norm[:, :1, :], x_norm[:, :-1, :]], dim=1)
        

        ncc = (x_norm * x_prev).sum(dim=-1) 

        motion_cost = 1.0 - ncc 

        physical_coords = torch.cumsum(motion_cost, dim=1)
        
        return physical_coords

    def get_physics_pos_embed(self, coords, Ca):

        B, T = coords.shape
        
        sin_inp = coords.unsqueeze(-1) * self.inv_freq.unsqueeze(0).unsqueeze(0)

        pos_emb = torch.cat([sin_inp.sin(), sin_inp.cos()], dim=-1)
        
        return pos_emb

    def forward(self, x, pe_type='learnable'):

        BT, C = x.size()
        T = self.T
        B = BT // T
        Ca = self.fc1.out_features

        x_res = x 

        x_mid = self.act(self.fc1(x))  # [BT, Ca]

        x_conv = x_mid.view(B, T, Ca).permute(0, 2, 1).contiguous()  # [B, Ca, T]
        x_conv = self.conv(x_conv)  # depthwise conv
        x_conv = x_conv.permute(0, 2, 1).contiguous()  # [B, T, Ca]

        x_att = x_mid.view(B, T, Ca) 


        if pe_type == 'physics':

            phy_coords = self.get_physics_coordinate(x, B, T)
            pos_emb = self.get_physics_pos_embed(phy_coords, Ca)
            x_att = x_att + pos_emb.to(x_att.dtype)
            
        elif pe_type == 'sinusoidal':
            fixed_coords = torch.arange(T, device=x.device, dtype=torch.float32)
            fixed_coords = fixed_coords.unsqueeze(0).expand(B, -1) # [B, T]
            pos_emb = self.get_physics_pos_embed(fixed_coords, Ca)
            x_att = x_att + pos_emb.to(x_att.dtype)
            
        elif pe_type == 'learnable':

            x_att = x_att + self.learnable_pe.to(x_att.dtype)
            
        elif pe_type == 'none':
            pass
        
        qkv = self.to_qkv(x_att).chunk(3, dim=-1)
        q, k, v = qkv  # [B, T, Ca]

        attn = (q @ k.transpose(-2, -1)) * self.scale   # [B, T, T]

        attn = attn.softmax(dim=-1)

        if not self.training:

            self.last_attn_weights = attn.detach().cpu()

        x_global = attn @ v  # [B, T, Ca]
        x_fuse = x_conv + x_global  # [B, T, Ca]
        x_fuse = x_fuse.view(BT, Ca)
        x_out = self.fc2(x_fuse)

        return x_res + x_out


class CrossModalAttentionAdapter(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        self.ln_q = LayerNorm(dim) 
        
        self.ln_k = LayerNorm(512) 
        self.ln_v = LayerNorm(512)

        self.attn.out_proj.weight.data.zero_()
        self.attn.out_proj.bias.data.zero_()
        
    def forward(self, x, text_seq):
        """
        x: [BT, L_vis, 768]
        text_seq: [B, L_txt, 512]
        """
        BT, L_vis, C = x.size()  # C = 768
        B, L_txt, txt_dim = text_seq.size() # txt_dim = 512
        T = BT // B

        text_expanded = text_seq.unsqueeze(1).expand(-1, T, -1, -1).reshape(BT, L_txt, txt_dim)

        k_512 = self.ln_k(text_expanded)
        v_512 = self.ln_v(text_expanded)

        padding_size = C - txt_dim
        k = F.pad(k_512, (0, padding_size), "constant", 0)
        v = F.pad(v_512, (0, padding_size), "constant", 0)

        q = self.ln_q(x)

        out, _ = self.attn(query=q, key=k, value=v, need_weights=False)
        
        return x + out
class Adapter(nn.Module):

    def __init__(self, in_channels, adapter_channels, kernel_size,T):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, adapter_channels)
        self.conv = nn.Conv3d(
            adapter_channels, adapter_channels,
            kernel_size=kernel_size,
            stride=(1, 1, 1),
            padding=tuple(x // 2 for x in kernel_size),
            groups=adapter_channels,
        )
        self.fc2 = nn.Linear(adapter_channels, in_channels)
        self.T = T
        self.offset_adapter = OffsetAdapter(in_channels,adapter_channels,(1,3,3),T)
        nn.init.constant_(self.conv.weight, 0.)
        nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x):
        offset = self.offset_adapter(x)
        T = self.T
        BT, L, C = x.size()
        B = BT // T
        Ca = self.conv.in_channels
        H = W = round(math.sqrt(L - 1))
        
        assert L - 1 == H * W
        x_id = x
        x = x[:, 1:, :]
        x = self.fc1(x)
        x = x.view(B, T, H, W, Ca).permute(0, 4, 1, 2, 3).contiguous()

        cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = cudnn_enabled and DWCONV3D_DISABLE_CUDNN
        x = self.conv(x)
        torch.backends.cudnn.enabled = cudnn_enabled

        x = x.permute(0, 2, 3, 4, 1).contiguous().view(BT, L - 1, Ca)
        x = self.fc2(x) + offset
        x_id[:, 1:, :] += x
        return x_id


        
class Adapter1(nn.Module):

    def __init__(self, in_channels, adapter_channels, kernel_size,T):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, adapter_channels)
        self.conv = nn.Conv1d(
            adapter_channels, adapter_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=adapter_channels,
        )
        self.fc2 = nn.Linear(adapter_channels, in_channels)
        self.T = T
        nn.init.constant_(self.conv.weight, 0.)
        nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x):
        T = self.T
        BT, C = x.size()
        B = BT // T
        Ca = self.conv.in_channels
        x_id = x
        x = self.fc1(x)
        x = x.view(B, T, Ca).permute(0, 2, 1).contiguous().view(B,Ca,T)  #

        cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = cudnn_enabled and DWCONV3D_DISABLE_CUDNN
        x = self.conv(x)
        torch.backends.cudnn.enabled = cudnn_enabled

        x = x.permute(0, 2, 1).contiguous().view(-1,Ca)
        x = self.fc2(x)
        x_id = x + x_id
        # pdb.set_trace()
        return x_id

class Adapter2(nn.Module):

    def __init__(self, in_channels, adapter_channels, kernel_size,T):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, adapter_channels)
        self.conv = nn.Conv1d(
            adapter_channels, adapter_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=adapter_channels,
        )
        self.fc2 = nn.Linear(adapter_channels, in_channels)
        self.T = T
        self.offset_adapter = OffsetAdapter(in_channels,adapter_channels,(1,3,3),T)
        nn.init.constant_(self.conv.weight, 0.)
        nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x):
        offset = self.offset_adapter(x)
        # pdb.set_trace()
        T = self.T
        BT,L, C = x.size()
        B = BT // T
        Ca = self.conv.in_channels
        x_id = x
        x = self.fc1(x)
        x = x.view(B, T,L, Ca).permute(0, 2,3, 1).contiguous().view(B*L,Ca,T)  #

        cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = cudnn_enabled and DWCONV3D_DISABLE_CUDNN
        x = self.conv(x)
        torch.backends.cudnn.enabled = cudnn_enabled

        x = x.permute(0, 2, 1).contiguous().view(B,L,T,Ca)
        x = x.permute(0, 2, 1,3).contiguous().view(BT,L,Ca)
        x = self.fc2(x)
        x_id[:, 1:, :] += offset
        x_id = x + x_id
        
        return x_id

class OffsetAdapter(nn.Module):

    def __init__(self, in_channels, adapter_channels, kernel_size,T):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, adapter_channels)
        self.conv = nn.Conv3d(
            adapter_channels, adapter_channels,
            kernel_size=kernel_size,
            stride=(1, 1, 1),
            padding=tuple(x // 2 for x in kernel_size),
            groups=adapter_channels,
        )
        self.fc2 = nn.Linear(adapter_channels, in_channels)
        self.T = T
        nn.init.constant_(self.conv.weight, 0.)
        nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x):
        T = self.T
        BT, L, C = x.size()
        B = BT // T
        Ca = self.conv.in_channels
        H = W = round(math.sqrt(L - 1))
        assert L - 1 == H * W
        x_id = x
        x = x[:, 1:, :].view(B,T,-1,C)
        former_id = [0] + [i for i in range(T)][:-1]
        x_former = x[:,former_id]
        # pdb.set_trace()
        offset = x - x_former
        offset = self.fc1(offset)
        offset = offset.view(B, T, H, W, Ca).permute(0, 4, 1, 2, 3).contiguous()
        # 
        cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = cudnn_enabled and DWCONV3D_DISABLE_CUDNN
        offset = self.conv(offset)
        torch.backends.cudnn.enabled = cudnn_enabled

        offset = offset.permute(0, 2, 3, 4, 1).contiguous().view(BT, L - 1, Ca)
        offset = self.fc2(offset)
        # x_id[:, 1:, :] += offset
        return offset

class TextAdapter(nn.Module):

    def __init__(self, in_channels, adapter_channels):
        super().__init__()
        self.textad_fc1 = nn.Linear(in_channels, adapter_channels)
        self.textad_gelu = nn.GELU()
        self.textad_fc2 = nn.Linear(adapter_channels, in_channels)
        nn.init.constant_(self.textad_fc1.bias, 0.)
        nn.init.constant_(self.textad_fc2.bias, 0.)

    def forward(self, x):
        # pdb.set_trace()
        x1 = self.textad_fc1(x)
        x1 = self.textad_gelu(x1)
        x1 = self.textad_fc2(x1)
        x = x + x1
        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        # pdb.set_trace()
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_head: int,
                 adapter_width: int,
                 adapter_kernel_size: Tuple[int, int, int],
                 adapter_pre_attn: bool,
                 adapter_pre_mlp: bool,
                 num_frames: int,
                 attn_mask: torch.Tensor = None,
                 text_dim: int = 512,
                 use_dwt_adapter: bool = True,
                 ) -> None:
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.T = num_frames
        self.attn_mask = attn_mask

        # adapter_class = functools.partial(
        #     Adapter2,
        #     in_channels=d_model,
        #     adapter_channels=adapter_width,
        #     kernel_size=3,
        #     T=self.T
        # )

        text_adapter_class = functools.partial(
            TextAdapter, 
            in_channels=d_model, 
            adapter_channels=adapter_width
        )

        if num_frames > 0:
            # self.adapter_pre_attn = adapter_class() if adapter_pre_attn else None
            # self.adapter_pre_mlp = adapter_class() if adapter_pre_mlp else None

            self.adapter_pre_attn = None
            self.adapter_pre_mlp = None

            self.spatial_adapter = DWTInteractiveAdapter(
                in_channels=d_model,
                adapter_channels=adapter_width
            ) if (adapter_pre_attn and use_dwt_adapter) else None 
            
            # self.cross_adapter = CrossModalAttentionAdapter(
            #     dim=d_model, 
            #     num_heads=n_head
            # ) if adapter_pre_mlp else None

            
        else:
            self.adapter_pre_attn = text_adapter_class() if adapter_pre_attn else None
            self.adapter_pre_mlp = text_adapter_class() if adapter_pre_mlp else None

            self.spatial_adapter = None

        self.adapter_pre_attn_off = None
        self.adapter_pre_mlp_off = None
        

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        B, L, C = x.size()
        # pdb.set_trace()
        # H = self.attn.num_heads

        # qkv = F.linear(x, weight=self.attn.in_proj_weight, bias=self.attn.in_proj_bias)
        # qkv = qkv.view(B, L, H * 3, -1).permute(0, 2, 1, 3)
        # q, k, v = qkv.split([H, H, H], dim=1)
        # out = F.scaled_dot_product_attention(q, k, v)
        # out = out.permute(0, 2, 1, 3).flatten(-2)
        # out = self.attn.out_proj(out)

        # return out
        x = x.permute(1,0,2)

        # x = self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        try:
            x, weights = self.attn(x, x, x, need_weights=True, average_attn_weights=False, attn_mask=self.attn_mask)
        except TypeError:
            x, weights = self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)

        self.last_attn_weights = weights.detach().cpu()

        return x.permute(1,0,2)

    def cross_attention(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        
        B, L, C = x.size()
        # pdb.set_trace()
        # H = self.attn.num_heads

        # qkv = F.linear(x, weight=self.attn.in_proj_weight, bias=self.attn.in_proj_bias)
        # qkv = qkv.view(B, L, H * 3, -1).permute(0, 2, 1, 3)
        # q, k, v = qkv.split([H, H, H], dim=1)
        # out = F.scaled_dot_product_attention(q, k, v)
        # out = out.permute(0, 2, 1, 3).flatten(-2)
        # out = self.attn.out_proj(out)

        # return out
        x = x.permute(1,0,2)
        k = k.permute(1,0,2)
        # print(x.shape)
        # pdb.set_trace()
        x = self.attn(x, k, k, need_weights=False, attn_mask=None)[0]

        return x.permute(1,0,2)
        

    def forward(self, x: torch.Tensor, text_features: torch.Tensor = None) -> torch.Tensor:


        if self.adapter_pre_attn is not None:
            x = self.adapter_pre_attn(x)

        x = x + self.attention(self.ln_1(x))

        if self.spatial_adapter is not None:
            x = self.spatial_adapter(x)

        if self.adapter_pre_mlp is not None: # Text Branch
            x = self.adapter_pre_mlp(x)


        x = x + self.mlp(self.ln_2(x))
        return x

    def forward_cross(self,
                x: torch.Tensor,
                k: torch.Tensor
                ) -> torch.Tensor:
        
        # pdb.set_trace()
        if self.adapter_pre_attn is not None:
            x = self.adapter_pre_attn(x)
        x = x + self.cross_attention(self.ln_1(x),self.ln_1(k))
        if self.adapter_pre_mlp is not None:
            x = self.adapter_pre_mlp(x)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 adapter_width: int,
                 adapter_layers: int,
                 adapter_kernel_size: Tuple[int, int, int],
                 adapter_pre_attn: bool,
                 adapter_pre_mlp: bool,
                 num_frames: int,
                 attn_mask: torch.Tensor = None,
                 text_dim: int=512,
                 use_dwt_adapter: bool = True,
                 ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                d_model=width,
                n_head=heads,
                adapter_width=adapter_width,
                adapter_kernel_size=adapter_kernel_size,
                adapter_pre_attn=adapter_pre_attn and i >= layers - adapter_layers,
                adapter_pre_mlp=adapter_pre_mlp and i >= layers - adapter_layers,
                num_frames=num_frames,
                attn_mask=attn_mask,
                use_dwt_adapter=use_dwt_adapter,
            )
            for i in range(layers)
        ])        
    
    def forward(self, x: torch.Tensor, text_features: Union[torch.Tensor, list] = None, return_intermediate: bool = False) -> Union[torch.Tensor, list]:
        intermediate_outputs = []
        for i, block in enumerate(self.resblocks):
            x = block(x, text_features=text_features)
            
            if return_intermediate:
                intermediate_outputs.append(x)
                
        if return_intermediate:
            return x, intermediate_outputs
        return x


class VisionTransformer(nn.Module):
    def __init__(self,
                 input_resolution: int,
                 patch_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 output_dim: int,
                 adapter_width: int,
                 adapter_layers: int,
                 adapter_kernel_size: Tuple[int, int, int],
                 adapter_pre_attn: bool,
                 adapter_pre_mlp: bool,
                 num_classes: int,
                 num_frames:int,
                 class_fc: bool = True,
                 use_dwt_adapter: bool = True,
                 use_drift_adapter: bool = True,
                 ):
        super().__init__()
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width,
            kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(
                (input_resolution // patch_size) ** 2 + 1, width
            )
        )
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads,
            adapter_width, adapter_layers, adapter_kernel_size,
            adapter_pre_attn, adapter_pre_mlp,num_frames, text_dim=output_dim, use_dwt_adapter=use_dwt_adapter)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        if use_drift_adapter:
            self.t_adapter = UnifiedDriftAwareAdapter(
                in_channels=output_dim,
                adapter_channels=output_dim // 2,
                T=num_frames
            )
        else:
            self.t_adapter = nn.Identity()
        # self.t_adapter = Adapter1(
        #     in_channels=output_dim,
        #     adapter_channels=output_dim//2,
        #     kernel_size=3,
        #     T=num_frames
        # )
        # for n, p in self.named_parameters():
        #   if 'adapter' not in n:
        #     p.requires_grad_(False)
        #     p.data = p.data.half()
        
        self.adapter_dropout = nn.Dropout(0.5)
        
        self.adapter_fc = \
            nn.Linear(width, num_classes) if class_fc else None 
        if self.adapter_fc is not None:
            nn.init.normal_(self.adapter_fc.weight, std=0.02)
            nn.init.constant_(self.adapter_fc.bias, 0.)


    def forward(self, x: torch.Tensor, text_features: torch.Tensor = None):
        B, T = x.size(0), x.size(2)
        # pdb.set_trace()
        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        spatial_size = tuple(x.size()[2:])
        x = x.flatten(-2).permute(0, 2, 1)
        x = torch.cat([
            self.class_embedding.view(1, 1, -1).expand(x.shape[0], -1, -1), x
            ], dim=1)  # [*, grid ** 2 + 1, width]
        # pdb.set_trace()
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = self.transformer(x, text_features=text_features)

        # x = x.view(B, T, x.size(1), x.size(2)).flatten(0, 1) # BT, L, D
        
        # pdb.set_trace()
             

        if self.adapter_fc is not None:
            x2 = x.contiguous().view(B, T, spatial_size[0] * spatial_size[1] + 1, x.size(-1))
            x2 = x2[:, :, 0, :].mean(dim=1)

            x2 = self.ln_post(x2)
            x2 = self.adapter_dropout(x2)
            x2 = self.adapter_fc(x2)
        else:
            x2= None
        
        x1 = self.ln_post(x)

        if not isinstance(self.t_adapter, nn.Identity):
            x1 = x1[:,0,:] @ self.proj
            x1 = self.t_adapter(x1) 
            x = x1.view(B,T,-1).mean(dim=1, keepdim=False)
        else:
            x1 = x1[:,0,:] @ self.proj
            x = x1.view(B,T,-1).mean(dim=1, keepdim=False)
            

        return x,x1.view(B,T,-1),x2, x1

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


# ============================================
# Hard-coded class text descriptors (edit here)
# ============================================


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 vision_adapter_width: int,
                 vision_adapter_layers: int,
                 vision_adapter_kernel_size: Tuple[int, int, int],
                 vision_adapter_pre_attn: bool,
                 vision_adapter_pre_mlp: bool,           
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 text_adapter_width: int,
                 text_adapter_layers: int,
                 text_adapter_kernel_size: Tuple[int, int, int],
                 text_adapter_pre_attn: bool,
                 text_adapter_pre_mlp: bool,  
                 num_classes: int,
                 num_frames: int,
                 use_vision_gated: bool = True,
                 ):
        super().__init__()

        self.context_length = context_length

        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    adapter_width=vision_adapter_width,
                    adapter_layers=vision_adapter_layers,
                    adapter_kernel_size=vision_adapter_kernel_size,
                    adapter_pre_attn=vision_adapter_pre_attn,
                    adapter_pre_mlp=vision_adapter_pre_mlp,
                    output_dim=embed_dim,
                    num_classes=num_classes,
                    class_fc=True,
                    num_frames=num_frames
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            adapter_width=text_adapter_width, 
            adapter_layers=text_adapter_layers, 
            adapter_kernel_size=text_adapter_kernel_size,
            adapter_pre_attn=text_adapter_pre_attn,
            adapter_pre_mlp=text_adapter_pre_mlp,
            num_frames=0,

            )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.initialize_parameters()

        self.vg_agg = None

        self.tokenizer = SimpleTokenizer()

        self.class_prompts = {
            0: [
                "abdominal organs and potential spaces appear normal",
                "no abnormal fluid collection observed",
                "abdominal cavity appears clear without anechoic areas",
                "normal abdominal sonographic appearance",
                "The liver and right kidney surfaces are in tight coaptation, showing no fluid interface",
                "Diaphragm appears intact and no fluid is noted in the pleural space",
                "Probe movement shows normal sliding of the peritoneal surfaces without abnormal separation",
                "Fascial planes and peritoneal layers demonstrate normal echo structure and adherence",
                "Morrison's pouch demonstrates a clean, hyperechoic interface between the liver and right kidney",
                "The splenorenal recess is visualized with no evidence of hypoechoic separation",
            ],
            1: [
                "anechoic free fluid visible in dependent abdominal spaces",
                "free fluid accumulation detected in abdominal cavity",
                "presence of anechoic abnormal liquid collection",
                "fluid-filled region observed in abdominal scan",
                "The organ margins are pulled apart by a hypoechoic, triangular fluid area",
                "Fluid is present extending from the superior pole of the spleen to the diaphrag",
                "During the scan a fluid-filled area was detected, indicating a possible accumulation of fluid",
                "An anechoic region suggestive of free fluid is seen in the abdominal cavity",
                "A distinct anechoic strip is visualized separating the liver capsule from the right kidney",
                "Free fluid is noted accumulating in the pelvis posterior to the bladder",
   

            ]
        }


        self.class_text_descriptors = self.build_class_text_descriptors()
        if use_vision_gated:
            self.vg_agg = VisionGatedTextAggregator(
                vision_dim=512,  
                text_dim=embed_dim,      
            )
        else:
            self.vg_agg = None
        # for n, p in self.named_parameters():
        #   if 'adapter' not in n:
        #     p.requires_grad_(False)
        #     p.data = p.data.half()
    
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

        

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, text_features=None):
        return self.visual(image.type(self.dtype), text_features=text_features)

    def encode_text(self, text, return_all_layers=False):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        
        x_final, intermediate_outputs = self.transformer(x, return_intermediate=True)
        
        final_emb = self.ln_final(x_final).type(self.dtype)
        final_emb = final_emb[torch.arange(final_emb.shape[0]), text.argmax(dim=-1)]
        final_emb = final_emb @ self.text_projection

        if return_all_layers:
            return final_emb, x_final, intermediate_outputs
        return final_emb, x_final
    
    def build_class_text_descriptors(self):
        """
        Convert hard-coded class prompts into text embeddings using CLIP encoder.
        Returns Tensor of shape [C, N, Dt].
        """
        all_class_embs = []
        device = next(self.parameters()).device

        for class_id in sorted(self.class_prompts.keys()):
            prompts = self.class_prompts[class_id]  # a list of sentences

            tokenized = []
            for s in prompts:
                ids = [self.tokenizer.encoder["<|startoftext|>"]] + \
                    self.tokenizer.encode(s)[:self.context_length - 2] + \
                    [self.tokenizer.encoder["<|endoftext|>"]]
                # padding
                pad_len = self.context_length - len(ids)
                ids = ids + [0] * pad_len
                tokenized.append(ids)

            tok = torch.tensor(tokenized, dtype=torch.long).to(device)   # [N, 77]

            with torch.no_grad():
                # encode_text should take token ids
                emb, _ = self.encode_text(tok, return_all_layers=False)  # [N, Dt]

            all_class_embs.append(emb)

        # → [C, N, Dt]
        return torch.stack(all_class_embs, dim=0)


    
    def forward(self, image, text):

        final_emb, x_final_seq = self.encode_text(text, return_all_layers=False)
        
        vision_text_inputs = x_final_seq 

        image_features, _, image_fc, t_visual = self.encode_image(image, text_features=vision_text_inputs)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        if t_visual is not None and self.vg_agg is not None:
            text_features = self.vg_agg(t_visual, self.class_text_descriptors)
        else:
            text_features = final_emb  

        text_features_norm = final_emb / final_emb.norm(dim=1, keepdim=True) 

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features_norm.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text, image_features, image_fc

def copy_weights(source_module, target_module):
    source_params = dict(source_module.named_parameters())
    target_params = dict(target_module.named_parameters())
    
    for target_name, target_param in target_params.items():
        # pdb.set_trace()
        if target_name in source_params:
            source_param = source_params[target_name]
            if source_param.data.shape == target_param.data.shape:
                target_param.data.copy_(source_param.data)
            else:
                print(f"Warning: Shapes mismatch for parameters {target_name}. Skipping copying.")
        else:
            print(f"Warning: Parameter {target_name} not found in source module. Skipping copying.")


def clip_vit_base_patch16_adapter24x384(**kwargs):
    model = VisionTransformer(
        input_resolution=224,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        adapter_width=384,
        adapter_layers=12,
        adapter_kernel_size=(3, 1, 1),
        adapter_pre_attn=True,
        adapter_pre_mlp=True,
        **kwargs,
    )
    assert CLIP_VIT_B16_PATH is not None, \
        'Please set CLIP_VIT_B16_PATH in configs.py.'
    checkpoint = torch.jit.load(CLIP_VIT_B16_PATH, map_location='cpu')
    print(model.load_state_dict(checkpoint.visual.state_dict(), strict=False))
    return model

def clip_vit_base_patch16_adapter12x384(**kwargs):
    model = VisionTransformer(
        input_resolution=224,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        adapter_width=384,
        adapter_layers=12,
        adapter_kernel_size=(3, 1, 1),
        adapter_pre_attn=True,
        adapter_pre_mlp=True,
        **kwargs,
    )
    assert CLIP_VIT_B16_PATH is not None, \
        'Please set CLIP_VIT_B16_PATH in configs.py'
    checkpoint = torch.jit.load(CLIP_VIT_B16_PATH, map_location='cpu')
    print(model.load_state_dict(checkpoint.visual.state_dict(), strict=False))
    return model
    

def clip_vit_base_patch32_adapter12x384(**kwargs):
    model = VisionTransformer(
        input_resolution=224,
        patch_size=32,
        width=768,
        layers=12,
        heads=12,
        adapter_width=384,
        adapter_layers=12,
        adapter_kernel_size=(3, 1, 1),
        adapter_pre_attn=False,
        adapter_pre_mlp=True,
        **kwargs,
    )
    assert CLIP_VIT_B32_PATH is not None, \
        'Please set CLIP_VIT_B32_PATH in configs.py'
    # pdb.set_trace()
    checkpoint = torch.jit.load(CLIP_VIT_B32_PATH, map_location='cpu')
    # print(model.load_state_dict(checkpoint.visual.state_dict(), strict=False))
    model.load_state_dict(checkpoint.visual.state_dict(), strict=False)
    return model

def clip_vit_base_patch32_adapter24x384(**kwargs):
    model = VisionTransformer(
        input_resolution=224,
        patch_size=32,
        width=768,
        layers=12,
        heads=12,
        adapter_width=384,
        adapter_layers=12,
        adapter_kernel_size=(3, 1, 1),
        adapter_pre_attn=True,
        adapter_pre_mlp=True,
        **kwargs,
    )
    assert CLIP_VIT_B32_PATH is not None, \
        'Please set CLIP_VIT_B32_PATH in configs.py'
    # pdb.set_trace()
    checkpoint = torch.jit.load(CLIP_VIT_B32_PATH, map_location='cpu')
    # print(model.load_state_dict(checkpoint.visual.state_dict(), strict=False))
    model.load_state_dict(checkpoint.visual.state_dict(), strict=False)
    return model

def clip_vit_base_patch16_multimodal_adapter24x384(**kwargs):
    checkpoint = torch.jit.load(CLIP_VIT_B16_PATH, map_location='cpu')
    model = CLIP(
        embed_dim=512,
        image_resolution=224,
        vision_patch_size=16,
        vision_width=768,
        vision_layers=12,
        vision_adapter_width=384,
        vision_adapter_layers=12,
        vision_adapter_kernel_size=(3, 1, 1),
        vision_adapter_pre_attn=True,
        vision_adapter_pre_mlp=True,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=13,
        text_adapter_width=384,
        text_adapter_layers=13,
        text_adapter_kernel_size=(3, 1, 1),
        text_adapter_pre_attn=True,
        text_adapter_pre_mlp=True,  
        **kwargs,
    )
    assert CLIP_VIT_B16_PATH is not None, \
        'Please set CLIP_VIT_B16_PATH in configs.py'
    # pdb.set_trace()
    
    
    # print(model.load_state_dict(checkpoint.visual.state_dict(), strict=False))
    model.load_state_dict(checkpoint.state_dict(), strict=False)
    
    copy_weights(model.transformer.resblocks[11], model.transformer.resblocks[12])
    convert_weights(model)
    return model

def clip_vit_base_patch32_multimodal_adapter12x384(**kwargs):
    checkpoint = torch.jit.load(CLIP_VIT_B32_PATH, map_location='cpu')
    model = CLIP(
        embed_dim=512,
        image_resolution=224,
        vision_patch_size=32,
        vision_width=768,
        vision_layers=12,
        vision_adapter_width=384,
        vision_adapter_layers=12,
        vision_adapter_kernel_size=(3, 1, 1),
        vision_adapter_pre_attn=False,
        vision_adapter_pre_mlp=True,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=13,
        text_adapter_width=384,
        text_adapter_layers=1,
        text_adapter_kernel_size=(3, 1, 1),
        text_adapter_pre_attn=False,
        text_adapter_pre_mlp=True,  
        **kwargs,
    )
    # pdb.set_trace()
    assert CLIP_VIT_B32_PATH is not None, \
        'Please set CLIP_VIT_B32_PATH in configs.py'
    # pdb.set_trace()
    # convert_weights(model)
    
    # print(model.load_state_dict(checkpoint.visual.state_dict(), strict=False))
    model.load_state_dict(checkpoint.state_dict(), strict=False)
    
    copy_weights(model.transformer.resblocks[11], model.transformer.resblocks[12])
    convert_weights(model)
    return model

def clip_vit_base_patch16_multimodal_adapter12x384(**kwargs):
    checkpoint = torch.jit.load(CLIP_VIT_B16_PATH, map_location='cpu')
    model = CLIP(
        embed_dim=512,
        image_resolution=224,
        vision_patch_size=16,
        vision_width=768,
        vision_layers=12,
        vision_adapter_width=384,
        vision_adapter_layers=12,
        vision_adapter_kernel_size=(3, 1, 1),
        vision_adapter_pre_attn=True,
        vision_adapter_pre_mlp=True,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=13,
        text_adapter_width=384,
        text_adapter_layers=3,
        text_adapter_kernel_size=(3, 1, 1),
        text_adapter_pre_attn=True,
        text_adapter_pre_mlp=True,  
        **kwargs,
    )
    assert CLIP_VIT_B16_PATH is not None, \
        'Please set CLIP_VIT_B16_PATH in configs.py'
    # pdb.set_trace()
    # convert_weights(model)
    
    # print(model.load_state_dict(checkpoint.visual.state_dict(), strict=False))
    model.load_state_dict(checkpoint.state_dict(), strict=False)
    
    copy_weights(model.transformer.resblocks[11], model.transformer.resblocks[12])
    convert_weights(model)
    return model
    
def clip_vit_base_patch32_multimodal_adapter24x384(**kwargs):
    checkpoint = torch.jit.load(CLIP_VIT_B32_PATH, map_location='cpu')
    model = CLIP(
        embed_dim=512,
        image_resolution=224,
        vision_patch_size=32,
        vision_width=768,
        vision_layers=12,
        vision_adapter_width=384,
        vision_adapter_layers=12,
        vision_adapter_kernel_size=(3, 1, 1),
        vision_adapter_pre_attn=True,
        vision_adapter_pre_mlp=True,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=13,
        text_adapter_width=384,
        text_adapter_layers=13,
        text_adapter_kernel_size=(3, 1, 1),
        text_adapter_pre_attn=False,
        text_adapter_pre_mlp=True,  
        **kwargs,
    )
    assert CLIP_VIT_B32_PATH is not None, \
        'Please set CLIP_VIT_B32_PATH in configs.py'
    # pdb.set_trace()
    # convert_weights(model)
    
    # print(model.load_state_dict(checkpoint.visual.state_dict(), strict=False))
    model.load_state_dict(checkpoint.state_dict(), strict=False)
    
    copy_weights(model.transformer.resblocks[11], model.transformer.resblocks[12])
    return model

