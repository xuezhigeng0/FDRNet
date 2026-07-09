import math
import torch
import torch.nn as nn
import torch.nn.functional as F 
from timm.models.layers import DropPath, trunc_normal_
from timm.models.convnext import ConvNeXtBlock
from timm.models.mlp_mixer import MixerBlock
from timm.models.swin_transformer import SwinTransformerBlock, window_partition, window_reverse
from timm.models.vision_transformer import Block as ViTBlock

from .layers import (HorBlock, ChannelAggregationFFN, MultiOrderGatedAggregation,
                     PoolFormerBlock, CBlock, SABlock, MixMlp, VANBlock)


class BasicConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 upsampling=False,
                 act_norm=False,
                 act_inplace=True):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if upsampling is True:
            self.conv = nn.Sequential(*[
                nn.Conv2d(in_channels, out_channels*4, kernel_size=kernel_size,
                          stride=1, padding=padding, dilation=dilation),
                nn.PixelShuffle(2)
            ])
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation)

        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.SiLU(inplace=act_inplace)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
       # x=x.view(x.size(0)*x.size(1),x.size(2),x.size(3),x.size(4))
       # print(x.shape)
        #B, T, C, H, W = x.shape
       # x = x.view(B * T, C, H, W)
        if len(x.shape) == 5:
            B, T, C, H, W = x.shape
        # 将时间维度 T 和 batch 大小 B 合并到一起，形成 4D 张量
            x = x.view(B * T, C, H, W)
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y

class ConvSC(nn.Module):

    def __init__(self,
                 C_in,
                 C_out,
                 kernel_size=3,
                 downsampling=False,
                 upsampling=False,
                 act_norm=True,
                 act_inplace=True):
        super(ConvSC, self).__init__()

        stride = 2 if downsampling is True else 1
        padding = (kernel_size - stride + 1) // 2

        self.conv = BasicConv2d(C_in, C_out, kernel_size=kernel_size, stride=stride,
                                upsampling=upsampling, padding=padding,
                                act_norm=act_norm, act_inplace=act_inplace)

    def forward(self, x):
        y = self.conv(x)
        return y


class GroupConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 groups=1,
                 act_norm=False,
                 act_inplace=True):
        super(GroupConv2d, self).__init__()
        self.act_norm=act_norm
        if in_channels % groups != 0:
            groups=1
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=groups)
        self.norm = nn.GroupNorm(groups,out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=act_inplace)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y


class gInception_ST(nn.Module):
    """A IncepU block for SimVP"""

    def __init__(self, C_in, C_hid, C_out, incep_ker = [3,5,7,11], groups = 8):        
        super(gInception_ST, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)

        layers = []
        for ker in incep_ker:
            layers.append(GroupConv2d(
                C_hid, C_out, kernel_size=ker, stride=1,
                padding=ker//2, groups=groups, act_norm=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y





class ConvMixerSubBlock(nn.Module):
    """具有多尺度卷积、1D卷积和加权融合的ConvMixer块。"""
    
    def __init__(self, dim, kernel_sizes=[7, 9, 11], activation=nn.GELU):
        super().__init__()

        # 确保 kernel_sizes 至少包含一个值
        self.kernel_sizes = kernel_sizes
        
        # 为不同的卷积核尺寸创建卷积层列表
        self.conv_dw = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=k, groups=dim, padding=k//2) for k in kernel_sizes
        ])
        
        # 为每个尺度（卷积核尺寸）创建可学习的权重
        self.weights = nn.Parameter(torch.ones(len(kernel_sizes)))  # 形状: (len(kernel_sizes),)

        # 用于时间感知的1D卷积层
        self.conv_1d = nn.Conv1d(dim, dim, kernel_size=3, padding=1)  # 示例1D卷积
        
        # 激活函数和归一化
        self.act_1 = activation()
        self.norm_1 = nn.BatchNorm2d(dim)
        
        # 通道混合
        self.conv_pw = nn.Conv2d(dim, dim, kernel_size=1)
        self.act_2 = activation()
        self.norm_2 = nn.BatchNorm2d(dim)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return dict()

    def forward(self, x):
        # 对不同尺度应用多个卷积，并加权求和
        conv_outs = [self.conv_dw[i](x) for i in range(len(self.kernel_sizes))]
        
        # 应用学习的权重并求和结果
        weighted_sum = sum(w * conv_outs[i] for i, w in enumerate(self.weights))
        
        # 跳跃连接：将原始输入 'x' 加到加权和上
        x = x + self.norm_1(self.act_1(weighted_sum))
        #print(x.shape)
        # 应用1D卷积以增强时间感知能力（假设时间是第二个维度）
        x = x.permute(0, 2, 3, 1)  # 调整为 (batch, height, width, channels)
        #print(x.shape)
        x= x.view(x.size(0), -1, x.size(3))  # 压平 height 和 width 维度
       # print(x.shape)
        x=x.reshape(x.size(0), x.size(2), x.size(1))  # 压平 height 和 width 维度
        x = self.conv_1d(x)  # 在最后一维（时间维度）上应用1D卷积
        #print(x.shape)
        x = x.view(x.size(0), -1, x.size(2), x.size(1))  # 重新调整形状
       # print(x.shape)
        # 假设x的初始形状为 [1, 1, 512, 384]
# 我们将通道数从 1 调整为 384，确保符合卷积层的输入要求
        x = x.reshape(x.size(0), x.size(3), x.size(2), x.size(1))  # 将通道数调整为 384
       # print(x.shape)
        # 使用1x1卷积进行通道混合
        x = self.norm_2(self.act_2(self.conv_pw(x)))
        return x


class ConvNeXtSubBlock(ConvNeXtBlock):
    """A block of ConvNeXt."""

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.1):
        super().__init__(dim, mlp_ratio=mlp_ratio,
                         drop_path=drop_path, ls_init_value=1e-6, conv_mlp=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'gamma'}

    def forward(self, x):
        x = x + self.drop_path(
            self.gamma.reshape(1, -1, 1, 1) * self.mlp(self.norm(self.conv_dw(x))))
        return x


class HorNetSubBlock(HorBlock):
    """A block of HorNet."""

    def __init__(self, dim, mlp_ratio=4., drop_path=0.1, init_value=1e-6):
        super().__init__(dim, mlp_ratio=mlp_ratio, drop_path=drop_path, init_value=init_value)
        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'gamma1', 'gamma2'}

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


class MLPMixerSubBlock(MixerBlock):
    """A block of MLP-Mixer."""

    def __init__(self, dim, input_resolution=None, mlp_ratio=4., drop=0., drop_path=0.1):
        seq_len = input_resolution[0] * input_resolution[1]
        super().__init__(dim, seq_len=seq_len,
                         mlp_ratio=(0.5, mlp_ratio), drop_path=drop_path, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return dict()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x.reshape(B, H, W, C).permute(0, 3, 1, 2)


class MogaSubBlock(nn.Module):
    """A block of MogaNet."""

    def __init__(self, embed_dims, hidden_size,mlp_ratio=4., drop_rate=0., drop_path_rate=0., init_value=1e-5,
                 ):
        super(MogaSubBlock, self).__init__()
        self.out_channels = embed_dims
        # spatial attention
        self.norm1 = nn.BatchNorm2d(embed_dims)#2D 批量归一化
        self.attn = MultiOrderGatedAggregation(
            embed_dims,hidden_size)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # channel MLP
        self.norm2 = nn.BatchNorm2d(embed_dims)
        mlp_hidden_dims = int(embed_dims * mlp_ratio)
        self.mlp = ChannelAggregationFFN(
            embed_dims=embed_dims, mlp_hidden_dims=mlp_hidden_dims, ffn_drop=drop_rate)
        # init layer scale
        self.layer_scale_1 = nn.Parameter(init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'layer_scale_1', 'layer_scale_2', 'sigma'}

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1 * self.attn(self.norm1(x)))
        x = x.view(x.size(0), x.size(1), x.size(2), x.size(3))  
        x= self.norm2(x)  
        x = x + self.drop_path(self.layer_scale_2 * self.mlp(x))
        return x


class PoolFormerSubBlock(PoolFormerBlock):
    """A block of PoolFormer."""

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.1):
        super().__init__(dim, pool_size=3, mlp_ratio=mlp_ratio, drop_path=drop_path,
                         drop=drop, init_value=1e-5)
        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'layer_scale_1', 'layer_scale_2'}

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class SwinSubBlock(SwinTransformerBlock):
    """A block of Swin Transformer."""

    def __init__(self, dim, input_resolution=None, layer_i=0, mlp_ratio=4., drop=0., drop_path=0.1):
        window_size = 7 if input_resolution[0] % 7 == 0 else max(4, input_resolution[0] // 16)
        window_size = min(8, window_size)
        shift_size = 0 if (layer_i % 2 == 0) else window_size // 2
        super().__init__(dim, input_resolution, num_heads=8, window_size=window_size,
                         shift_size=shift_size, mlp_ratio=mlp_ratio,
                         drop_path=drop_path, attn_drop=drop, proj_drop=drop, qkv_bias=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        x = super().forward(x)

        return x.reshape(B, H, W, C).permute(0, 3, 1, 2)


def UniformerSubBlock(embed_dims, mlp_ratio=4., drop=0., drop_path=0.,
                      init_value=1e-6, block_type='Conv'):
    """Build a block of Uniformer."""

    assert block_type in ['Conv', 'MHSA']
    if block_type == 'Conv':
        return CBlock(dim=embed_dims, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
    else:
        return SABlock(dim=embed_dims, num_heads=8, mlp_ratio=mlp_ratio, qkv_bias=True,
                       drop=drop, drop_path=drop_path, init_value=init_value)


class VANSubBlock(VANBlock):
    """A block of VAN."""

    def __init__(self, dim, mlp_ratio=4., drop=0.,drop_path=0., init_value=1e-2, act_layer=nn.GELU):
        super().__init__(dim=dim, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path,
                         init_value=init_value, act_layer=act_layer)
        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'layer_scale_1', 'layer_scale_2'}

    def _init_weights(self, m):
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_  # 保留其他timm导入

class ActivationLinear(nn.Module):
    def __init__(self, in_dim, out_dim, use_gelu=False, bias=True):
        super().__init__()
        layers = [nn.Linear(in_dim, out_dim, bias=bias)]
        if use_gelu:
            layers.append(nn.GELU())

            print(f"[ActivationLinear] Using GELU on layer with in_dim={in_dim}, out_dim={out_dim}")
        else:
            print(f"[ActivationLinear] Not using GELU on layer with in_dim={in_dim}, out_dim={out_dim}")
        self.proj = nn.Sequential(*layers)

    def forward(self, x):
            
        return self.proj(x)


class TokenStatisticsSelfAttention(nn.Module):
    """
    Token Statistics Self-Attention (TSSA) adapted from Token Statistics Transformer (TOST).

    Input:  (B, N, C)
    Output: (B, N, C), plus membership Pi of shape (B, heads, N).

    Complexity: O(B * heads * N * (C/heads)) time and memory (linear in N).
    """

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.0, max_len: int = 4096):
        super().__init__()
        assert dim % heads == 0, f'dim ({dim}) must be divisible by heads ({heads})'
        self.heads = heads
        self.dim = dim
        self.head_dim = dim // heads

        self.temp = nn.Parameter(torch.ones(heads, 1))
        # Bias term for numerical stability / flexibility (sliced by N at runtime)
        self.denom_bias = nn.Parameter(torch.zeros(heads, max_len, 1))

        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, w):
        """
        Args:
            w: (B, N, C)
        Returns:
            y:  (B, N, C)
            Pi: (B, heads, N) membership over heads for each token
        """
        B, N, C = w.shape
        w = w.view(B, N, self.heads, self.head_dim).transpose(1, 2)  # (B, heads, N, head_dim)

        w_sq = w ** 2
        denom = torch.cumsum(w_sq, dim=-2).clamp_min(1e-12)
        w_normed = (w_sq / denom) + self.denom_bias[:, :N, :].unsqueeze(0)
        tmp = torch.sum(w_normed, dim=-1) * self.temp.unsqueeze(0)  # (B, heads, N)

        Pi = F.softmax(tmp, dim=1)
        dots = torch.cumsum(w_sq * Pi.unsqueeze(-1), dim=-2) / (Pi.cumsum(dim=-1) + 1e-8).unsqueeze(-1)
        attn = 1.0 / (1.0 + dots)
        attn = self.attn_dropout(attn)

        y = -w * Pi.unsqueeze(-1) * attn
        y = y.transpose(1, 2).contiguous().view(B, N, C)
        y = self.proj_dropout(self.proj(y))
        return y, Pi


class RankAugmentedTokenStatisticsSelfAttention(nn.Module):
    """TSSA + RALA-style KV-buffer rank augmentation (context-aware reweighting).

    The RALA paper attributes linear attention performance drop to low-rank KV buffer
    and low-rank output features. It proposes:
      (1) context-aware token weights alpha_j to reweight each token's contribution
          to the KV buffer (Eq.3-4), and
      (2) token-specific modulation phi(X_i) on the output features (Eq.6).

    Here we adapt (1) to Token-Statistics Self-Attention (TSSA) without breaking
    linear-time complexity:
      - compute alpha from a positive kernel kappa(.) = ELU(.) + 1,
      - use alpha to reweight the cumulative statistics that form TSSA's linear buffer.

    We keep Pi (membership) computation *unchanged* from TSSA for stability, and only
    apply alpha to the buffer-like cumulative statistics (num/den) that determine attn.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dropout: float = 0.0,
        max_len: int = 4096,
        use_alpha: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert dim % heads == 0, f'dim ({dim}) must be divisible by heads ({heads})'
        self.heads = heads
        self.dim = dim
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.use_alpha = use_alpha
        self.eps = eps

        # Same parameters as TokenStatisticsSelfAttention
        self.temp = nn.Parameter(torch.ones(heads, 1))
        self.denom_bias = nn.Parameter(torch.zeros(heads, max_len, 1))

        # Positive kernel used in RALA: kappa(.) = ELU(.) + 1
        self.kappa = nn.ELU()

        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(dropout)

    def _compute_alpha(self, w: torch.Tensor) -> torch.Tensor:
        """Compute context-aware token weights alpha.

        Args:
            w: (B, heads, N, head_dim)
        Returns:
            alpha: (B, heads, N) with alpha.sum(-1) == N
        """
        wk = self.kappa(w) + 1.0
        qg = wk.mean(dim=-2, keepdim=True)  # (B, heads, 1, head_dim)
        logits = (qg * wk).sum(dim=-1) * self.scale  # (B, heads, N)
        alpha = torch.softmax(logits, dim=-1) * w.shape[-2]
        return alpha

    def forward(self, w):
        """Forward.

        Args:
            w: (B, N, C)
        Returns:
            y:  (B, N, C)
            Pi: (B, heads, N)
            alpha: (B, heads, N) or None
        """
        B, N, C = w.shape
        w = w.view(B, N, self.heads, self.head_dim).transpose(1, 2)  # (B, heads, N, head_dim)

        w_sq = w ** 2
        denom = torch.cumsum(w_sq, dim=-2).clamp_min(1e-12)
        w_normed = (w_sq / denom) + self.denom_bias[:, :N, :].unsqueeze(0)
        tmp = torch.sum(w_normed, dim=-1) * self.temp.unsqueeze(0)  # (B, heads, N)

        # Membership over heads (unchanged)
        Pi = F.softmax(tmp, dim=1)

        alpha = self._compute_alpha(w) if self.use_alpha else None
        if alpha is None:
            num = torch.cumsum(w_sq * Pi.unsqueeze(-1), dim=-2)
            den = Pi.cumsum(dim=-1)
        else:
            # Reweight the buffer-like cumulative statistics (KV-buffer analogue)
            num = torch.cumsum(w_sq * Pi.unsqueeze(-1) * alpha.unsqueeze(-1), dim=-2)
            den = (Pi * alpha).cumsum(dim=-1)

        dots = num / (den + self.eps).unsqueeze(-1)
        attn = 1.0 / (1.0 + dots)
        attn = self.attn_dropout(attn)

        y = -w * Pi.unsqueeze(-1) * attn
        y = y.transpose(1, 2).contiguous().view(B, N, C)
        y = self.proj_dropout(self.proj(y))
        return y, Pi, alpha


# 多阶Krylov空间注意力头
class MPABlock(nn.Module):
    """
    Multi-order Projection Attention head.

    - Default: standard QKV attention (O(N^2)).
    - Optionally: Token Statistics Self-Attention (TSSA) (O(N)) for each order.
    """
    def __init__(self, dim, krylov_dim, heads=4, order=3, use_bias=True,
                 use_tssa: bool = False, tssa_max_len: int = 4096, attn_drop: float = 0.0,
                 # --- RALA-style rank augmentation (only used when use_tssa=True) ---
                 use_rala: bool = False, rala_use_alpha: bool = True, rala_use_phi: bool = True,
                 rala_phi_act: str = 'none', rala_eps: float = 1e-6):
        super().__init__()
        self.heads = heads
        self.head_dim = krylov_dim // heads
        self.scale = self.head_dim ** -0.5
        self.order = order

        self.use_tssa = use_tssa
        self.tssa_max_len = tssa_max_len
        self.attn_drop = attn_drop

        # RALA flags (only meaningful when use_tssa=True)
        self.use_rala = use_rala
        self.rala_use_alpha = rala_use_alpha
        self.rala_use_phi = rala_use_phi
        self.rala_phi_act = rala_phi_act
        self.rala_eps = rala_eps

        # 每阶使用不同激活策略：i=0 不激活，i>0 使用 GELU
        self.q_projs = nn.ModuleList([
            ActivationLinear(dim, krylov_dim, use_gelu=(i > 0), bias=use_bias)
            for i in range(order)
        ])
        self.k_projs = nn.ModuleList([
            ActivationLinear(dim, krylov_dim, use_gelu=(i > 0), bias=use_bias)
            for i in range(order)
        ])
        self.v_projs = nn.ModuleList([
            ActivationLinear(dim, krylov_dim, use_gelu=(i > 0), bias=use_bias)
            for i in range(order)
        ])

        # For TSSA we only need one projection per order (w), not Q/K/V.
        self.w_projs = nn.ModuleList([
            ActivationLinear(dim, krylov_dim, use_gelu=(i > 0), bias=use_bias)
            for i in range(order)
        ])

        self.tssa_layers = nn.ModuleList([
            TokenStatisticsSelfAttention(dim=krylov_dim, heads=heads, dropout=attn_drop, max_len=tssa_max_len)
            for _ in range(order)
        ])

        # RALA: KV-buffer rank augmentation via context-aware alpha weights
        self.rala_tssa_layers = nn.ModuleList([
            RankAugmentedTokenStatisticsSelfAttention(
                dim=krylov_dim, heads=heads, dropout=attn_drop, max_len=tssa_max_len,
                use_alpha=rala_use_alpha, eps=rala_eps
            )
            for _ in range(order)
        ]) if use_rala else None

        # RALA: output-feature augmentation (phi) as a light projection conditioned on original x
        self.phi_projs = nn.ModuleList([
            nn.Linear(dim, krylov_dim, bias=use_bias)
            for _ in range(order)
        ]) if (use_rala and rala_use_phi) else None

        # Residual scaling for TSSA branch (stabilizes training and often improves MAE).
        # Initialized to a very small value, similar to LayerScale used in modern Transformers.
        self.tssa_gamma = nn.Parameter(torch.full((order, 1, 1), 1e-5))

        self.to_out = nn.Sequential(
            nn.Linear(krylov_dim * order, dim, bias=use_bias),
            nn.LayerNorm(dim) if use_bias else nn.Identity()
        )

        self.order_weights = nn.Parameter(torch.ones(order, 1, 1, 1))  # 每阶一个权重参数

    def forward(self, x):
        B, N, C = x.shape
        outputs = []

        for i in range(self.order):
            if self.use_tssa:
                # Linear-time Token Statistics Self-Attention
                w = self.w_projs[i](x)  # (B, N, krylov_dim)
                if self.use_rala:
                    out, _Pi, _alpha = self.rala_tssa_layers[i](w)
                    if self.phi_projs is not None:
                        phi = self.phi_projs[i](x)
                        if self.rala_phi_act == 'sigmoid':
                            phi = torch.sigmoid(phi)
                        out = out * phi
                else:
                    out, _Pi = self.tssa_layers[i](w)
                # LayerScale-style residual scaling (helps stabilize and often improves MAE)
                out = out * self.tssa_gamma[i]
            else:
                # Standard QKV attention
                q = self.q_projs[i](x).view(B, N, self.heads, self.head_dim).transpose(1, 2)
                k = self.k_projs[i](x).view(B, N, self.heads, self.head_dim).transpose(1, 2)
                v = self.v_projs[i](x).view(B, N, self.heads, self.head_dim).transpose(1, 2)

                # memory-efficient attention via PyTorch scaled_dot_product_attention (SDPA)
                drop_p = self.attn_drop if self.training else 0.0
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=drop_p, is_causal=False)
                out = out.transpose(1, 2).reshape(B, N, -1)

            outputs.append(out * self.order_weights[i])

        out = torch.cat(outputs, dim=-1)
        return self.to_out(out)

# 分组前馈网络
class GroupedMLP(nn.Module):
    def __init__(self, dim, mlp_hidden_dim, groups=4, drop=0.):
        super().__init__()
        self.groups = groups
        self.group_dim = dim // groups
        self.group_hidden_dim = mlp_hidden_dim // groups

        self.linear1 = nn.Linear(self.group_dim, self.group_hidden_dim, bias=True)
        self.linear2 = nn.Linear(self.group_hidden_dim, self.group_dim, bias=True)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        B, N, C = x.shape
        x = x.view(B, N, self.groups, self.group_dim)
        x = x.permute(0, 2, 1, 3).contiguous().view(B * self.groups, N, self.group_dim)

        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)

        x = x.view(B, self.groups, N, self.group_dim).permute(0, 2, 1, 3).contiguous()
        return x.view(B, N, C)

# 整体KAN块
class MPA(nn.Module):
    """增强版MultiKAN层：添加门控机制和FFN"""
    def __init__(self, dim, krylov_dim, order=3, rank=8, heads=4,
                 mlp_ratio=2.0, drop=0., drop_path=0.1, use_gate=True,
                 use_tssa: bool = False, tssa_max_len: int = 4096, attn_drop: float = 0.0,
                 # --- RALA-style rank augmentation (only used when use_tssa=True) ---
                 use_rala: bool = False, rala_use_alpha: bool = True, rala_use_phi: bool = True,
                 rala_phi_act: str = 'none', rala_eps: float = 1e-6):
        super().__init__()

        self.multi_head_kan = MPABlock(
            dim=dim,
            krylov_dim=krylov_dim,
            heads=heads,
            order=order,
            use_tssa=use_tssa,
            tssa_max_len=tssa_max_len,
            attn_drop=attn_drop,
            use_rala=use_rala,
            rala_use_alpha=rala_use_alpha,
            rala_use_phi=rala_use_phi,
            rala_phi_act=rala_phi_act,
            rala_eps=rala_eps,
        )

        self.use_gate = use_gate
        if use_gate:
            self.gate = nn.Sequential(
                nn.Linear(dim, 1),
                nn.Sigmoid()
            )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = GroupedMLP(dim, mlp_hidden_dim, groups=4, drop=drop)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        residual = x
        x_kan = self.norm1(x)
        x_kan = self.multi_head_kan(x_kan)

        if self.use_gate:
            gate_value = self.gate(x_kan)
            x_kan = x_kan * gate_value

        x = residual + self.drop_path(x_kan)

        residual = x
        x_mlp = self.norm2(x)
        x_mlp = self.mlp(x_mlp)
        x = residual + self.drop_path(x_mlp)

        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_


class MultiStepODEFusion2D(nn.Module):
    """
    FuseUNet-inspired multi-scale feature fusion via linear multistep ODE integration.

    Expected input: a list of feature maps ordered from coarse -> fine, all with the same (H, W).
    Output: a fused feature map with the same shape.
    """

    def __init__(self, channels: int, use_norm: bool = True):
        super().__init__()
        self.channels = channels

        gn = lambda c: nn.GroupNorm(num_groups=math.gcd(c, min(32, c)), num_channels=c) if use_norm else nn.Identity()

        # x (skip) projection and y (memory) processing
        self.opx = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            gn(channels),
            nn.GELU()
        )
        # separate projection for the "predictor" step (matches FuseUNet's x_upper usage)
        self.opx_upper = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            gn(channels),
            nn.GELU()
        )
        # memory stream operator (depthwise for efficiency)
        self.opy = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            gn(channels),
            nn.GELU()
        )
        # gated fusion for (x, y)
        self.opxy = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.GELU()
        )

        # step-size scale (learnable); initialized to 1.0
        self.step_scale = nn.Parameter(torch.tensor(1.0))

    def _ode_f(self, x: torch.Tensor, y: torch.Tensor, x_proj: nn.Module) -> torch.Tensor:
        y_mem = self.opy(y)
        return -y_mem + self.opxy(torch.cat([x_proj(x), y_mem], dim=1))

    def ode_eq(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._ode_f(x, y, self.opx)

    def ode_eq_pre(self, x: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return self._ode_f(x, y_pred, self.opx_upper)

    def forward(self, xs: list) -> torch.Tensor:
        L = len(xs)
        if L == 0:
            raise ValueError("MultiStepODEFusion2D expects a non-empty list of feature maps.")
        if L == 1:
            return xs[0]

        # FuseUNet uses delta = 1 / L for an L-stage fusion; we keep the same scaling and make it learnable.
        h = (1.0 / float(L)) * self.step_scale

        y = torch.zeros_like(xs[0])
        f_hist = []  # store recent Fi

        # Predictor-Corrector across stages (startup uses lower-step methods)
        for i in range(L - 1):
            fi = self.ode_eq(xs[i], y)

            if i == 0:
                # AB1 predictor + AM1 corrector (Heun)
                y_pred = y + h * fi
                f_next = self.ode_eq_pre(xs[i + 1], y_pred)
                y_next = y + (h / 2.0) * (fi + f_next)
            elif i == 1:
                # AB2 predictor + AM2 corrector
                f_im1 = f_hist[-1]
                y_pred = y + (h / 2.0) * (3.0 * fi - f_im1)
                f_next = self.ode_eq_pre(xs[i + 1], y_pred)
                y_next = y + (h / 12.0) * (5.0 * f_next + 8.0 * fi - f_im1)
            elif i == 2:
                # AB3 predictor + AM3 corrector
                f_im1, f_im2 = f_hist[-1], f_hist[-2]
                y_pred = y + (h / 12.0) * (23.0 * fi - 16.0 * f_im1 + 5.0 * f_im2)
                f_next = self.ode_eq_pre(xs[i + 1], y_pred)
                y_next = y + (h / 24.0) * (9.0 * f_next + 19.0 * fi - 5.0 * f_im1 + f_im2)
            else:
                # AB4 predictor + AM3 corrector (sliding window, FuseUNet-style)
                f_im1, f_im2, f_im3 = f_hist[-1], f_hist[-2], f_hist[-3]
                y_pred = y + (h / 24.0) * (55.0 * fi - 59.0 * f_im1 + 37.0 * f_im2 - 9.0 * f_im3)
                f_next = self.ode_eq_pre(xs[i + 1], y_pred)
                y_next = y + (h / 24.0) * (9.0 * f_next + 19.0 * fi - 5.0 * f_im1 + f_im2)

            f_hist.append(fi)
            if len(f_hist) > 4:
                f_hist = f_hist[-4:]
            y = y_next

        # Final explicit "calculator" step (AB) to obtain Y_final from Y_L and F_1:L
        f_last = self.ode_eq(xs[-1], y)

        if L == 2:
            f_im1 = f_hist[-1]
            y_final = y + (h / 2.0) * (3.0 * f_last - f_im1)  # AB2
        elif L == 3:
            f_im1, f_im2 = f_hist[-1], f_hist[-2]
            y_final = y + (h / 12.0) * (23.0 * f_last - 16.0 * f_im1 + 5.0 * f_im2)  # AB3
        else:
            # AB4 (for L >= 4)
            f_im1, f_im2, f_im3 = f_hist[-1], f_hist[-2], f_hist[-3]
            y_final = y + (h / 24.0) * (55.0 * f_last - 59.0 * f_im1 + 37.0 * f_im2 - 9.0 * f_im3)

        return y_final


class MDP(nn.Module):

    def __init__(
        self,
        in_channels,
        max_levels=3,
        use_ts_attention: bool = True,
        ts_heads: int = 8,
        ts_max_len: int = 4096,
        use_ode_fusion: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.max_levels = max_levels
        self.use_ts_attention = use_ts_attention
        self.ts_heads = ts_heads
        self.ts_max_len = ts_max_len
        self.use_ode_fusion = use_ode_fusion

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.dilations = [1, 2, 3]
        self.conv_weights = nn.Parameter(torch.ones(3))
        self.conv_diff = nn.Conv2d(in_channels, in_channels,
                                   kernel_size=3, padding=1,
                                   groups=in_channels)

        # Token-statistics spatial attention (optional)
        # We project to ts_heads channels, then use TSSA with head_dim=1 (dim==heads) to estimate membership.
        self.ts_proj = nn.Conv2d(in_channels, ts_heads, kernel_size=1, bias=False)
        self.ts_attn = TokenStatisticsSelfAttention(dim=ts_heads, heads=ts_heads, dropout=0.0, max_len=ts_max_len)

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

        group_num = math.gcd(in_channels, min(32, in_channels))
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=1,
                      groups=group_num,
                      bias=False),
            nn.GroupNorm(num_groups=group_num, num_channels=in_channels),
            nn.GELU()
        )

        # FuseUNet-inspired multi-scale ODE fusion (works on the same per-level Wi maps after resizing)
        # FuseUNet-inspired multi-scale ODE fusion (memory-efficient):
        # compress channels before ODE fusion, then project back.
        ode_dim = max(32, min(128, int(in_channels * 0.25)))
        self.ode_dim = ode_dim
        group_ode = math.gcd(ode_dim, min(32, ode_dim))
        self.ode_down = nn.Sequential(
            nn.Conv2d(in_channels, ode_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=group_ode, num_channels=ode_dim),
            nn.GELU(),
        )
        self.ode_fusion = MultiStepODEFusion2D(ode_dim, use_norm=True)
        self.ode_up = nn.Sequential(
            nn.Conv2d(ode_dim, in_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=group_num, num_channels=in_channels),
            nn.GELU(),
        )
        # learnable gate to blend ODE-fused feature into the original MDP output
        self.ode_gate = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2) ~ 0.12

        self.level_weights = nn.Parameter(torch.ones(1, max_levels, 1, 1))
        self.softmax = nn.Softmax(dim=0)
        self.softmax_level = nn.Softmax(dim=1)

        # Background-aware gating (learnable). Helps reduce MAE/MSE on sparse fields.
        self.bg_tau = nn.Parameter(torch.tensor(0.05))
        self.bg_k   = nn.Parameter(torch.tensor(10.0))

    def sqrt_abs_diff(self, x, y):
        return torch.sqrt(torch.abs(x - y) + 1e-6)

    def forward(self, x):
        conv_outs = []
        norm_weights = self.softmax(self.conv_weights)
        for i, dilation in enumerate(self.dilations):

            out = F.conv2d(
                x,
                self.base_conv.weight,
                bias=self.base_conv.bias,
                padding=dilation,
                dilation=dilation,
                groups=self.in_channels
            )
            conv_outs.append(out * norm_weights[i])

        diff_inputs = conv_outs
        diffs = []
        for i in range(len(diff_inputs)):
            for j in range(i + 1, len(diff_inputs)):
                diffs.append(self.sqrt_abs_diff(diff_inputs[i], diff_inputs[j]))

        active_levels = min(len(diffs), self.max_levels)
        if active_levels == 0:
            return torch.zeros_like(x)

        pyramid = []
        for i in range(active_levels):
            if i == 0:
                resized_diff = diffs[i]
            else:
                resized_diff = F.avg_pool2d(diffs[i], kernel_size=2 ** i)
            processed = self.conv_diff(resized_diff)
            pyramid.append(processed)

        weighted_features = []
        norm_level_weights = self.softmax_level(self.level_weights)[:, :active_levels]
        base_size = pyramid[0].shape[2:]

        for feat, weight in zip(pyramid, norm_level_weights.split(1, dim=1)):
            if self.use_ts_attention:
                # Token-statistics spatial saliency via membership concentration
                Bf, Cf, Hf, Wf = feat.shape
                w = self.ts_proj(feat)  # (B, ts_heads, H, W)
                w_flat = w.flatten(2).transpose(1, 2)  # (B, N, ts_heads)
                _y, Pi = self.ts_attn(w_flat)  # Pi: (B, ts_heads, N)

                # Pi.max is in [1/ts_heads, 1]. Normalize to [0, 1] for a stable attention scale.
                sal = Pi.max(dim=1).values  # (B, N)
                minv = 1.0 / float(self.ts_heads)
                sal = (sal - minv) / (1.0 - minv + 1e-6)
                sal = sal.clamp(0.0, 1.0)
                spatial_attn = sal.view(Bf, 1, Hf, Wf)
            else:
                avg_pool = torch.mean(feat, dim=1, keepdim=True)
                max_pool, _ = torch.max(feat, dim=1, keepdim=True)
                spatial_attn = self.spatial_attention(torch.cat([avg_pool, max_pool], dim=1))

            if feat.shape[2:] != base_size:
                spatial_attn = F.interpolate(spatial_attn, size=base_size, mode='bilinear', align_corners=False)
                feat = F.interpolate(feat, size=base_size, mode='bilinear', align_corners=False)

            weighted = spatial_attn * feat * weight
            weighted_features.append(weighted)

        # ===== Original MDP fusion (concat + conv) =====
        if len(weighted_features) < self.max_levels:
            dummy = torch.zeros_like(weighted_features[0])
            weighted_features_padded = weighted_features + [dummy] * (self.max_levels - len(weighted_features))
        else:
            weighted_features_padded = weighted_features

        # Background-aware mask: suppress perturbations on near-zero/background regions
        bg = x.abs().mean(dim=1, keepdim=True)  # (B,1,H,W)
        k = self.bg_k.clamp(1.0, 50.0)
        tau = self.bg_tau.clamp(-1.0, 1.0)
        bg_mask = torch.sigmoid(k * (bg - tau))
        if bg_mask.shape[2:] != base_size:
            bg_mask = F.interpolate(bg_mask, size=base_size, mode='bilinear', align_corners=False)

        # Memory-friendly fusion: sum instead of cat (keeps channels at C)
        fused = weighted_features_padded[0]
        for f in weighted_features_padded[1:]:
            fused = fused + f
        fused = (fused / float(self.max_levels)) * bg_mask
        fused_base = self.fusion_conv(fused)

        # ===== FuseUNet-style ODE fusion on multi-scale Wi maps (coarse->fine) =====
        if self.use_ode_fusion and active_levels >= 2:
            ode_seq = list(reversed(weighted_features[:active_levels]))  # coarse -> fine
            ode_small_seq = [self.ode_down(w) for w in ode_seq]
            fused_ode_small = self.ode_fusion(ode_small_seq)
            fused_ode = self.ode_up(fused_ode_small)
            gate = torch.sigmoid(self.ode_gate)
            return fused_base + gate * fused_ode

        return fused_base


class FreqBandResidualMix(nn.Module):
    """
    Frequency-Dynamic Residual Mixing (FBRM) - a light-weight, OOM-safe module inspired by
    'Frequency Dynamic Convolution for Dense Image Prediction' (CVPR 2025).

    Key idea we borrow (NOT a direct copy):
    - Use frequency-related cues to modulate convolutional responses.
    Our implementation approximates 'frequency decomposition/modulation' by:
    (1) estimating per-channel low/mid/high frequency energies using pooled residuals + Laplacian,
    (2) mixing multiple depthwise conv branches (different receptive fields) conditioned on those energies,
    (3) applying a conservative residual (LayerScale) + background-aware mask to avoid harming sparse targets.

    Input/Output: x in (B, C, H, W)
    """
    def __init__(self, dim: int, hidden_ratio: float = 0.25, k_small: int = 3, k_large: int = 5, dil_large: int = 2):
        super().__init__()
        self.dim = dim

        # depthwise branches (different implicit frequency responses)
        self.dw_small = nn.Conv2d(dim, dim, kernel_size=k_small, padding=k_small // 2, groups=dim, bias=False)
        self.dw_large = nn.Conv2d(dim, dim, kernel_size=3, padding=dil_large, dilation=dil_large, groups=dim, bias=False)

        # pointwise fusion
        self.pw = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        group_num = math.gcd(dim, min(32, dim))
        self.norm = nn.GroupNorm(num_groups=group_num, num_channels=dim)
        self.act = nn.GELU()

        # fixed Laplacian (depthwise) for high-frequency proxy (trace-friendly)
        lap = torch.tensor([[0., -1., 0.],
                            [-1., 4., -1.],
                            [0., -1., 0.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("_lap_kernel", lap, persistent=False)

        stat_dim = 4  # [E_low, E_mid, E_high, ratio]
        hidden = max(8, int(dim * hidden_ratio))
        # channel-wise gating: apply an MLP to each channel's stats (B, C, stat_dim) -> (B, C, 2)
        self.gate_fc1 = nn.Linear(stat_dim, hidden, bias=True)
        self.gate_fc2 = nn.Linear(hidden, 2, bias=True)

        # background-aware mask params (suppress perturbation on near-zero regions)
        self.bg_k = nn.Parameter(torch.tensor(10.0))
        self.bg_tau = nn.Parameter(torch.tensor(0.18))  # heuristic init (your earlier radar bias)

        # LayerScale for safe residual
        self.gamma = nn.Parameter(torch.ones(dim) * 1e-3)

        self.reset_parameters()

    def reset_parameters(self):
        # safe init: small residual and well-behaved gates
        nn.init.kaiming_normal_(self.dw_small.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.dw_large.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.pw.weight, mode='fan_out')
        nn.init.constant_(self.gate_fc1.bias, 0.0)
        nn.init.constant_(self.gate_fc2.bias, 0.0)
        nn.init.normal_(self.gate_fc1.weight, std=0.02)
        nn.init.normal_(self.gate_fc2.weight, std=0.02)

    def _band_stats(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> stats: (B, C, 4)
        # low-frequency proxy: local average
        low = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        # mid: difference of two smoothings (band-pass proxy)
        mid = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1) - F.avg_pool2d(x, kernel_size=7, stride=1, padding=3)
        # high: Laplacian magnitude
        B, C, H, W = x.shape
        lap_k = self._lap_kernel.repeat(C, 1, 1, 1)  # (C,1,3,3)
        high = F.conv2d(x, lap_k, padding=1, groups=C)

        e_low = low.abs().mean(dim=(2, 3))
        e_mid = mid.abs().mean(dim=(2, 3))
        e_high = high.abs().mean(dim=(2, 3))
        ratio = e_high / (e_low + 1e-6)
        stats = torch.stack([e_low, e_mid, e_high, ratio], dim=-1)
        # normalize across stats dim for stability (per channel)
        stats = stats / (stats.mean(dim=-1, keepdim=True) + 1e-6)
        return stats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gating weights
        stats = self._band_stats(x)  # (B, C, 4)
        g = self.gate_fc2(self.act(self.gate_fc1(stats)))  # (B, C, 2)
        w = torch.softmax(g, dim=-1)  # mixture weights

        y_small = self.dw_small(x)
        y_large = self.dw_large(x)

        # mix branches per channel without concatenation (OOM-safe)
        w1 = w[..., 0].unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        w2 = w[..., 1].unsqueeze(-1).unsqueeze(-1)
        y = w1 * y_small + w2 * y_large

        y = self.pw(y)
        y = self.act(self.norm(y))

        # background mask
        bg = x.abs().mean(dim=1, keepdim=True)  # (B,1,H,W)
        k = self.bg_k.clamp(1.0, 50.0)
        tau = self.bg_tau.clamp(-1.0, 1.0)
        bg_mask = torch.sigmoid(k * (bg - tau))

        return (y * bg_mask) * self.gamma.view(1, self.dim, 1, 1)


class ViTSubBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=2., drop=0., drop_path=0.1, max_diff_levels=3, krylov_dim=32, order=3, rank=8, heads=4):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)

        # KAN/MPA 模块
        self.MPA = MPA(
            dim=dim,
            krylov_dim=krylov_dim,
            order=order,
            rank=rank,
            heads=heads
        )

        self.diff = MDP(
            in_channels=dim,
            max_levels=max_diff_levels
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # LayerScale: stabilize residual magnitudes (often improves MAE on sparse targets)
        self.gamma_diff = nn.Parameter(torch.ones(dim) * 1e-3)
        self.gamma_mpa  = nn.Parameter(torch.ones(dim) * 1e-3)
        self.fbrm = FreqBandResidualMix(dim)
        self.gamma_fbrm = nn.Parameter(torch.ones(dim) * 1.0)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        x_orig = x

        x_norm = self.norm1(x.flatten(2).transpose(1, 2))
        x_norm = x_norm.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        attn_out = self.diff(x_norm)
        attn_out = attn_out * self.gamma_diff.view(1, C, 1, 1)
        x = x_orig + self.drop_path(attn_out)
        x_flat = x.flatten(2).transpose(1, 2)
        mpa_out = self.MPA(self.norm1(x_flat))
        mpa_out = mpa_out * self.gamma_mpa.view(1, 1, C)
        x = x_flat + self.drop_path(mpa_out)
        
        x2d = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x2d = x2d + self.drop_path(self.fbrm(x2d) * self.gamma_fbrm.view(1, C, 1, 1))
        return x2d
    
          
class TemporalAttention(nn.Module):
    """A Temporal Attention block for Temporal Attention Unit"""

    def __init__(self, d_model, kernel_size=21, attn_shortcut=True):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)  
        self.activation = nn.GELU()                 
        self.spatial_gating_unit = TemporalAttentionModule(d_model, kernel_size)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)        
        self.attn_shortcut = attn_shortcut

    def forward(self, x):
        if self.attn_shortcut:
            shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        if self.attn_shortcut:
            x = x + shortcut
        return x
    

class TemporalAttentionModule(nn.Module):
    """Large Kernel Attention for SimVP"""

    def __init__(self, dim, kernel_size, dilation=3, reduction=16):
        super().__init__()
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        dd_p = (dilation * (dd_k - 1) // 2)

        self.conv0 = nn.Conv2d(dim, dim, d_k, padding=d_p, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, dd_k, stride=1, padding=dd_p, groups=dim, dilation=dilation)
        self.conv1 = nn.Conv2d(dim, dim, 1)

        self.reduction = max(dim // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // self.reduction, bias=False), # reduction
            nn.ReLU(True),
            nn.Linear(dim // self.reduction, dim, bias=False), # expansion
            nn.Sigmoid()
        )

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)           
        attn = self.conv_spatial(attn) 
        f_x = self.conv1(attn)         

        b, c, _, _ = x.size()
        se_atten = self.avg_pool(x).view(b, c)
        se_atten = self.fc(se_atten).view(b, c, 1, 1)
        return se_atten * f_x * u
    
class AttentionModule(nn.Module):
    """Large Kernel Attention for SimVP"""

    def __init__(self, dim, kernel_size, dilation=3):
        super().__init__()
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        dd_p = (dilation * (dd_k - 1) // 2)

        self.conv0 = nn.Conv2d(dim, dim, d_k, padding=d_p, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, dd_k, stride=1, padding=dd_p, groups=dim, dilation=dilation)
        self.conv1 = nn.Conv2d(dim, 2*dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)           # depth-wise conv
        attn = self.conv_spatial(attn) # depth-wise dilation convolution
        
        f_g = self.conv1(attn)
        split_dim = f_g.shape[1] // 2
        f_x, g_x = torch.split(f_g, split_dim, dim=1)
        return torch.sigmoid(g_x) * f_x


class SpatialAttention(nn.Module):
    """A Spatial Attention block for SimVP"""

    def __init__(self, d_model, kernel_size=21, attn_shortcut=True):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)         # 1x1 conv
        self.activation = nn.GELU()                          # GELU
        self.spatial_gating_unit = AttentionModule(d_model, kernel_size)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)         # 1x1 conv
        self.attn_shortcut = attn_shortcut

    def forward(self, x):
        if self.attn_shortcut:
            shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        if self.attn_shortcut:
            x = x + shortcut
        return x


class GASubBlock(nn.Module):
    """A GABlock (gSTA) for SimVP"""

    def __init__(self, dim, kernel_size=21, mlp_ratio=4.,
                 drop=0., drop_path=0.1, init_value=1e-2, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim, kernel_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MixMlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.layer_scale_1 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'layer_scale_1', 'layer_scale_2'}

    def forward(self, x):
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x

class TAUSubBlock(GASubBlock):
    """A TAUBlock (tau) for Temporal Attention Unit"""

    def __init__(self, dim, kernel_size=21, mlp_ratio=4.,
                 drop=0., drop_path=0.1, init_value=1e-2, act_layer=nn.GELU):
        super().__init__(dim=dim, kernel_size=kernel_size, mlp_ratio=mlp_ratio,
                 drop=drop, drop_path=drop_path, init_value=init_value, act_layer=act_layer)
        
        self.attn = TemporalAttention(dim, kernel_size)

