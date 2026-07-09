import torch
from torch import nn

from openstl.modules import (ConvSC, ConvNeXtSubBlock, ConvMixerSubBlock, GASubBlock, gInception_ST,
                             HorNetSubBlock, MLPMixerSubBlock, MogaSubBlock, PoolFormerSubBlock,
                             SwinSubBlock, UniformerSubBlock, VANSubBlock, ViTSubBlock, TAUSubBlock)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import List

class SFFE(nn.Module):
    def __init__(self, C_in, C_out, n_bands=4, use_ts_band_gate: bool = True):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.n_bands = n_bands
        self.use_ts_band_gate = use_ts_band_gate
        kernel_sizes = [3, 5, 7, 11][:n_bands]
        self.filters = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(C_in, C_in, kernel_size=k, padding=k // 2, groups=C_in, bias=False),
                nn.BatchNorm1d(C_in),
                nn.GELU()
            ) for k in kernel_sizes
        ])
        self.scale_weights = nn.Parameter(torch.ones(n_bands))
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # ---- Innovation-2 (SFFE): Dynamic Cross-Scale Spectral Fusion (DCSF) ----
        # per-sample dynamic re-weighting over frequency bands + dynamic conv/freq blending
        # gates are initialized to 0 so the module starts as the original Innovation-1 behavior.
        self.dcsf_scale_gate = nn.Parameter(torch.zeros(1))
        self.dcsf_alpha_gate = nn.Parameter(torch.zeros(1))

        self.dcsf_mlp = nn.Sequential(
            nn.LayerNorm(n_bands),
            nn.Linear(n_bands, n_bands),
            nn.GELU(),
            nn.Linear(n_bands, n_bands),
        )
        self.dcsf_alpha_mlp = nn.Sequential(
            nn.LayerNorm(n_bands),
            nn.Linear(n_bands, n_bands),
            nn.GELU(),
            nn.Linear(n_bands, 1),
        )
        nn.init.zeros_(self.dcsf_mlp[-1].weight)
        nn.init.zeros_(self.dcsf_mlp[-1].bias)
        nn.init.zeros_(self.dcsf_alpha_mlp[-1].weight)
        nn.init.zeros_(self.dcsf_alpha_mlp[-1].bias)

        # temperature for band attention (init=1.0)
        self.log_tau = nn.Parameter(torch.zeros(1))
        self.proj = nn.Linear(C_in * n_bands, C_out)
        # Background-aware gating (learnable)
        self.bg_tau = nn.Parameter(torch.tensor(0.05))
        self.bg_k   = nn.Parameter(torch.tensor(10.0))

    def forward(self, x):
        B, C, H, W = x.shape
        L = H * W
        x_seq = x.view(B, C, L).transpose(1, 2)
        x_conv_in = x_seq.transpose(1, 2)
        norm_weights = torch.softmax(self.scale_weights, dim=0)
        
        conv_bands = []
        for i, filt in enumerate(self.filters):
            b_conv = filt(x_conv_in).transpose(1, 2) * norm_weights[i]
            conv_bands.append(b_conv)
        target_lengths = [b.size(1) for b in conv_bands]
        
        x_freq_domain = torch.fft.rfft(x_seq, dim=1)
        freqs = torch.fft.rfftfreq(L, device=x.device)
        n_freqs = len(freqs)
        band_size = n_freqs // self.n_bands

        freq_bands = []
        band_energy_list = []
        for i in range(self.n_bands):
            mask = torch.zeros_like(freqs, dtype=torch.bool)
            start = i * band_size
            end = (i + 1) * band_size if i < self.n_bands - 1 else n_freqs
            # energy of this band in frequency domain (B,)
            band_spec = x_freq_domain[:, start:end, :]  # complex
            band_energy_list.append((band_spec.abs() ** 2).mean(dim=(1, 2)))
            mask[start:end] = True
            band_freq = x_freq_domain * mask.unsqueeze(-1).to(x_freq_domain.dtype)
            band_time = torch.fft.irfft(band_freq, n=L, dim=1) # [B, L, C]
            if target_lengths is not None:
                length = target_lengths[i]
                band_time_resized = F.interpolate(
                    band_time.transpose(1, 2),
                    size=length,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            else:
                band_time_resized = F.avg_pool1d(
                    band_time.transpose(1, 2), kernel_size=2, stride=2
                ).transpose(1, 2)
            
            freq_bands.append(band_time_resized)
        band_energy = torch.stack(band_energy_list, dim=1)  # (B, n_bands)
        # normalize energy to reduce scale sensitivity across batches
        band_energy = band_energy / band_energy.mean(dim=1, keepdim=True).clamp_min(1e-6)
        # DCSF: per-sample dynamic conv/freq blending (starts as scalar alpha)
        dcsf_in = torch.log1p(band_energy)
        alpha_gate = torch.tanh(self.dcsf_alpha_gate)  # scalar in [-1,1], 0 at init
        alpha_base = self.alpha.clamp(0.0, 1.0)       # keep Innovation-1 behavior at init
        alpha_delta = torch.tanh(self.dcsf_alpha_mlp(dcsf_in)).squeeze(-1)  # (B,), [-1,1]
        alpha_val = (alpha_base + 0.25 * alpha_gate * alpha_delta).clamp(0.0, 1.0).view(B, 1, 1)
        fused_raw = [alpha_val * c + (1 - alpha_val) * f for c, f in zip(conv_bands, freq_bands)]
        # Token-statistics adaptive band gating (memory-safe)
        # Previous versions used cumsum(w_sq) which creates extra B×L×C tensors and can OOM.
        # Here we use lightweight token statistics (abs-mean) to score each band (B,).
        if self.use_ts_band_gate:
            # scores: (B, n_bands)
            score_mat = torch.stack(
                [f.abs().mean(dim=(1, 2)) for f in fused_raw], dim=1
            )
            # stabilize: normalize per-sample scale and inject a small spectral-energy prior
            score_mat = score_mat / score_mat.mean(dim=1, keepdim=True).clamp_min(1e-6)
            score_mat = score_mat + 0.25 * torch.log1p(band_energy)

            # DCSF: dynamic band prior (adds a small learnable bias to band scores)
            scale_gate = torch.tanh(self.dcsf_scale_gate)
            score_bias = self.dcsf_mlp(dcsf_in)  # (B, n_bands), last layer is zero-init
            score_mat = score_mat + scale_gate * score_bias

            tau = torch.exp(self.log_tau).clamp_min(0.5).clamp_max(5.0)
            band_pi = torch.softmax(score_mat / tau, dim=1)

            # in-place weighting (saves memory)
            fused = fused_raw
            for i in range(self.n_bands):
                fused[i] = fused[i] * band_pi[:, i].view(B, 1, 1)
        else:
            fused = fused_raw

        x_fused = torch.cat(fused, dim=-1)
        x_proj = self.proj(x_fused)
        # Suppress background disturbance (important for sparse precipitation fields)
        bg = x.abs().mean(dim=1, keepdim=True)  # (B,1,H,W)
        k = self.bg_k.clamp(1.0, 50.0)
        tau = self.bg_tau.clamp(-1.0, 1.0)
        bg_mask = torch.sigmoid(k * (bg - tau)).view(B, 1, L).transpose(1, 2)  # (B,L,1)
        x_proj = x_proj * bg_mask
        out = x_proj.transpose(1, 2).view(B, self.C_out, H, W)
        return out
class Encoder(nn.Module):
    def __init__(self, C_in, C_hid, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S)
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
            SFFE(C_in, C_hid),
            *[SFFE(C_hid, C_hid) for _ in samplings[1:]]
        )

    def forward(self, x):
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1

class Decoder(nn.Module):
    def __init__(self, C_hid, C_out, N_S, spatio_kernel, act_inplace=True):
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            *[SFFE(C_hid, C_hid) for _ in range(N_S)]
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(len(self.dec) - 1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](hid + enc1)
        Y = self.readout(Y)
        return Y

# ---------- 7. SimVP 主模型 ----------
class SimVP_Model(nn.Module):
    def __init__(self, in_shape, hid_S=16, hid_T=256, N_S=4, N_T=4, model_type='gSTA',
                 mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, act_inplace=True, **kwargs):
        super(SimVP_Model, self).__init__()
        T, C, H, W = in_shape
        H, W = int(H / 2**(N_S/2)), int(W / 2**(N_S/2))
        act_inplace = False
        C = 1

        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc, act_inplace)
        self.dec = Decoder(hid_S, C, N_S, spatio_kernel_dec, act_inplace)

        model_type = 'gsta' if model_type is None else model_type.lower()
        if model_type == 'incepu':
            self.hid = MidIncepNet(T * hid_S, hid_T, N_T)
        else:
            self.hid = MidMetaNet(T * hid_S, hid_T, N_T,
                                  input_resolution=(H, W), model_type=model_type,
                                  mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)

    def forward(self, x_raw, **kwargs):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B * T, C, H, W)
        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        use_ckpt = self.training and (not torch.jit.is_tracing()) and (not torch.jit.is_scripting())
        hid = checkpoint(self.hid, z, use_reentrant=False) if use_ckpt else self.hid(z)
        hid = hid.view(B * T, C_, H_, W_)
        Y = checkpoint(lambda a, b: self.dec(a, b), hid, skip, use_reentrant=False) if use_ckpt else self.dec(hid, skip)
        Y = Y.view(B, T, C, H, W)
        return Y

# ---------- 8. Sampling 控制函数 ----------
def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    return list(reversed(samplings[:N])) if reverse else samplings[:N]


class MidIncepNet(nn.Module):
    """The hidden Translator of IncepNet for SimVPv1"""

    def __init__(self, channel_in, channel_hid, N2, incep_ker=[3,5,7,11], groups=8, **kwargs):
        super(MidIncepNet, self).__init__()
        assert N2 >= 2 and len(incep_ker) > 1
        self.N2 = N2
        enc_layers = [gInception_ST(
            channel_in, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1,N2-1):
            enc_layers.append(
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        enc_layers.append(
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        dec_layers = [
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups)]
        for i in range(1,N2-1):
            dec_layers.append(
                gInception_ST(2*channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        dec_layers.append(
                gInception_ST(2*channel_hid, channel_hid//2, channel_in,
                              incep_ker=incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        # encoder
        skips = []
        z = x
        for i in range(self.N2):
            z = self.enc[i](z)
            if i < self.N2-1:
                skips.append(z)
        # decoder
        z = self.dec[0](z)
        for i in range(1,self.N2):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1) )

        y = z.reshape(B, T, C, H, W)
        return y


class MetaBlock(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, in_channels, out_channels, input_resolution=None, model_type=None,
                 mlp_ratio=8., drop=0.0, drop_path=0.0, layer_i=0):
        super(MetaBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        model_type = model_type.lower() if model_type is not None else 'gsta'

        if model_type == 'gsta':
            self.block = GASubBlock(
                in_channels, kernel_size=21, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        elif model_type == 'convmixer':
            self.block = ConvMixerSubBlock(in_channels, activation=nn.GELU)
        elif model_type == 'convnext':
            self.block = ConvNeXtSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'hornet':
            self.block = HorNetSubBlock(in_channels, mlp_ratio=mlp_ratio, drop_path=drop_path)
        elif model_type in ['mlp', 'mlpmixer']:
            self.block = MLPMixerSubBlock(
                in_channels, input_resolution, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type in ['moga', 'moganet']:
            self.block = MogaSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop_rate=drop, drop_path_rate=drop_path,hidden_size=256)
        elif model_type == 'poolformer':
            self.block = PoolFormerSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'swin':
            self.block = SwinSubBlock(
                in_channels, input_resolution, layer_i=layer_i, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path)
        elif model_type == 'uniformer':
            block_type = 'MHSA' if in_channels == out_channels and layer_i > 0 else 'Conv'
            self.block = UniformerSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop,
                drop_path=drop_path, block_type=block_type)
        elif model_type == 'van':
            self.block = VANSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        elif model_type == 'vit':
            self.block = ViTSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'tau':
            self.block = TAUSubBlock(
                in_channels, kernel_size=21, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        else:
            assert False and "Invalid model_type in SimVP"

        if in_channels != out_channels:
            self.reduction = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        z = self.block(x)
        
        return z if self.in_channels == self.out_channels else self.reduction(z)


class MidMetaNet(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, channel_in, channel_hid, N2,
                 input_resolution=None, model_type=None,
                 mlp_ratio=4., drop=0.0, drop_path=0.1):
        super(MidMetaNet, self).__init__()
        assert N2 >= 1 and mlp_ratio > 1  # 修改断言，允许N2>=1
        self.N2 = N2
        dpr = [x.item() for x in torch.linspace(1e-2, drop_path, max(1, self.N2))]  # 确保dpr至少有1个元素

        enc_layers = []
        if N2 == 1:
            print("N2=",N2)
            # 当N2=1时，只使用一个中间层，保持输入输出通道相同
            enc_layers.append(MetaBlock(
                channel_in, channel_in, input_resolution, model_type,
                mlp_ratio, drop, drop_path=dpr[0], layer_i=0))
        else:
            # 原始逻辑，N2>=2时
            # downsample
            print("N2=",N2)
            enc_layers.append(MetaBlock(
                channel_in, channel_hid, input_resolution, model_type,
                mlp_ratio, drop, drop_path=dpr[0], layer_i=0))
            # middle layers
            for i in range(1, N2-1):
                enc_layers.append(MetaBlock(
                    channel_hid, channel_hid, input_resolution, model_type,
                    mlp_ratio, drop, drop_path=dpr[i], layer_i=i))
            # upsample
            enc_layers.append(MetaBlock(
                channel_hid, channel_in, input_resolution, model_type,
                mlp_ratio, drop, drop_path=dpr[-1], layer_i=N2-1))
        
        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        #print("x", x.shape)
        x = x.reshape(B, T*C, H, W)
        #print("x,met", x.shape)
        z = x
        
        for i in range(self.N2):
            #print("z before",z.shape)
            z = self.enc[i](z)
            #print("z after",z.shape)

        y = z.reshape(B, T, C, H, W)
        return y
