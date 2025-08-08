import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import math
from typing import Callable
from functools import partial

# Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

# Channel Attention (CBAM style) Layer
class CBAMLayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAMLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class VSSBlock(nn.Module):
    def __init__(
            self,
            in_channels: int = 0,
            out_channels: int = 0,
            drop_path: float = 0,
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            is_light_sr: bool = False,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(in_channels)
        self.self_attention = SS2D(d_model=in_channels, d_state=d_state,expand=expand, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(in_channels))
        self.conv_blk = CBAMLayer(in_channels)
        self.ln_2 = nn.LayerNorm(in_channels)
        self.skip_scale2 = nn.Parameter(torch.ones(in_channels))

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

    def forward(self, input):
        input = input.permute(0, 2, 3, 1).contiguous()
        x = self.ln_1(input)
        x = input*self.skip_scale + self.drop_path(self.self_attention(x))
        x = x*self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = x.permute(0, 3, 1, 2).contiguous()
        return self.shortcut(x)
    
class AttentionBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))

        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dropout: float,
                 n_groups: int = 32,
                 has_attn: bool = False,
                 use_lora: bool = False,
                 lora_r: int = 4,
                 lora_alpha: float = 1.0):
        super().__init__()

        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = nn.SiLU()

        conv_class = LoRAConv2d if use_lora else nn.Conv2d
        conv1_kwargs = dict(kernel_size=3, padding=1)
        self.conv1 = conv_class(in_channels, out_channels, **conv1_kwargs) \
            if not use_lora else conv_class(in_channels, out_channels, **conv1_kwargs, r=lora_r, alpha=lora_alpha)

        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        self.conv2 = conv_class(out_channels, out_channels, **conv1_kwargs) \
            if not use_lora else conv_class(out_channels, out_channels, **conv1_kwargs, r=lora_r, alpha=lora_alpha)

        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else nn.Identity()

        self.attn = VSSBlock(out_channels, out_channels, drop_path=dropout) if has_attn else nn.Identity()

    def forward(self, x):
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return self.attn(h + self.shortcut(x))
    
class LoRAConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=4, alpha=1.0,
                 stride=1, padding=1, bias=True):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=bias)
        for param in self.conv.parameters():
            param.requires_grad = False

        self.lora_A = nn.Conv2d(in_channels, r, kernel_size=1, bias=False)
        self.lora_B = nn.Conv2d(r, out_channels, kernel_size=1, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.conv(x) + self.scaling * self.lora_B(self.lora_A(x))


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, has_attn: bool, dropout: int):
        super().__init__()
        self.res = ResidualBlock(
            in_channels, out_channels, dropout=dropout, has_attn=has_attn)

    def forward(self, x: torch.Tensor):
        return self.res(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, has_attn: bool, dropout: int):
        super().__init__()
        self.res = ResidualBlock(
            in_channels, out_channels, dropout=dropout, has_attn=has_attn)

    def forward(self, x: torch.Tensor):
        return self.res(x)


class MiddleBlock(nn.Module):
    def __init__(self, n_channels: int, dropout: int):
        super().__init__()
        self.res1 = ResidualBlock(
            n_channels, n_channels, dropout=dropout, has_attn=True)
        self.res2 = ResidualBlock(n_channels, n_channels, dropout=dropout)

    def forward(self, x: torch.Tensor):
        x = self.res1(x)
        x = self.res2(x)
        return x


class Downsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels,
                              kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.convT = nn.ConvTranspose2d(
            n_channels, n_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv = nn.Conv2d(n_channels, n_channels,
                              kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor):
        # Bx, Cx, Hx, Wx = x.size()
        # x = F.interpolate(x, size=(2*Hx, 2*Wx), mode='bicubic', align_corners=False)
        return self.conv(self.convT(x))


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class MAMBAUNET(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 n_features: int = 64,
                 dropout: int = 0.1,
                 block_out_channels=[64, 128, 128, 256],
                 layers_per_block=2,
                 is_attn_layers=(False, False, True, False),
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_features = n_features
        self.dropout = dropout
        self.block_out_channels = block_out_channels
        self.layers_per_block = layers_per_block
        self.is_attn_layers = is_attn_layers

        self.sub_mean = MeanShift(255)
        self.add_mean = MeanShift(255, sign=1)

        self.shallow_feature_extraction = nn.Conv2d(
            in_channels, n_features, kernel_size=3, padding=1)
        self.image_rescontruction = nn.Conv2d(
            n_features, in_channels, kernel_size=3, padding=1)

        self.left_model = self.left_unet()
        self.middle_model = MiddleBlock(
            block_out_channels[-1], dropout=self.dropout)
        self.right_model = self.right_unet()

    def left_unet(self):
        left_model = []

        in_channel = out_channel = self.n_features
        for i in range(len(self.block_out_channels)):
            out_channel = self.block_out_channels[i]

            down_block = [DownBlock(in_channel, out_channel, dropout=self.dropout, has_attn=self.is_attn_layers[i])] \
                + [DownBlock(out_channel, out_channel, dropout=self.dropout,
                             has_attn=self.is_attn_layers[i])] * (self.layers_per_block - 1)
            in_channel = out_channel
            left_model.append(nn.Sequential(*down_block))
            if i < len(self.block_out_channels):
                left_model.append(Downsample(out_channel))

        return nn.ModuleList(left_model)

    def right_unet(self):
        right_unet = []

        in_channel = out_channel = self.block_out_channels[-1]
        for i in reversed(range(len(self.block_out_channels))):

            out_channel = self.block_out_channels[i]

            up_block = [UpBlock(in_channel, out_channel, dropout=self.dropout, has_attn=self.is_attn_layers[i - 1])] \
                + [UpBlock(out_channel, out_channel, dropout=self.dropout, has_attn=self.is_attn_layers[i - 1])
                   ] * (self.layers_per_block - 1)

            in_channel = out_channel * 2
            right_unet.append(nn.Sequential(*up_block))
            right_unet.append(Upsample(out_channel))

        in_channel, out_channel = self.block_out_channels[0] * \
            2, self.n_features
        up_block = [UpBlock(in_channel, out_channel, dropout=self.dropout, has_attn=self.is_attn_layers[0])] \
            + [UpBlock(out_channel, out_channel, dropout=self.dropout, has_attn=self.is_attn_layers[0])
               ] * (self.layers_per_block - 1)
        right_unet.append(nn.Sequential(*up_block))
        return nn.ModuleList(right_unet)

    def forward(self, x):
        x = x * 255
        x = self.sub_mean(x)

        feature_maps = self.shallow_feature_extraction(x)
        feature_x = [feature_maps]
        # print(feature_maps.shape)
        feature_block = feature_maps
        for block in self.left_model:
            feature_block = block(feature_block)
            if not isinstance(block, Downsample):
                # print(feature_block.shape)
                feature_x.append(feature_block)

        bottleneck = self.middle_model(feature_block)

        feature_x.reverse()
        # print('Middle::: ', feature_maps.shape)

        recover = bottleneck
        d = 0
        for block in self.right_model:
            if isinstance(block, Upsample):
                # print('UP-CAT::: ', recover.shape)
                recover = block(recover)
                # print('UP-CAT-END::: ', recover.shape, feature_x[d].shape)
                recover = torch.cat([recover, feature_x[d]], 1)
                # print('UP-CAT-END::: ', recover.shape, feature_x[d].shape)
                d += 1
            else:
                recover = block(recover)
                # print('UP-RES::: ', recover.shape)

        recover = self.image_rescontruction(recover)
        recover = self.add_mean(recover) / 255
        return recover