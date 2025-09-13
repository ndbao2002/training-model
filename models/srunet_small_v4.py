import torch
import torch.nn as nn
import torch.nn.functional as F

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

# Channel Attention (CBAM style) Layer
class CALayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CALayer, self).__init__()
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

class SALayer(nn.Module):
    def __init__(self, kernel_size=7):
        super(SALayer, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return x * self.sigmoid(out)

class CBAMLayer(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16, kernel_size=7):
        super(CBAMLayer, self).__init__()
        self.channel_att = CALayer(in_channels, reduction)
        self.spatial_att = SALayer(kernel_size)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return self.shortcut(x)

class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dropout: float,
                 n_groups: int = 32,
                 has_attn: bool = False,
                 adaptive_weight: bool = True,
                 fixed_weight_value: float = 1.0,
                 local_conv: str = 'conv_1x1'):
        super().__init__()

        self.cbam = CBAMLayer(in_channels, out_channels)
        if local_conv == 'conv_1x1':
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        elif local_conv == 'conv_3x3':
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        else:
            raise NotImplementedError()

        self.norm1 = nn.GroupNorm(n_groups, out_channels)
        self.norm2 = nn.GroupNorm(n_groups, out_channels)

        self.dropout = nn.Dropout(dropout)

        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else nn.Identity()

        self.attn = AttentionBlock(out_channels) if has_attn else nn.Identity()
        if adaptive_weight:
            self.weight = nn.Parameter(torch.ones(1))
        else:
            self.weight = fixed_weight_value

    def forward(self, x):
        h1 = self.norm1(F.silu(self.conv(x)))
        h2 = self.norm2(F.silu(self.cbam(x)))
        h = self.dropout(h1 + h2)
        return self.attn(h) * self.weight + self.shortcut(x) 

class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, has_attn: bool, dropout: int, 
                 adaptive_weight: bool = True, fixed_weight_value: float = 1.0, local_conv: str = 'conv_1x1',
                 layers_per_block: int = 4):
        super().__init__()

        self.res = nn.Sequential(*[ResidualBlock(
            in_channels if i == 0 else out_channels,
            out_channels,
            dropout=dropout,
            has_attn=has_attn,
            adaptive_weight=adaptive_weight,
            fixed_weight_value=fixed_weight_value,
            local_conv=local_conv
        ) for i in range(layers_per_block)])

        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else nn.Identity()
        
        if adaptive_weight:
            self.weight = nn.Parameter(torch.ones(1))
        else:
            self.weight = fixed_weight_value

    def forward(self, x: torch.Tensor):
        return self.res(x) * self.weight + self.shortcut(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, has_attn: bool, dropout: int, 
                 adaptive_weight: bool = True, fixed_weight_value: float = 1.0, local_conv: str = 'conv_1x1',
                 layers_per_block: int = 4):
        super().__init__()
        self.res = nn.Sequential(*[ResidualBlock(
            in_channels if i == 0 else out_channels,
            out_channels,
            dropout=dropout,
            has_attn=has_attn,
            adaptive_weight=adaptive_weight,
            fixed_weight_value=fixed_weight_value,
            local_conv=local_conv
        ) for i in range(layers_per_block)])

        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else nn.Identity()
        
        if adaptive_weight:
            self.weight = nn.Parameter(torch.ones(1))
        else:
            self.weight = fixed_weight_value

    def forward(self, x: torch.Tensor):
        return self.res(x) * self.weight + self.shortcut(x)


class MiddleBlock(nn.Module):
    def __init__(self, n_channels: int, dropout: int, num_layers: int, adaptive_weight: bool = True, fixed_weight_value: float = 1.0, bottleneck_attention: bool = False, local_conv: str = 'conv_1x1'):
        super().__init__()
        self.model = nn.Sequential(
            *[ResidualBlock(n_channels,
                            n_channels,
                            dropout=dropout,
                            has_attn=bottleneck_attention,
                            adaptive_weight=adaptive_weight,
                            fixed_weight_value=fixed_weight_value,
                            local_conv=local_conv) for _ in range(num_layers)]
        )

        if adaptive_weight:
            self.weight = nn.Parameter(torch.ones(1))
        else:
            self.weight = fixed_weight_value

    def forward(self, x: torch.Tensor):
        return self.model(x) * self.weight + x


class Downsample(nn.Module):
    def __init__(self, n_channels, downtype='conv_1x1'):
        super().__init__()
        if downtype == 'conv_3x3':
            self.conv = nn.Conv2d(n_channels, n_channels,
                                kernel_size=3, stride=2, padding=1)
        elif downtype == 'conv_1x1':
            self.conv = nn.Conv2d(n_channels, n_channels,
                                kernel_size=1, stride=2, padding=0)
        else:
            raise NotImplementedError()

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, n_channels, uptype='pixelshuffle_1x1'):
        super().__init__()
        if uptype == 'pixelshuffle_3x3':
            self.up = nn.Sequential(
                nn.Conv2d(n_channels, 4*n_channels, 3, 1, 1),
                nn.PixelShuffle(2)
            )
        elif uptype == 'pixelshuffle_1x1':
            self.up = nn.Sequential(
                nn.Conv2d(n_channels, 4*n_channels, 1, 1, 0),
                nn.PixelShuffle(2)
            )
        elif uptype == 'transpose':
            self.up = nn.Sequential(
                nn.ConvTranspose2d(n_channels, n_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.Conv2d(n_channels, n_channels, 3, 1, 1),
            )
        else: raise NotImplementedError()

    def forward(self, x: torch.Tensor):
        # Bx, Cx, Hx, Wx = x.size()
        # x = F.interpolate(x, size=(2*Hx, 2*Wx), mode='bicubic', align_corners=False)
        return self.up(x)


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

class SRUNET_SMALL_V4(nn.Module):
    def __init__(self,
                in_channels: int = 3,
                out_channels: int = 3,
                n_features: int = 64,
                dropout: int = 0.1,
                block_out_channels=[64, 128, 128],
                layers_per_block=4,
                is_attn_layers=(False, False, False, False),
                upsample_type='pixelshuffle_1x1',
                downsample_type='conv_1x1',
                adaptive_weight: bool = True,
                fixed_weight_value: float = 1.0,
                bottleneck_attention: bool = False,
                local_conv: str = 'conv_1x1',
                img_range = 255
                ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_features = n_features
        self.dropout = dropout
        self.block_out_channels = block_out_channels
        self.layers_per_block = layers_per_block
        self.is_attn_layers = is_attn_layers
        self.local_conv = local_conv
        self.bottleneck_attention = bottleneck_attention

        self.upsample_type = upsample_type
        self.downsample_type = downsample_type
        self.adaptive_weight = adaptive_weight
        self.fixed_weight_value = fixed_weight_value

        self.img_range = img_range
        self.sub_mean = MeanShift(img_range)
        self.add_mean = MeanShift(img_range, sign=1)

        self.shallow_feature_extraction = nn.Conv2d(
            in_channels, n_features, kernel_size=3, padding=1)
        self.image_rescontruction = nn.Conv2d(
            n_features, in_channels, kernel_size=3, padding=1)

        self.left_model = self.left_unet()
        self.middle_model = MiddleBlock(block_out_channels[-1], dropout=self.dropout, num_layers=layers_per_block,
                                        bottleneck_attention=bottleneck_attention, local_conv=local_conv)
        self.right_model = self.right_unet()

    def left_unet(self):
        left_model = []

        in_channel = out_channel = self.n_features
        for i in range(len(self.block_out_channels)):
            out_channel = self.block_out_channels[i]

            down_block = DownBlock(in_channel, out_channel, dropout=self.dropout, has_attn=self.is_attn_layers[i],
                                    adaptive_weight=self.adaptive_weight, fixed_weight_value=self.fixed_weight_value, local_conv=self.local_conv,
                                    layers_per_block=self.layers_per_block)
            in_channel = out_channel
            left_model.append(down_block)
            if i < len(self.block_out_channels):
                left_model.append(Downsample(out_channel, downtype=self.downsample_type))

        return nn.ModuleList(left_model)

    def right_unet(self):
        right_unet = []

        in_channel = out_channel = self.block_out_channels[-1]
        for i in reversed(range(len(self.block_out_channels))):

            is_attn = self.bottleneck_attention if i == len(self.block_out_channels) - 1 \
                                                else self.is_attn_layers[i + 1]

            out_channel = self.block_out_channels[i]

            up_block = [UpBlock(in_channel, out_channel, dropout=self.dropout, has_attn=is_attn,
                                adaptive_weight=self.adaptive_weight, fixed_weight_value=self.fixed_weight_value, local_conv=self.local_conv,
                                layers_per_block=self.layers_per_block)]

            in_channel = out_channel * 2
            right_unet.append(nn.Sequential(*up_block))
            right_unet.append(Upsample(out_channel, uptype=self.upsample_type))

        in_channel, out_channel = self.block_out_channels[0] * \
            2, self.n_features
        up_block = [UpBlock(in_channel, out_channel, dropout=self.dropout, has_attn=self.is_attn_layers[0],
                            adaptive_weight=self.adaptive_weight, fixed_weight_value=self.fixed_weight_value, local_conv=self.local_conv,
                            layers_per_block=self.layers_per_block)]
        right_unet.append(up_block)
        return nn.ModuleList(right_unet)

    def forward(self, x):
        x = x * self.img_range
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
        recover = feature_maps + recover # global residual
        recover = self.add_mean(recover) / self.img_range
        return recover