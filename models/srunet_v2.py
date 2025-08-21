import torch
import torch.nn as nn

class SelfEnhancementBlock(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // reduction, 1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // reduction, in_ch, 1, stride=1, padding=0),
        )
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_ch // 2, in_ch // 2, 1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape

        msws = self.se(x)
        msws = msws.view(B, C, H * W)

        # Split the tensor into two halves along the channel dimension
        a, b = x.split(C // 2, dim=1)
        a = self.shared_conv(a)
        b = self.shared_conv(b)

        a = a.permute(0, 2, 3, 1).view(B, H * W, C // 2)
        b = b.view(B, C // 2, H * W)

        result = torch.bmm(msws, a)
        result = torch.bmm(result, b)
        result = result.view(B, C, H, W)

        return result


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

        self.attn = SelfEnhancementBlock(out_channels) if has_attn else nn.Identity()

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


class SRUNET_V2(nn.Module):
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