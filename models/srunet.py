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
        # print(x.shape)
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
                 n_groups: int = 32,
                 dropout: float = 0.1):
        super().__init__()

        self.model = nn.Sequential(
            nn.GroupNorm(n_groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(n_groups, in_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()


    def forward(self, x: torch.Tensor):
        return self.shortcut(x) + self.model(x)
    
class ResidualBlockLora(ResidualBlock):
    def __init__(self, in_channels, out_channels, n_groups=32, dropout=0.1, lora_rank=4, lora_alpha=1.0):
        super().__init__(in_channels, out_channels, n_groups, dropout)
        self.lora_down = nn.Conv2d(in_channels, lora_rank, kernel_size=1, bias=False)
        self.lora_up = nn.Conv2d(lora_rank, out_channels, kernel_size=1, bias=False)
        self.scaling = lora_alpha / lora_rank
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return super().forward(x) + self.scaling * self.lora_up(self.lora_down(x))

class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_blocks: int, dropout: int):
        super().__init__()
        self.model = nn.Sequential(ResidualBlock(in_channels, out_channels, dropout=dropout),
                                    *[ResidualBlock(out_channels, out_channels, dropout=dropout) for _ in range(n_blocks - 1)])

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor):
        return self.model(x)

class BasicBlockWithAttention(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_blocks: int, dropout: int):
        super().__init__()
        self.model = [ResidualBlock(in_channels, out_channels, dropout=dropout),
                      AttentionBlock(out_channels)]

        for _ in range(n_blocks - 1):
            self.model.append(ResidualBlock(out_channels, out_channels, dropout=dropout))
            self.model.append(AttentionBlock(out_channels))

        self.model = nn.Sequential(*self.model)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor):
        return self.model(x)


class MiddleBlock(nn.Module):
    def __init__(self, n_channels: int, dropout: int):
        super().__init__()

        self.model = nn.Sequential(
            ResidualBlock(n_channels, n_channels, dropout=dropout),
            AttentionBlock(n_channels),
            ResidualBlock(n_channels, n_channels, dropout=dropout),
            AttentionBlock(n_channels),
            ResidualBlock(n_channels, n_channels, dropout=dropout),
            AttentionBlock(n_channels)
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)


class Downsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.convT = nn.ConvTranspose2d(n_channels, n_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor):
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

class SRUNET(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 n_features: int = 64,
                 dropout: int = 0.2,
                 block_out_channels=[64, 64, 128, 128, 256],
                 layers_per_block = 2
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_features = n_features
        self.dropout = dropout
        self.block_out_channels = [n_features] + block_out_channels
        self.layers_per_block = layers_per_block

        self.sub_mean = MeanShift(255)
        self.add_mean = MeanShift(255, sign=1)

        self.shallow_feature_extraction = nn.Conv2d(
            in_channels, n_features, kernel_size=3, padding=1)
        self.image_rescontruction = nn.Conv2d(
            n_features, in_channels, kernel_size=3, padding=1)

        self.encoder_model = self.encoder()
        self.middle_model = MiddleBlock(block_out_channels[-1], dropout=self.dropout)
        self.decoder_model = self.decoder()



    def encoder(self):
        encoder_block = []

        n_floor = len(self.block_out_channels) - 1
        for i in range(1, n_floor + 1):
            if i == n_floor - 1:
                encoder_block.append(BasicBlockWithAttention(in_channels= self.block_out_channels[i - 1],
                                                out_channels= self.block_out_channels[i],
                                                n_blocks= self.layers_per_block,
                                                dropout= self.dropout
                                                ))
            else:
                encoder_block.append(BasicBlock(in_channels= self.block_out_channels[i - 1],
                                out_channels= self.block_out_channels[i],
                                n_blocks= self.layers_per_block,
                                dropout= self.dropout
                                ))

            if i < n_floor: encoder_block.append(Downsample(self.block_out_channels[i]))

        return nn.ModuleList(encoder_block)

    def decoder(self):
        decoder_block = []

        n_floor = len(self.block_out_channels) - 1
        for i in range(n_floor, 0, -1):
            if i < n_floor:
                decoder_block.append(Upsample(self.block_out_channels[i]))

            if i == n_floor - 1:
                decoder_block.append(BasicBlockWithAttention(in_channels= self.block_out_channels[i] * 2,
                                                out_channels= self.block_out_channels[i - 1],
                                                n_blocks= self.layers_per_block,
                                                dropout= self.dropout
                                                ))
            else:
                if i == n_floor: in_channel = self.block_out_channels[i]
                else: in_channel = self.block_out_channels[i] * 2
                decoder_block.append(BasicBlock(in_channels= in_channel,
                                out_channels= self.block_out_channels[i - 1],
                                n_blocks= self.layers_per_block,
                                dropout= self.dropout
                                ))


        return nn.ModuleList(decoder_block)

    def forward(self, x: torch.Tensor):
        # x = self.sub_mean(x)
        x = self.shallow_feature_extraction(x)

        encoder_output = []
        for block in self.encoder_model:
            # print(type(block))
            if isinstance(block, Downsample): encoder_output.append(x)
            x = block(x)
            # print(x.shape)

        x = self.middle_model(x)

        encoder_output.reverse()
        for block in self.decoder_model:
            # print(type(block))
            if isinstance(block, Upsample):
                x = block(x)
                x = torch.cat([x, encoder_output[0]], 1)
                encoder_output.pop(0)
            else:
                x = block(x)

        x = self.image_rescontruction(x)
        # x = self.add_mean(x) / 255
        return x