from torch import nn
from typing import Callable
from .srunet import LoRAConv2d, ResidualBlock

def replace_resblocks_with_lora(module: nn.Module,
                                lora_r: int = 4,
                                lora_alpha: float = 1.0,
                                dropout: float = 0.1,
                                n_groups: int = 32) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, ResidualBlock):
            # Get original block info
            in_channels = child.norm1.num_channels
            out_channels = child.norm2.num_channels
            has_attn = not isinstance(child.attn, nn.Identity)

            # Create new LoRA-enabled ResidualBlock
            new_block = ResidualBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                dropout=dropout,
                n_groups=n_groups,
                has_attn=has_attn,
                use_lora=True,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
            )

            # Copy weights
            new_block.norm1.load_state_dict(child.norm1.state_dict())
            new_block.norm2.load_state_dict(child.norm2.state_dict())

            # Copy Conv2d weights into base layers of LoRAConv2d
            new_block.conv1.base_layer.load_state_dict(child.conv1.state_dict())
            new_block.conv2.base_layer.load_state_dict(child.conv2.state_dict())

            # Copy shortcut if it's a Conv2d
            if isinstance(child.shortcut, nn.Conv2d) and isinstance(new_block.shortcut, nn.Conv2d):
                new_block.shortcut.load_state_dict(child.shortcut.state_dict())

            # Copy attention if used
            if has_attn:
                new_block.attn.load_state_dict(child.attn.state_dict())

            # Replace
            setattr(module, name, new_block)

        else:
            # Recurse into children
            replace_resblocks_with_lora(child, lora_r, lora_alpha, dropout, n_groups)

    return module

def freeze_model_except_lora(model):
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, LoRAConv2d):
            m.lora_A.weight.requires_grad = True
            m.lora_B.weight.requires_grad = True

def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False