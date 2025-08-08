from torch import nn
from typing import Callable
from srunet import ResidualBlock, ResidualBlockLora

def inject_lora_residual_block(model, lora_rank=4, lora_alpha=1.0):
    for name, module in model.named_children():
        if isinstance(module, ResidualBlock) and not isinstance(module, ResidualBlockLora):
            # Create replacement with same config
            lora_block = ResidualBlockLora(
                in_channels=module.model[0].num_channels,  # GroupNorm → num_channels
                out_channels=module.model[-1].out_channels,
                n_groups=module.model[0].num_groups,
                dropout=module.model[5].p,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha
            )
            # Copy weights
            lora_block.model.load_state_dict(module.model.state_dict())
            if isinstance(module.shortcut, nn.Conv2d):
                lora_block.shortcut.load_state_dict(module.shortcut.state_dict())
            setattr(model, name, lora_block)
        else:
            inject_lora_residual_block(module, lora_rank, lora_alpha)  # Recurse into submodules

def freeze_model_except_lora(model):
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name

def freeze_model(model):
    for name, param in model.named_parameters():
        param.requires_grad = False