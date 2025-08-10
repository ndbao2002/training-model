import torch

def dynamic_quantization(model, layer=torch.nn.Linear):
    """
    Applies dynamic quantization to the specified layer of the model.
    
    Args:
        model (torch.nn.Module): The model to be quantized.
        layer (torch.nn.Module): The layer type to be quantized, default is torch.nn.Linear.
        
    Returns:
        torch.nn.Module: The quantized model.
    """
    model_dynamic = torch.quantization.quantize_dynamic(
        model,
        {layer},  # Specify layers to quantize
        dtype=torch.qint8
    )

    return model_dynamic

def static_quantization(model, calibration_data: torch.utils.data.DataLoader):
    """
    Applies static quantization to the specified layer of the model.
    
    Args:
        model (torch.nn.Module): The model to be quantized.
        calibration_data (torch.utils.data.DataLoader): The data loader for calibration data.
        
    Returns:
        torch.nn.Module: The quantized model.
    """
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    torch.quantization.prepare(model, inplace=True)

    with torch.no_grad():
        for _, img_lr_bicubic, _ in calibration_data:
            model(img_lr_bicubic)

    torch.quantization.convert(model, inplace=True)
    return model

