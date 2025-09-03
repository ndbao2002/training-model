import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

def find_padding(x, window_size):
    _, _, h, w = x.size()
    mod_pad_h = (window_size - h % window_size) % window_size
    mod_pad_w = (window_size - w % window_size) % window_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x, mod_pad_h, mod_pad_w

def remove_padding(x_padded, mod_pad_h, mod_pad_w):
    if mod_pad_h > 0:
        x_padded = x_padded[:, :, :-mod_pad_h, :]
    if mod_pad_w > 0:
        x_padded = x_padded[:, :, :, :-mod_pad_w]
    return x_padded

def calc_psnr_and_ssim_torch_metric(hr, sr, shaved=4):
    sr = sr * 255
    hr = hr * 255
    # if (sr.size() != hr.size()):
    #     h_min = min(sr.size(2), hr.size(2))
    #     w_min = min(sr.size(3), hr.size(3))
    #     sr = sr[:, :, :h_min, :w_min]
    #     hr = hr[:, :, :h_min, :w_min]

    sr = sr.round().cpu()
    hr = hr.round().cpu()

    sr[:, 0, :, :] = sr[:, 0, :, :] * 65.738/256.0
    sr[:, 1, :, :] = sr[:, 1, :, :] * 129.057/256.0
    sr[:, 2, :, :] = sr[:, 2, :, :] * 25.064/256.0
    sr = sr.sum(dim=1, keepdim=True) + 16.0
    
    hr[:, 0, :, :] = hr[:, 0, :, :] * 65.738/256.0
    hr[:, 1, :, :] = hr[:, 1, :, :] * 129.057/256.0
    hr[:, 2, :, :] = hr[:, 2, :, :] * 25.064/256.0
    hr = hr.sum(dim=1, keepdim=True) + 16.0

    if shaved:
        sr = sr[:, :, shaved:-shaved, shaved:-shaved]
        hr = hr[:, :, shaved:-shaved, shaved:-shaved]

    ssim_cal = StructuralSimilarityIndexMeasure(data_range=255)
    psnr_cal = PeakSignalNoiseRatio(data_range=255)

    return psnr_cal(sr, hr), ssim_cal(sr, hr)




def plot_images(images, nrows=1, ncols=None, titles=None, cmaps=None, figsize=(12, 6)):
    """
    Hiển thị một danh sách ảnh dưới dạng grid sử dụng matplotlib.

    Parameters:
    - images: list các ảnh (kiểu PIL.Image, numpy array (cv2), hoặc torch.Tensor)
    - nrows: số dòng trong grid
    - ncols: số cột trong grid (nếu None, tự tính)
    - titles: list tiêu đề cho từng ảnh
    - cmaps: list cmap (color map) cho từng ảnh hoặc None
    - figsize: kích thước toàn bộ figure

    Returns:
    - figure matplotlib
    """

    def to_numpy(img):
        # Chuyển ảnh sang numpy array RGB hoặc grayscale
        if isinstance(img, Image.Image):
            return np.array(img)
        elif isinstance(img, torch.Tensor):
            img = img.detach().cpu()
            if img.ndim == 3 and img.shape[0] in (1, 3):  # CHW -> HWC
                img = img.permute(1, 2, 0)
            img = img.numpy()
            if img.max() <= 1:
                img = img * 255
            return img.astype(np.uint8)
        elif isinstance(img, np.ndarray):
            return img
        else:
            raise TypeError(f"Không hỗ trợ kiểu ảnh: {type(img)}")

    total = len(images)
    if ncols is None:
        ncols = int(np.ceil(total / nrows))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for i in range(len(axes)):
        axes[i].axis("off")
        if i < total:
            img_np = to_numpy(images[i])
            cmap = None
            if cmaps and i < len(cmaps):
                cmap = cmaps[i]
            axes[i].imshow(img_np, cmap=cmap)
            if titles and i < len(titles):
                axes[i].set_title(titles[i])

    plt.tight_layout()
    return fig

def forward_chop(model, x, scale, shave=10, min_size=160000) -> torch.Tensor:
        n_GPUs = 1
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]

        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                sr_batch = model(lr_batch)
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [
                forward_chop(model, patch, scale, shave=shave, min_size=min_size) \
                for patch in lr_list
            ]

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        output = x.new(b, c, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output
