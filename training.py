import os
import time
from dataloader import TrainingDataset, TestingDataset
from models.lora_utils import freeze_model, freeze_model_except_lora, replace_resblocks_with_lora
from models.quantization import dynamic_quantization, static_quantization
from models.srunet_small import SRUNET_SMALL
from models.srunet_small_v2 import SRUNET_SMALL_V2
from models.srunet_small_v3 import SRUNET_SMALL_V3
from models.srunet_small_v4 import SRUNET_SMALL_V4
from models.srunet_v2 import SRUNET_V2
from models.swinir import SwinIR
from models.srunet import SRUNET
from models.mambaunet import MAMBAUNET
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml

from utils.tools import find_padding, forward_chop, remove_padding
from loss.content_loss import ContentLoss
# from loss.gan_loss import GANLoss
import torch
from tqdm import tqdm

from utils.tools import calc_psnr_and_ssim_torch_metric

# Add argument parser
import argparse
parser = argparse.ArgumentParser(description='Training script for super-resolution models')

# Experiment settings
parser.add_argument('--exp_name', type=str, default='srunet', help='Experiment name')
parser.add_argument('--model', type=str, default='srunet', help='Model to train', required=True)
parser.add_argument('--loss', type=str, nargs='+', default=['mae'], help='Loss function to use', required=True)
parser.add_argument('--loss_weight', type=float, nargs='+', default=[1.0], help='Loss weights for each loss function', required=True)
parser.add_argument('--scale', type=int, default=4, help='Scale factor for super-resolution')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')

# Checkpoint - Model specific settings
parser.add_argument('--checkpoint', type=str, default='', help='Path to model checkpoint for resuming training')
parser.add_argument('--uptype', type=str, default='pixelshuffle_1x1', help='Upsampling type')
parser.add_argument('--downtype', type=str, default='conv_1x1', help='Downsampling type')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
parser.add_argument('--n_features', type=int, default=64, help='Number of features in the model')
parser.add_argument('--channel_per_level', type=int, nargs='+', default=[64, 128, 128], help='Number of channels per level in the model')
parser.add_argument('--attention_per_level', type=int, nargs='+', default=[0, 0, 0], help='Use attention for each level in the model')
parser.add_argument('--num_layers_per_block', type=int, default=4, help='Number of layers per block in the model')
parser.add_argument('--adaptive_weight', action='store_true', help='Use adaptive weight for residual blocks')
parser.add_argument('--fixed_weight_value', type=float, default=1.0, help='Fixed weight value for residual blocks')
parser.add_argument('--bottleneck_attention', action='store_true', help='Use bottleneck attention for residual blocks')
parser.add_argument('--local_conv', type=str, default='conv_1x1', help='Local convolution type')
parser.add_argument('--img_range', type=float, default=255, help='Image range for normalization')

# Additional settings
parser.add_argument('--chop', action='store_true', help='Enable memory-efficient forward')
parser.add_argument('--lora', action='store_true', help='Use LoRA for training')
parser.add_argument('--lora_rank', type=int, default=4, help='Rank for LoRA layers')
parser.add_argument('--lora_alpha', type=float, default=1.0, help='Alpha for LoRA layers')
parser.add_argument('--quantization', type=str, default='', help='Quantization method to use')
parser.add_argument('--original', type=str, default='', help='Path to original model checkpoint (if using LoRA)')
parser.add_argument('--distil_path', type=str, default='', help='Path to teacher model checkpoint (if using distillation)')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

config = yaml.load(open('config/config.yml', 'r'), Loader=yaml.FullLoader)

print('Creating Dataloader ...')
# Training dataloader
training_dataset = TrainingDataset(root_paths=config['train_dataset']['root_paths'],
                                   inp_size=config['train_dataset']['inp_size'],
                                   repeat=config['train_dataset']['repeat'],
                                   scale=args.scale)

trainloader = DataLoader(training_dataset,
                         batch_size= args.batch_size,
                         shuffle=True,
                         num_workers=2)


# Validation dataloader
validation_dataset = TestingDataset(hr_root=config['val_dataset'], scale=args.scale)
validloader = DataLoader(validation_dataset, batch_size=1)

print('Creating Model ...')
# Create the model
if args.model == 'srunet':
    model = SRUNET(in_channels=3,
            out_channels=3,
            n_features=args.n_features,
            dropout=0.1,
            block_out_channels=args.channel_per_level,
            layers_per_block=args.num_layers_per_block,
            is_attn_layers=args.attention_per_level)
elif args.model == 'srunet_small_v2':
        model = SRUNET_SMALL_V2(in_channels=3,
            out_channels=3,
            n_features=args.n_features,
            dropout=args.dropout,
            block_out_channels=args.channel_per_level,
            layers_per_block=args.num_layers_per_block,
            is_attn_layers=args.attention_per_level,
            upsample_type=args.uptype,
            downsample_type=args.downtype,
            adaptive_weight=args.adaptive_weight,
            fixed_weight_value=args.fixed_weight_value,
            bottleneck_attention=args.bottleneck_attention,
            local_conv=args.local_conv
            )
elif args.model == 'srunet_small_v3':
        model = SRUNET_SMALL_V3(in_channels=3,
            out_channels=3,
            n_features=args.n_features,
            dropout=args.dropout,
            block_out_channels=args.channel_per_level,
            layers_per_block=args.num_layers_per_block,
            is_attn_layers=args.attention_per_level,
            upsample_type=args.uptype,
            downsample_type=args.downtype,
            adaptive_weight=args.adaptive_weight,
            fixed_weight_value=args.fixed_weight_value,
            bottleneck_attention=args.bottleneck_attention,
            local_conv=args.local_conv,
            img_range=args.img_range
            )
elif args.model == 'srunet_small_v4':
        model = SRUNET_SMALL_V4(in_channels=3,
            out_channels=3,
            n_features=args.n_features,
            dropout=args.dropout,
            block_out_channels=args.channel_per_level,
            layers_per_block=args.num_layers_per_block,
            is_attn_layers=args.attention_per_level,
            upsample_type=args.uptype,
            downsample_type=args.downtype,
            adaptive_weight=args.adaptive_weight,
            fixed_weight_value=args.fixed_weight_value,
            bottleneck_attention=args.bottleneck_attention,
            local_conv=args.local_conv,
            img_range=args.img_range
            )
else:
    raise ValueError(f"Model {args.model} is not supported")

# Logging
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(config['logger']['tensorboard'])

import logging

# Cấu hình logging cơ bản
if not os.path.exists(os.path.join(config['logger']['logging'], args.exp_name)):
    os.makedirs(os.path.join(config['logger']['logging'], args.exp_name), exist_ok=True)

# Save args
with open(os.path.join(config['logger']['logging'], args.exp_name, 'parser.yml'), 'w') as f:
    yaml.dump(vars(args), f)

logging.basicConfig(
    filename=os.path.join(config['logger']['logging'], args.exp_name, args.exp_name + '.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)

print('Loading model ...')
# Setup training
start_point = 0
iteration = 0
max_epoch = config['train_model']['max_epoch']
psnr_max = 0


checkpoint_path = args.checkpoint
if checkpoint_path and os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    start_point = checkpoint['epoch'] + 1
    iteration = checkpoint['iteration'] + 1
    psnr_max = checkpoint['psnr_max']
    max_psnr_epoch = checkpoint['epoch']
    print(f'Load checkpoint from {checkpoint_path}: start_point = {start_point}, iteration = {iteration}')
elif args.lora:
    checkpoint = torch.load(args.original, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['best_model_state_dict'])

    replace_resblocks_with_lora(model, lora_r=args.lora_rank, lora_alpha=args.lora_alpha)
    freeze_model_except_lora(model)
    print(f'Inject LoRA into model: lora_rank = {args.lora_rank}, lora_alpha = {args.lora_alpha}')
else:
    print('Starting new model')
model = model.to(device)

if args.quantization == 'aware':
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    model = torch.quantization.prepare_qat(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=config['train_model']['learning_rate'])

teacher_model = None
if 'distil' in ''.join(args.loss):
    teacher_model = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
    teacher_model.load_state_dict(torch.load(args.distil_path, map_location=torch.device('cpu'))['params'], strict=True)
    freeze_model(teacher_model)
content_loss = ContentLoss(types=args.loss, weights=args.loss_weight, teacher_model=teacher_model)

# Add lr_scheduler
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[250, 400, 450, 475, 500], gamma=0.5)
for _ in range(start_point):
    lr_scheduler.step()

print('Starting training ...')
# Training model
for epoch in range(start_point, max_epoch):
    progress_bar = tqdm(trainloader, total=len(trainloader), miniters=10)
    progress_bar.set_description(f"Epoch {epoch}")
    model.train(True)

    writer.add_scalar('ATraining/LR', config['train_model']['learning_rate'], epoch)

    for img_lr, img_lr_bicubic, img_hr in progress_bar:

        img_lr, img_lr_bicubic, img_hr = img_lr.to(device), img_lr_bicubic.to(device), img_hr.to(device)
        img_pred = model(img_lr_bicubic)

        # Train Generator
        img_pred = model(img_lr_bicubic)
        loss_content, loss_content_info = content_loss(img_pred, img_hr, img_lr)
        # loss_gan = gan_loss(img_pred)

        loss = loss_content #+ 10**-3*loss_gan
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        logs = {**loss_content_info}
        progress_bar.set_postfix(**logs)

        for keys in logs.keys():
            writer.add_scalar(f'ATraining/{keys}', logs[keys], iteration)

        iteration += 1
        # break
    lr_scheduler.step()

    model.eval()
    with torch.no_grad():
        progress_bar = tqdm(validloader, total=len(validloader))
        progress_bar.set_description(f"Testing")

        psnr, ssim = 0, 0
        pics = []
        for filename, img_lr, img_hr in progress_bar:
            img_lr, img_hr = img_lr.to(device), img_hr.to(device)
            # Padding
            img_lr, mod_pad_h, mod_pad_w = find_padding(img_lr, window_size=2**4)
            if args.chop:
                img_pred = forward_chop(model, img_lr).clip(0, 1)
            else:
                img_pred = model(img_lr).clip(0, 1)
            img_pred = remove_padding(img_pred, mod_pad_h, mod_pad_w)
            img_lr = remove_padding(img_lr, mod_pad_h, mod_pad_w)
            # print(filename, img_pred.shape, img_hr.shape, img_lr.shape)

            psnr_img, ssim_img = calc_psnr_and_ssim_torch_metric(img_hr, img_pred, shaved=4)

            psnr += psnr_img
            ssim += ssim_img

            img_hr = torch.squeeze(img_hr.detach().cpu(), 0)
            img_lr = torch.squeeze(img_lr.detach().cpu(), 0)
            img_pred = torch.squeeze(img_pred.detach().cpu(), 0)

            images = make_grid([img_lr, img_pred, img_hr], nrow=3)
            pics.append((filename[0], images))

        if psnr/len(validloader) > psnr_max:
            if not os.path.exists(os.path.join(config['train_model']['save_path'], args.exp_name)):
                os.makedirs(os.path.join(config['train_model']['save_path'], args.exp_name))

            psnr_max = max(psnr/len(validloader), psnr_max)
            max_psnr_epoch = epoch
            torch.save({
                'epoch': epoch,
                'iteration': iteration - 1,
                'psnr_max': psnr_max,
                'model': model.state_dict()
            }, os.path.join(config['train_model']['save_path'], args.exp_name, f'best_model.pt'))

            os.makedirs(os.path.join(config['logger']['logging'], args.exp_name, str(epoch)), exist_ok=True)
            for filename, tensor_image in pics:
                img_path = os.path.join(config['logger']['logging'], args.exp_name, str(epoch), filename)
                image = transforms.ToPILImage()(tensor_image)
                os.makedirs(os.path.dirname(img_path), exist_ok=True)
                image.save(img_path)

        torch.save({
            'epoch': epoch,
            'iteration': iteration - 1,
            'psnr_max': psnr_max,
            'model': model.state_dict()
        }, os.path.join(config['train_model']['save_path'], args.exp_name, f'curr_model.pt'))

        
        writer.add_scalar('Testing/PSNR_max', psnr / len(validloader), epoch)
        writer.add_scalar('Testing/SSIM_max', ssim / len(validloader), epoch)

       

        writer.add_scalar('Testing/PSNR', psnr/len(validloader), epoch)
        writer.add_scalar('Testing/SSIM', ssim/len(validloader), epoch)

        print(f"Epoch: {epoch} | PSNR: {psnr / len(validloader)} | SSIM: {ssim / len(validloader)} | Max_Epoch: {max_psnr_epoch} | PSNR_max: {psnr_max}")
        logging.info(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')} | Epoch: {epoch} | PSNR: {psnr / len(validloader)} | SSIM: {ssim / len(validloader)} | Max_Epoch: {max_psnr_epoch} | PSNR_max: {psnr_max}")
