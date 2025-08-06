import os
from time import time
from dataloader import TrainingDataset, TestingDataset
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml

from utils.tools import find_padding, remove_padding
from loss.content_loss import ContentLoss
from loss.gan_loss import GANLoss
from models import *
import torch
from tqdm import tqdm

from utils.tools import calc_psnr_and_ssim_torch_metric

# Add argument parser
import argparse
parser = argparse.ArgumentParser(description='Training script for super-resolution models')
parser.add_argument('--model', type=str, default='srunet', choices=['srunet', 'mambaunet'], help='Model to train')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

config = yaml.load(open('config/config.yml', 'r'), Loader=yaml.FullLoader)

print('Creating Dataloader ...')
# Training dataloder
training_dataset = TrainingDataset(root_paths=config['train_dataset']['root_paths'],
                                   inp_size=config['train_dataset']['inp_size'],
                                   repeat=config['train_dataset']['repeat'])

trainloader = DataLoader(training_dataset,
                         batch_size= config['training_batch_size'],
                         shuffle=True,
                         num_workers=4)


# Validation dataloder
validation_dataset = TestingDataset(hr_root=config['val_dataset'])
validloader = DataLoader(validation_dataset, batch_size=1)

print('Creating Model ...')
# Create the model
if args.model == 'srunet':
    model = SRUNET(in_channels=3,
            out_channels=3,
            n_features=64,
            dropout=0.2,
            block_out_channels=[64, 64, 128, 128, 256],
            layers_per_block=2)
elif args.model == 'mambaunet':
    model = MambaUnet(in_channels=3,
            out_channels=3,
            n_features=64,
            dropout=0.2,
            block_out_channels=[64, 64, 128],
            layers_per_block=2,
            num_middle_block=3)
else:
    raise ValueError(f"Model {config['train_model']['model']} is not supported")

# Logging
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(config['logger']['tensorboard'])

import logging

# Cấu hình logging cơ bản
logging.basicConfig(
    filename=os.path.join(config['logger']['logging'], args.model + '.log'),
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

activate_checkpoint = config['train_model']['activate']

checkpoint_path = config['train_model']['checkpoint']
if activate_checkpoint and checkpoint_path and os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    start_point = checkpoint['epoch'] + 1
    iteration = checkpoint['iteration'] + 1
    psnr_max = checkpoint['psnr_max']
    max_psnr_epoch = checkpoint['epoch']
    print(f'Load checkpoint from {checkpoint_path}: start_point = {start_point}, iteration = {iteration}')
else:
    print('Starting new model')

optimizer = torch.optim.AdamW(model.parameters(), lr=config['train_model']['learning_rate'])

content_loss = ContentLoss(types=['mae'], weights=[1])

print('Starting training ...')
# Training model
model = model.to(device)

for epoch in range(start_point, max_epoch):
    progress_bar = tqdm(trainloader, total=len(trainloader))
    progress_bar.set_description(f"Epoch {epoch}")

    writer.add_scalar('ATraining/LR', config['train_model']['learning_rate'], epoch)

    for img_lr, img_hr in progress_bar:

        img_lr, img_hr = img_lr.to(device), img_hr.to(device)
        img_pred = model(img_lr)

        # Train Generator
        model.train(True)
        img_pred = model(img_lr)
        loss_content, loss_content_info = content_loss(img_pred, img_hr)
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

    model.eval()
    with torch.no_grad():
        progress_bar = tqdm(validloader, total=len(validloader))
        progress_bar.set_description(f"Testing")

        psnr, ssim = 0, 0
        pics = []
        inference_times = []
        for filename, img_lr, img_hr in progress_bar:
            img_lr, img_hr = img_lr.to(device), img_hr.to(device)
            # Padding
            img_lr, mod_pad_h, mod_pad_w = find_padding(img_lr, window_size=2**4)
            start_infer = time()
            img_pred = model(img_lr).clip(0, 1)
            infer_time = time() - start_infer
            inference_times.append(infer_time)
            img_pred = remove_padding(img_pred, mod_pad_h, mod_pad_w)
            # print(filename, img_pred.shape, img_hr.shape, img_lr.shape)

            psnr_img, ssim_img = calc_psnr_and_ssim_torch_metric(img_hr, img_pred, shaved=4)

            psnr += psnr_img
            ssim += ssim_img

            img_hr = torch.squeeze(img_hr.detach().cpu(), 0)
            img_lr = torch.squeeze(img_lr.detach().cpu(), 0)
            img_pred = torch.squeeze(img_pred.detach().cpu(), 0)

            images = make_grid([img_lr, img_pred, img_hr], nrow=3)
            pics.append((filename[0], images))

        avg_infer_time = sum(inference_times) / len(inference_times) if inference_times else 0
        writer.add_scalar('Testing/Avg_Inference_Time', avg_infer_time, epoch)
        print(f"Average inference time per image: {avg_infer_time:.4f} seconds")

        if psnr/len(validloader) > psnr_max:
            if not os.path.exists(os.path.join(config['train_model']['save_path'], args.model)):
                os.makedirs(os.path.join(config['train_model']['save_path'], args.model))

            psnr_max = max(psnr/len(validloader), psnr_max)
            max_psnr_epoch = epoch
            torch.save({
                'epoch': epoch,
                'iteration': iteration - 1,
                'psnr_max': psnr_max,
                'model': model.state_dict()
            }, os.path.join(config['train_model']['save_path'], args.model, f'best_model.pt'))

            os.makedirs(os.path.join(config['logger']['results'], str(epoch)), exist_ok=True)
            for filename, tensor_image in pics:
                img_path = os.path.join(config['logger']['results'], str(epoch), filename)
                image = transforms.ToPILImage()(tensor_image)
                os.makedirs(os.path.dirname(img_path), exist_ok=True)
                image.save(img_path)

        torch.save({
            'epoch': epoch,
            'iteration': iteration - 1,
            'psnr_max': psnr_max,
            'model': model.state_dict()
        }, os.path.join(config['train_model']['save_path'], args.model, f'curr_model.pt'))

        
        writer.add_scalar('Testing/PSNR_max', psnr / len(validloader), epoch)
        writer.add_scalar('Testing/SSIM_max', ssim / len(validloader), epoch)

       

        writer.add_scalar('Testing/PSNR', psnr/len(validloader), epoch)
        writer.add_scalar('Testing/SSIM', ssim/len(validloader), epoch)

        print(f"Epoch: {epoch} | PSNR: {psnr / len(validloader)} | SSIM: {ssim / len(validloader)} | Max_Epoch: {max_psnr_epoch} | PSNR_max: {psnr_max}")
        logging.info(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')} | Epoch: {epoch} | PSNR: {psnr / len(validloader)} | SSIM: {ssim / len(validloader)} | Max_Epoch: {max_psnr_epoch} | PSNR_max: {psnr_max}")
        logging.info(f"Average inference time per image: {avg_infer_time:.4f} seconds")
