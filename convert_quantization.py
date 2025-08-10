import os
import torch
import yaml
from dataloader import TrainingDataset
from models.srunet import SRUNET
from models.mambaunet import MAMBAUNET
from torch.utils.data import DataLoader
from models.quantization import dynamic_quantization, static_quantization

import argparse
parser = argparse.ArgumentParser(description='Convert model to quantized version')
parser.add_argument('--model', type=str, default='srunet', choices=['srunet', 'mambaunet'], help='Model to convert', required=True)
parser.add_argument('--quantization', type=str, default='dynamic', choices=['dynamic', 'static', 'aware', 'half'], help='Type of quantization to apply', required=True)
parser.add_argument('--checkpoint', type=str, default='', help='Path to model checkpoint', required=True)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = yaml.load(open('config/config.yml', 'r'), Loader=yaml.FullLoader)

print('Creating Dataloader ...')
# Training dataloader
training_dataset = TrainingDataset(root_paths=config['train_dataset']['root_paths'],
                                   inp_size=config['train_dataset']['inp_size'],
                                   repeat=config['train_dataset']['repeat'])

trainloader = DataLoader(training_dataset,
                         batch_size= config['training_batch_size'],
                         shuffle=True,
                         num_workers=2)

if args.model == 'srunet':
    model = SRUNET(in_channels=3, out_channels=3, n_features=64, dropout=0.1, block_out_channels=[64, 128, 128, 256], layers_per_block=4, is_attn_layers=(False, False, True, False))
elif args.model == 'mambaunet':
    model = MAMBAUNET(in_channels=3, out_channels=3, n_features=64, dropout=0.1, block_out_channels=[64, 128, 128, 256], layers_per_block=4, is_attn_layers=(False, False, True, True))
else:
    raise ValueError(f"Model {args.model} is not supported")

model = model.to(device)

model.load_state_dict(torch.load(args.checkpoint, map_location=device)['model'])

if args.quantization == 'dynamic':
    model = dynamic_quantization(model)
elif args.quantization == 'static':
    model = static_quantization(model, trainloader)
elif args.quantization == 'aware':
    pass  # Quantization aware training is already applied during training
elif args.quantization == 'half':
    model = model.half()
else:
    raise ValueError(f"Quantization {args.quantization} is not supported")

torch.save({'model': model.state_dict()}, 
           os.path.join(config['train_model']['save_path'], args.model, f'quantized_{args.quantization}_model.pt'))