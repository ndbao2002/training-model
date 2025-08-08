import torch
import glob
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
from torchvision.transforms import ToTensor, Compose
from PIL import Image
import numpy as np
import os

from utils.color_util import rgb2ycbcr
from utils.matlab_functions import imresize


class TrainingDataset(Dataset):
    def __init__(self, root_paths, inp_size=(120, 120), repeat=1, scale=4):
        self.transform = transforms.Compose([
            transforms.RandomCrop(inp_size), #(h, w)
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
        ])

        self.to_tensor = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ])

        self.inp_size = inp_size
        self.repeat = repeat
        self.files = []
        self.scale = scale

        for root_path in root_paths.split('+'):
            self.files += glob.glob(root_path + "/*")

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        while True:
            img_hr = Image.open(self.files[idx % len(self.files)])
            width, height = img_hr.size
        
            if width >= self.inp_size[0] and height >= self.inp_size[1]:
                break
            else:
                idx = np.random.randint(len(self.files))
                
        
        img_hr = self.transform(img_hr)
        
        if self.scale:
            scale = self.scale
        else:
            scale = np.random.uniform(1, 4.5)

        h_hr, w_hr = self.inp_size
        img_lr = img_hr.resize((int(h_hr/scale), int(w_hr/scale)), resample=Image.BICUBIC)
        img_lr_bicubic = img_lr.resize((h_hr, w_hr), resample=Image.BICUBIC)

        img_hr = self.to_tensor(img_hr)
        img_lr = self.to_tensor(img_lr)
        img_lr_bicubic = self.to_tensor(img_lr_bicubic)

        return img_lr, img_lr_bicubic, img_hr

class TestingDataset(Dataset):
    def __init__(self, hr_root):
        self.hr_files = sorted(glob.glob(hr_root + "/*"))

        self.transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ])

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        filename =os.path.basename(self.hr_files[idx])
        img_hr = Image.open(self.hr_files[idx]).convert("RGB")

        h_hr, w_hr = img_hr.size
        h_hr, w_hr = h_hr//4*4, w_hr//4*4
        
        img_hr = Image.fromarray(np.array(img_hr)[:w_hr, :h_hr, ...]) 
        img_lr = img_hr.resize((h_hr//4, w_hr//4), resample=Image.BICUBIC)
        img_lr = img_lr.resize((h_hr, w_hr), resample=Image.BICUBIC)


        img_lr = np.array(img_lr)[:w_hr, :h_hr, ...]
        

        img_hr = self.transform(img_hr)
        img_lr = self.transform(img_lr)

        return filename, img_lr, img_hr

if __name__ == '__main__':
    from utils.tools import plot_images
    import matplotlib.pyplot as plt
    from tqdm import tqdm 
    from torch.utils.data import DataLoader

    # training_dataset = TrainingDataset(root_paths='/media02/btlen04/vntan/datasets/TrainingDataset/DIV2K/DIV2K_train_HR/DIV2K_train_HR+/media02/btlen04/vntan/datasets/TrainingDataset/FLICKR')
    # trainloader = DataLoader(training_dataset,
    #                      batch_size=32,
    #                      shuffle=False,
    #                      num_workers=4)

    # progress_bar = tqdm(trainloader, total=len(trainloader))

    # for _ in progress_bar:
    #     pass

    # print(lr.min(), lr.max())

    # for i in range(10):
    #     lr, hr = training_dataset[0]
    #     fig = plot_images([lr, hr], cmaps=['gray', 'gray'])
        # print(lr.shape, hr.shape)
        # plt.show()

    # testing_dataset = TestingDataset(hr_root='D:\SR Research\MyModel\dataset\Set14\Set14\original')

    # for i in range(len(testing_dataset)):
    #     filename, lr, hr = testing_dataset[i]
    #     fig = plot_images([lr, hr], cmaps=['gray', 'gray'])
    #     print(filename, lr.shape, hr.shape)
    #     plt.show()