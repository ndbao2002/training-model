import os

from torch import nn
import torch

from model.discriminator_arch import UNetDiscriminatorSN



class GANLoss(nn.Module):
    def __init__(self, lr=1e-4, load_path=None, save_path=None):
        super(GANLoss, self).__init__()

        self.lr = lr
        self.model =  UNetDiscriminatorSN(num_in_ch=1).to('cuda' if torch.cuda.is_available() else 'cpu')
        self.bce_loss = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if load_path is not None and os.path.exists(save_path):
            self.model.load_state_dict(torch.load(load_path))
        self.save_path = save_path

    def train(self, y_pred, y_hr):
        self.model.train()
        fake_preds = self.model(y_pred)
        real_preds = self.model(y_hr)

        real_loss = self.bce_loss(real_preds, torch.ones_like(real_preds))
        fake_loss = self.bce_loss(fake_preds, torch.zeros_like(fake_preds))

        self.optimizer.zero_grad()
        discriminator_loss = real_loss + fake_loss
        discriminator_loss.backward()
        self.optimizer.step()

        if self.save_path is not None:
            torch.save(self.model.state_dict(), self.save_path)

        return {'discriminator_loss': discriminator_loss.item()}

    def forward(self, y_pred):
        self.model.eval()
        real_preds = self.model(y_pred)
        gan_loss = self.bce_loss(real_preds, torch.ones_like(real_preds))
        return gan_loss

if __name__ == '__main__':

    loss = GANLoss()

    for i in range(100):
        y_lr, y_hr = torch.randn(1, 1, 256, 256).to(device='cuda'), torch.randn(1, 1, 256, 256).to(device='cuda')
        loss.train(y_lr, y_hr)
        print(loss(y_lr)[1]['gan_loss'])
    # loss = GANLoss().to('cuda' if torch.cuda.is_available() else 'cpu')

    # model = UNetDiscriminatorSN(num_in_ch=1).to(device='cuda')
    # bce_loss = nn.BCELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=10**-4)
    #
    # for i in range(100):
    #     y_pred = torch.randn(1, 1, 256, 256).to('cuda')
    #     y_hr = torch.randn(1, 1, 256, 256).to('cuda')
    #
    #     fake_preds = model(y_pred)
    #     real_preds = model(y_hr)
    #     real_loss = bce_loss(real_preds, torch.ones_like(real_preds))
    #     fake_loss = bce_loss(fake_preds, torch.zeros_like(fake_preds))
    #
    #     optimizer.zero_grad()
    #     loss = real_loss + fake_loss
    #     loss.backward()
    #     optimizer.step()

        # print(loss.item())
