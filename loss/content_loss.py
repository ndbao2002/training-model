from torch import nn
import torch.nn.functional as F

import torch
import torchvision

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.mse_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.mse_loss(gram_x, gram_y)
        return loss

def edge_detection(images: torch.Tensor):
    # Convert to grayscale if RGB
    def to_grayscale(batch):
        if batch.size(1) == 3:
            r, g, b = batch[:, 0:1], batch[:, 1:2], batch[:, 2:3]
            return 0.2989 * r + 0.5870 * g + 0.1140 * b
        return batch

    # Apply Sobel edge detection
    def sobel_edges(batch):
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32, device=batch.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]], dtype=torch.float32, device=batch.device).view(1, 1, 3, 3)
        edge_x = F.conv2d(batch, sobel_x, padding=1)
        edge_y = F.conv2d(batch, sobel_y, padding=1)
        return torch.sqrt(edge_x ** 2 + edge_y ** 2)

    images_edges = sobel_edges(to_grayscale(images))

    return images_edges

class ContentLoss(nn.Module):
    def __init__(self, types=['mse', 'mae', 'perceptual', 'mae_edge', 'distillation_mse', 'distillation_mae'], weights=[1, 1, 1, 1, 1, 1], teacher_model=None):
        super(ContentLoss, self).__init__()

        self.types = types
        self.weights = weights
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if teacher_model is not None:
            self.teacher_model = teacher_model.to(self.device)
        else:
            self.teacher_model = None

    def forward(self, hr_pred, hr, lr=None):
        loss = 0
        loss_type = {}

        for loss_type_name, weight in zip(self.types, self.weights):
            if loss_type_name == 'mse':
                l = nn.MSELoss()(hr_pred, hr)
                loss_type['mse'] = l.item()
            elif loss_type_name == 'mae':
                l = nn.L1Loss()(hr_pred, hr)
                loss_type['mae'] = l.item()
            elif loss_type_name == 'perceptual':
                l = VGGPerceptualLoss().to(self.device)(hr_pred, hr)
                loss_type['perceptual'] = l.item()
            elif loss_type_name == 'mae_edge':
                edge_gt = edge_detection(hr)
                edge_pred = edge_detection(edge_pred)
                l = nn.L1Loss()(edge_pred, edge_gt)
                loss_type['mae_edge'] = l.item()
            elif loss_type_name == 'distillation_mse':
                teacher_output = self.teacher_model(lr).detach()
                l = nn.MSELoss()(hr_pred, teacher_output)
                loss_type['distillation_mse'] = l.item()
            elif loss_type_name == 'distillation_mae':
                teacher_output = self.teacher_model(lr).detach()
                l = nn.L1Loss()(hr_pred, teacher_output)
                loss_type['distillation_mae'] = l.item()
            else:
                raise ValueError(f"Unknown loss type: {loss_type_name}")

            loss += weight * l

        return loss, loss_type
