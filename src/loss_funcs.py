# TODO: add documantations and comments

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth) / \
            ((inputs+ targets).sum() + smooth)

        return 1 - dice


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth)/(union + smooth)

        return 1 - IoU


class Combined_Loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Combined_Loss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.BCE_loss_func = nn.BCEWithLogitsLoss()
        self.BCE_loss = nn.BCELoss()
        self.dice_loss_func = DiceLoss()
        self.IoU_loss_func =  IoULoss()
        self.mse_loss = nn.MSELoss()
    def forward(self, inputs, targets):
        
        mse_loss = self.mse_loss(inputs, targets)
        
        # BCE_loss = self.BCE_loss_func(inputs, targets)
        BCE_loss = self.BCE_loss(inputs, targets)

        Dice_Loss = self.dice_loss_func(inputs, targets)

        IoU_Loss = self.IoU_loss_func(inputs, targets)
        
        # loss = [Dice_Loss , 0.1*BCE_loss]
        # loss = Dice_Loss #+ 100*BCE_loss + IoU_Loss
        loss = 10*Dice_Loss + mse_loss
        

        return loss
