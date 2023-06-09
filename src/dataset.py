'''a module to import data as pytorch into the framework'''
import os
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import nibabel as nib
import numpy as np
import torchvision
import torchvision.transforms as TF


class CTDataset(Dataset):
    ''' a class to import CT scans into the framework'''

    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.maskes = os.listdir(mask_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.maskes[index])

        image = np.load(img_path)
        mask = np.load(mask_path)

        #TODO: add comment about transform and the reason of unsqueeze
        image = transforms.Compose([transforms.ToTensor()])(image)
        mask = transforms.Compose([transforms.ToTensor()])(mask)

        image = torch.unsqueeze(image, dim=1).float()
        mask = torch.unsqueeze(mask, dim=1).float()

        Resize = TF.Resize(size=(128, 128)) # do resize to fit data on GPU's RAM
        image = Resize(image)/255.0 #normalize value between 0 and 1
        mask = torch.round(Resize(mask)/255.0)
        return image, mask


if __name__ == "__main__":
    #TODO: add comment about specifications of test function and replace
    #test function to test folder
    my_tensor = torch.randn((34, 1, 512, 512))
    print("first" + str(my_tensor.size()))
    Resize = TF.Resize(size=(128, 128))
    print("second"+str(TF(my_tensor).size()))
