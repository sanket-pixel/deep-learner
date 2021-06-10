import torchvision
import torch
import os
import cv2 as cv
import math
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import numpy as np
import utils

class RobotsHumansDataset(Dataset):

    def __init__(self, root_dir, label_path,  mode, transform = None):
        self.root_dir = root_dir
        self.images = utils.get_file_names_from_text(os.path.join(root_dir,"{}.txt".format(mode)))
        self.labels = utils.get_labels(os.path.join(self.root_dir, label_path))
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(cv.imread(os.path.join(self.root_dir, self.images[idx])))[:,:,[2,1,0]].permute(2,0,1)/255.0
        label = self.labels[self.images[idx].split("/")[0]]
        if self.transform:
            image = self.transform(image)
        return image, label


def mirror(image):
    return torch.flip(image, [1])

def flip(image):
    return torch.flip(image,[0,1])

def offline_augmentation(image_path_text, root_dir, operations ):
    image_paths = utils.get_file_names_from_text(os.path.join(root_dir,image_path_text))
    for image in image_paths:
        img = torch.from_numpy(cv.imread(os.path.join(root_dir,image)))[:,:,[2,1,0]]
        for op in operations:
            func = globals()[op]
            new_image = func(img)
            try:
                cv.imwrite(os.path.join(root_dir,"{0}_{1}{2}".format(image[:-4],op,image[-4:])),new_image.numpy()[:,:,[2,1,0]])
            except:
                print(image)

