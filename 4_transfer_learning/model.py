import torch
from torch import nn
import torchvision
from torchvision import models

class FineTune(nn.Module):

    def __init__(self, base_model):
        super(FineTune,self).__init__()
        self.model = getattr(models,base_model)(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 2)


    def forward(self, images):
        y = self.model(images)
        return y


class FixedFeatureMLP(nn.Module):

    def __init__(self, base_model):
        super(FixedFeatureMLP,self).__init__()
        self.model = getattr(models,base_model)(pretrained=True)
        self.model.fc = nn.Identity()
        in_features = self.model.fc.in_features
        self.mlp = nn.Linear(in_features, 2)


    def forward(self, images):
        with torch.no_grad():
            fixed_features = self.model(images)
        y = self.model(fixed_features)
        return y
