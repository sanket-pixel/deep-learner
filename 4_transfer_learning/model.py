import torch
from torch import nn
import torchvision
from torchvision import models

class FineTune(nn.Module):

    def __init__(self, base_model):
        super(FineTune,self).__init__()
        self.model = getattr(models,base_model)(pretrained=True)
        if base_model == "vgg16":
            in_features = self.model.classifier._modules['6'].in_features
            self.model.classifier._modules['6'] = nn.Linear(in_features, 2)
        elif base_model == "resnet18":
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, 2)
        elif base_model == "densenet121":
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features,2)



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
