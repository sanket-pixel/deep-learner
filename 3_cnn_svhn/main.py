import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

train_dataset = datasets.SVHN(root='./data', split="train", transform=transforms.ToTensor(), download=True)

test_dataset = datasets.SVHN(root='./data', split="test", transform=transforms.ToTensor(),download=True)

B_SIZE = 256

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=B_SIZE,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=B_SIZE,
                                          shuffle=False)

class CNN(nn.Module):

    def __init__(self):
        """ Model Initializer """

        super().__init__()

        # layer 1
        conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5,stride=1, padding=0)
        relu1 = nn.ReLU()
        maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.layer1 = nn.Sequential(
            conv1, relu1, maxpool1
        )

        # layer 2
        conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        relu2 = nn.ReLU()
        maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.layer2 = nn.Sequential(
            conv2, relu2, maxpool2
        )

        x = torch.randn((1,3,32,32))
        out_conv = self.conv_before_linear(x)
        in_dim = int(torch.prod(torch.tensor(out_conv.shape)))
        self.fc = nn.Linear(in_features=in_dim, out_features=10)
        # fully connnected classifier
        return


    def conv_before_linear(self,x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        return out2

    def forward(self,x):
        batch_size = x.shape[0]
        out_flat = self.conv_before_linear(x).view(batch_size,-1)
        y = self.fc(out_flat)
        return y


@torch.no_grad()
def eval_model(model):
    """ Computing model accuracy """
    correct = 0
    total = 0
    loss_list = []

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass only to get logits/output
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Get predictions from the maximum value
        preds = torch.argmax(outputs, dim=1)
        correct += len(torch.where(preds == labels)[0])
        total += len(labels)

    # Total correct predictions and loss
    accuracy = correct / total * 100
    loss = np.mean(loss_list)
    return accuracy, loss


def save_model(model, optimizer, epoch, stats):
    """ Saving model checkpoint """

    if (not os.path.exists("models")):
        os.makedirs("models")
    savepath = f"models/checkpoint_epoch_{epoch}.pth"

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'stats': stats
    }, savepath)
    return


def load_model(model, optimizer, savepath):
    """ Loading pretrained checkpoint """

    checkpoint = torch.load(savepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint["epoch"]
    stats = checkpoint["stats"]

    return model, optimizer, epoch, stats

# loading model

LR = 3e-4
EPOCHS = 10
EVAL_FREQ = 1
SAVE_FREQ = 10

cnn = CNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn = cnn.to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(params=cnn.parameters(), lr=LR)

# savepath = os.path.join(os.getcwd(), "models", "checkpoint_epoch_70.pth")
# model, optimizer, init_epoch, stats = load_model(cnn, optimizer, savepath)

stats = {
    "epoch": [],
    "train_loss": [],
    "valid_loss": [],
    "accuracy": []
}
init_epoch = 0

loss_hist = []
for epoch in range(init_epoch, EPOCHS):
    loss_list = []
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (img,label) in progress_bar:
        img = img.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        pred = cnn(img)
        loss = criterion(pred,label)
        loss_list.append(loss.item())

        loss.backward()
        optimizer.step()
        progress_bar.set_description(f"Epoch {epoch + 1} Iter {i + 1}: loss {loss.item():.5f}. ")

    loss_hist.append(np.mean(loss_list))
    stats['epoch'].append(epoch)
    stats['train_loss'].append(loss_hist[-1])
    # evaluating model
    if epoch % EVAL_FREQ == 0:
        accuracy, valid_loss = eval_model(cnn)
        print(f"Accuracy at epoch {epoch}: {round(accuracy, 2)}%")
    else:
        accuracy, valid_loss = -1, -1
    stats["accuracy"].append(accuracy)
    stats["valid_loss"].append(valid_loss)

    # saving checkpoint
    if epoch % SAVE_FREQ == 0:
        save_model(model=cnn, optimizer=optimizer, epoch=epoch, stats=stats)
