import os
import torch
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
from torchvision import transforms
import numpy as np

from dataset import RobotsHumansDataset, offline_augmentation, flip, mirror
from model import FineTune
import utils
from matplotlib import pyplot as plt


data_root = "data"
model_root = "model"
classes_path = "classes.txt"
# generate train, test, classes text files
if not os.path.exists(os.path.join(data_root,classes_path)):
    utils.generate_test_train_text_file(0.8,"data")

# offline_augmentation('train.txt',data_root,['flip','mirror'])
# utils.rewrite_text_file(data_root,classes_path)
#



# make train and test  dataset
dataset_train =  RobotsHumansDataset(data_root, classes_path, "train", transform=transforms.Compose([transforms.Resize((224,224)),transforms.RandomApply([transforms.ColorJitter(0.1,0.1,0.1),transforms.CenterCrop(100)], 0),transforms.Resize((224,224)), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) ]))
dataset_validation=  RobotsHumansDataset(data_root, classes_path, "validation", transform=transforms.Compose([transforms.Resize((224,224)), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]))
#
# dataset_train =  RobotsHumansDataset(data_root, classes_path, "train", transform=transforms.Compose([transforms.Resize((224,224)),transforms.RandomApply([transforms.CenterCrop(100)], 0),transforms.Resize((224,224)), ]))
# dataset_validation=  RobotsHumansDataset(data_root, classes_path, "validation", transform=transforms.Compose([transforms.Resize((224,224)),]))

a = dataset_train[10][0].permute(1,2,0)
plt.imshow(a)
plt.show()

# make train and test loaders
dataloader_train = DataLoader(dataset_train, batch_size=16,shuffle=True, num_workers=16)
dataloader_validation = DataLoader(dataset_validation, batch_size=16,shuffle=True, num_workers=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def eval_model(model):
    """ Computing model accuracy """
    correct = 0
    total = 0
    loss_list = []

    for batch in dataloader_validation:

        image = batch[0].to(device)
        labels = batch[1].to(device)

        # Forward pass only to get logits/output
        outputs = model(image)

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


def save_model(model, model_name):

    torch.save(model,os.path.join(model_root, "{}.pth".format(model_name)))

LR = 3e-4
EPOCHS = 10
EVAL_FREQ = 1
SAVE_FREQ = 10

stats = {
    "epoch": [],
    "train_loss": [],
    "valid_loss": [],
    "accuracy": []
}

model = FineTune(base_model='resnet18')
model.to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(params=model.parameters(),lr = LR)

init_epoch = 0
loss_hist = []
for epoch in range(2):
    loss_list = []
    progress_bar = tqdm(enumerate(dataloader_train), total=len(dataloader_train))
    for i, batch in progress_bar:
        image = batch[0].to(device)
        labels = batch[1].to(device)

        optimizer.zero_grad()
        y = model(image)
        loss = criterion(y, labels)
        loss_list.append(loss.item())

        loss.backward()
        optimizer.step()
        progress_bar.set_description(f"Epoch {0 + 1} Iter {i + 1}: loss {loss.item():.5f}. ")

    loss_hist.append(np.mean(loss_list))
    stats['epoch'].append(epoch)
    stats['train_loss'].append(loss_hist[-1])

    if epoch % EVAL_FREQ == 0:
        accuracy, valid_loss = eval_model(model)
        print(f"Accuracy at epoch {epoch}: {round(accuracy, 2)}%")
    else:
        accuracy, valid_loss = -1, -1
    stats["accuracy"].append(accuracy)
    stats["valid_loss"].append(valid_loss)

    # saving checkpoint
    if epoch % SAVE_FREQ == 0:
        save_model(model, "fine_tune_resnet_18")