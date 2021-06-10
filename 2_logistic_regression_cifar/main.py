import torch
import torch.nn as nn
import numpy as np
import tqdm
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import random as rand

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainset = dsets.CIFAR10('data', train=True, download=True, transform=transforms.ToTensor())
testset = dsets.CIFAR10('data', train=False, download=True, transform=transforms.ToTensor())

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

BATCH_SIZE = 256
train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=testset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False)
# defining model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim=32*32*3, hidden_dim_1=512,hidden_dim_2 =256, output_dim=10):
        super().__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2,output_dim)
        )

    def forward(self,x):
        x_flat = x.view(-1,self.input_dim)
        out = self.model(x_flat)
        return out

def count_model_params(model):
    """ Counting the number of learnable parameters in a nn.Module """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params

# instanciating model
model = LogisticRegressionModel()
model  = model.to(device)

n_params = count_model_params(model)
print(f"Model learnable parameters: {n_params} ")

N_ITERS = 1000
N_EPOCHS = 3
loss_hist = []
lr = 3e-3

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# for epoch in range(N_EPOCHS):
#     loss_list = []
#     progress_bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
#     for i, (imgs, labels) in progress_bar:
#
#         # everything needs to be on the same device
#         imgs = imgs.to(device)
#         labels = labels.to(device)
#
#         # forward pass
#         pred_labels = model(imgs)
#
#         # computing error
#         loss = criterion(pred_labels, labels)
#         loss_list.append(loss.item())
#
#         # removing accumulated gradients
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if (i % 1 == 0 or i == N_ITERS - 1):
#             progress_bar.set_description(f"Epoch {epoch + 1} Iter {i + 1}: loss {loss.item():.5f}. ")
#
#     loss_hist.append(np.mean(loss_list))
#
# n_correct = 0
# label_list = []
# pred_list = []
# with torch.no_grad():
#     progress_bar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader))
#     for i, (imgs, labels) in progress_bar:
#
#         # everything needs to be on the same device
#         imgs = imgs.to(device)
#         labels = labels.to(device)
#         label_list.append(labels)
#         # forward pass
#         pred_labels = model(imgs)
#         preds = torch.argmax(pred_labels, dim=-1)
#         pred_list.append(preds)
#         cur_correct = len(torch.where(preds == labels)[0])
#         n_correct = n_correct + cur_correct
#
# all_labels = torch.cat(label_list).to('cpu')
# all_predictions = torch.cat(pred_list).to('cpu')
# c_matrix = confusion_matrix(all_labels, all_predictions)
# accuracy = n_correct / len(testset) * 100
# print(f"Test accuracy: {round(accuracy, 2)}%")
#
# # hyperparameter testing
# N_EPOCHS =[5,10,30]
# loss_hist_epochs = {}
# accuracy_epochs = {}
# for e in N_EPOCHS:
#     for epoch in range(e):
#         loss_list = []
#         progress_bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
#         for i, (imgs, labels) in progress_bar:
#
#             # everything needs to be on the same device
#             imgs = imgs.to(device)
#             labels = labels.to(device)
#
#             # forward pass
#             pred_labels = model(imgs)
#
#             # computing error
#             loss = criterion(pred_labels, labels)
#             loss_list.append(loss.item())
#
#             # removing accumulated gradients
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             if (i % 1 == 0 or i == N_ITERS - 1):
#                 progress_bar.set_description(f"Epoch {epoch + 1} Iter {i + 1}: loss {loss.item():.5f}. ")
#
#         loss_hist.append(np.mean(loss_list))
#     loss_hist_epochs[e] = loss_hist
#
#     n_correct = 0
#     label_list = []
#     pred_list = []
#     with torch.no_grad():
#         progress_bar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader))
#         for i, (imgs, labels) in progress_bar:
#             # everything needs to be on the same device
#             imgs = imgs.to(device)
#             labels = labels.to(device)
#             label_list.append(labels)
#             # forward pass
#             pred_labels = model(imgs)
#             preds = torch.argmax(pred_labels, dim=-1)
#             pred_list.append(preds)
#             cur_correct = len(torch.where(preds == labels)[0])
#             n_correct = n_correct + cur_correct
#
#     all_labels = torch.cat(label_list).to('cpu')
#     all_predictions = torch.cat(pred_list).to('cpu')
#     c_matrix = confusion_matrix(all_labels, all_predictions)
#     accuracy = n_correct / len(testset) * 100
#     accuracy_epochs[e] = accuracy
#     print(f"Test accuracy: {round(accuracy, 2)}%")


# learning rate
lr_list =[3e-4,3e-3,3e-2]
N_EPOCHS = 10
loss_hist_lr = {}
accuracy_lr = {}
for lr in lr_list:
    model = LogisticRegressionModel()
    model  = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(N_EPOCHS):
        loss_list = []
        progress_bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (imgs, labels) in progress_bar:
            # everything needs to be on the same device
            imgs = imgs.to(device)
            labels = labels.to(device)

            # forward pass
            pred_labels = model(imgs)

            # computing error
            loss = criterion(pred_labels, labels)
            loss_list.append(loss.item())

            # removing accumulated gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i % 1 == 0 or i == N_ITERS - 1):
                progress_bar.set_description(f"Epoch {epoch + 1} Iter {i + 1}: loss {loss.item():.5f}. ")

        loss_hist.append(np.mean(loss_list))
    loss_hist_lr[lr] = loss_hist

    n_correct = 0
    label_list = []
    pred_list = []
    with torch.no_grad():
        progress_bar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader))
        for i, (imgs, labels) in progress_bar:
            # everything needs to be on the same device
            imgs = imgs.to(device)
            labels = labels.to(device)
            label_list.append(labels)
            # forward pass
            pred_labels = model(imgs)
            preds = torch.argmax(pred_labels, dim=-1)
            pred_list.append(preds)
            cur_correct = len(torch.where(preds == labels)[0])
            n_correct = n_correct + cur_correct

    all_labels = torch.cat(label_list).to('cpu')
    all_predictions = torch.cat(pred_list).to('cpu')
    c_matrix = confusion_matrix(all_labels, all_predictions)
    accuracy = n_correct / len(testset) * 100
    accuracy_lr[lr] = accuracy
    print(f"Test accuracy: {round(accuracy, 2)}%")

