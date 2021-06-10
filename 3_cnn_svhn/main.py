import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss().to(device)

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

    def __init__(self, p=0.2):
        """ Model Initializer """

        super().__init__()

        # layer 1
        conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5,stride=1, padding=0)
        relu1 = nn.ReLU()
        maxpool1 = nn.MaxPool2d(kernel_size=2)
        dropout1 = nn.Dropout2d(p)
        self.layer1 = nn.Sequential(
            conv1, relu1, maxpool1, dropout1
        )

        # layer 2
        conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        relu2 = nn.ReLU()
        maxpool2 = nn.MaxPool2d(kernel_size=2)
        dropout2 = nn.Dropout2d(p)
        self.layer2 = nn.Sequential(
            conv2, relu2, maxpool2, dropout2
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



def train_model(model_type = "CNN", model_name = "Basic", regularizer=None, dropout=False, pooling=None):
    """
    Train and save model with given conditions
    """
    LR = 3e-4
    EPOCHS = 10
    EVAL_FREQ = 1
    SAVE_FREQ = 10

    lambda_1 = 0
    lambda_2 = 0
    if regularizer=="L1":
        lambda_1 = 0.0001
    elif regularizer=="L2":
        lambda_2 = 0.001
    elif regularizer=="Elastic":
        lambda_1 = 0.0001
        lambda_2 = 0.001

    p=0
    if dropout:
        p = 0.2

    if model_type == "CNN" :
        model = CNN(p)
        model = model.to(device)
    elif model_type =="MLP":
        model = LogisticRegressionModel()
        model = model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR, weight_decay=1e-5)
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
            pred = model(img)

            penalty_L1 = [torch.sum(torch.abs(param)) for param in model.parameters()]
            penalty_L1 = torch.sum(torch.stack(penalty_L1))

            penalty_L2 = [torch.sum(torch.norm(param)) for param in model.parameters()]
            penalty_L2 = torch.sum(torch.stack(penalty_L2))

            loss = criterion(pred, label) + lambda_1*penalty_L1 + lambda_2*penalty_L2

            loss_list.append(loss.item())

            loss.backward()
            optimizer.step()
            progress_bar.set_description(f"Epoch {epoch + 1} Iter {i + 1}: loss {loss.item():.5f}. ")

        loss_hist.append(np.mean(loss_list))
        stats['epoch'].append(epoch)
        stats['train_loss'].append(loss_hist[-1])
        # evaluating model
        if epoch % EVAL_FREQ == 0:
            accuracy, valid_loss = eval_model(model)
            print(f"Accuracy at epoch {epoch}: {round(accuracy, 2)}%")
        else:
            accuracy, valid_loss = -1, -1
        stats["accuracy"].append(accuracy)
        stats["valid_loss"].append(valid_loss)

        # saving checkpoint
        if epoch % SAVE_FREQ == 0:
            save_model(model=model, optimizer=optimizer, epoch=epoch, stats=stats)

    savepath = os.path.join("trained_models", model_name)
    torch.save({"stats":stats, "model":model}, savepath)

    # torch.save({
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'stats': stats
    # }, savepath)

def visualize_kernels(model, layer=None, num_kernels=None, channels=0):
    filter_weights = list(model.layer1.parameters())[0]
    fig, ax = plt.subplots(4,4)
    for i in range(4):
        for j in range(4):
            ax[j, i].imshow(filter_weights[j*4+i].permute(1,2,0).detach())
    plt.show()

def compare_results(model_list, labels):
    train_loss = {}
    validation_loss = {}
    accuracy = {}
    stat_list = [model["stats"] for model in model_list]
    models = [m["model"] for m in model_list]
    for i,stat in enumerate(stat_list):
        accuracy[labels[i]] =  stat['accuracy']
        validation_loss[labels[i]] = stat['valid_loss']
        train_loss[labels[i]] = stat['train_loss']
    for i,label in enumerate(labels):
        plt.plot(accuracy[label], label=label)
    plt.suptitle("Accuracy comparision")
    plt.legend()
    # plt.show()

    for i,label in enumerate(labels):
        plt.plot(train_loss[label], label=label)
    plt.suptitle("Train  Loss comparision")
    plt.legend()
    # plt.show()

    for i,label in enumerate(labels):
        plt.plot(validation_loss[label], label=label)
    plt.suptitle("Val Loss comparision")
    plt.legend()
    # plt.show()

    testing_accuracy = {}
    for i, model in enumerate(models):
        testing_accuracy[labels[i]] = eval_model(model)[0]
    plt.bar(range(len(testing_accuracy)), list(testing_accuracy.values()), align='center')
    plt.xticks(range(len(testing_accuracy)), list(testing_accuracy.keys()))
    plt.legend()
    plt.show()

if __name__ == '__main__':
    model_L1 = torch.load(os.path.join("trained_models/L1_no_dropout.pth"))
    model_L2= torch.load(os.path.join("trained_models/L2_no_dropout.pth"))
    model_Elastic = torch.load(os.path.join("trained_models/Elastic_no_dropout.pth"))
    model_basic = torch.load(os.path.join("trained_models/basic.pth"))
    model_MLP = torch.load(os.path.join("trained_models/L2_MLP.pth"))
    model_dropout = torch.load(os.path.join("trained_models/L2_dropout.pth"))

    compare_results([model_basic, model_MLP], ["CNN", "MLP"])
    # visualize_kernels(model)
    # print(model)
    # train_model(model_type="CNN",model_name="basic.pth",regularizer="None", dropout=True)