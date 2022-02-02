import torch
import pandas as pd
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm


# load dataset
# todo: change the root param
dataset_train = CIFAR10(root='/Users/dor/PycharmProjects/Deep_Learning/', download=True, transform=ToTensor())

# todo: change the root param
dataset_test = CIFAR10(root='/Users/dor/PycharmProjects/Deep_Learning/', download=False, train=False, transform=ToTensor())


class ANN(nn.Module):
    def __init__(self, img_size=32, hidden_size=128, output_dimension=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.Linear1 = nn.Linear(img_size**2*3, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.Linear2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.bn2 = nn.BatchNorm1d(int(hidden_size/2))
        self.output = nn.Linear(int(hidden_size/2), output_dimension)
        self.dropout = nn.Dropout(0.3)
    def forward(self, input):
        flatten = self.flatten(input)
        Linear1 = self.Linear1(flatten)
        bn1 = self.bn1(Linear1)
        actvivition1 = nn.ReLU()(bn1)
        dropout1 = self.dropout(actvivition1)
        Linear2 = self.Linear2(dropout1)
        bn2 = self.bn2(Linear2)
        actvivition2 = nn.ReLU()(bn2)
        dropout2 = self.dropout(actvivition2)
        output = self.output(dropout2)
        return nn.Softmax(dim=1)(output)

class CNN(nn.Module):
    def __init__(self, img_size=32, hidden_size=128, output_dimension=10):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(7, 7))
        self.conv2d_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(7, 7))
        self.conv2d_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(7, 7))
        self.flatten = nn.Flatten()
        self.Linear1 = nn.Linear(12544, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.Linear2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.bn2 = nn.BatchNorm1d(int(hidden_size/2))
        self.output = nn.Linear(int(hidden_size/2), output_dimension)
        self.dropout = nn.Dropout(0.3)
    def forward(self, input):
        torch.swapdims(input, 1, -1)
        conv2d_1 = self.conv2d_1(input)
        relu_1 = nn.ReLU()(conv2d_1)
        conv2d_2 = self.conv2d_2(relu_1)
        relu_2 = nn.ReLU()(conv2d_2)
        conv2d_3 = self.conv2d_3(relu_2)
        relu_3 = nn.ReLU()(conv2d_3)
        flatten = self.flatten(relu_3)
        Linear1 = self.Linear1(flatten)
        bn1 = self.bn1(Linear1)
        actvivition1 = nn.ReLU()(bn1)
        dropout1 = self.dropout(actvivition1)
        Linear2 = self.Linear2(dropout1)
        bn2 = self.bn2(Linear2)
        actvivition2 = nn.ReLU()(bn2)
        dropout2 = self.dropout(actvivition2)
        output = self.output(dropout2)
        return nn.Softmax(dim=1)(output)


def train_loop(dataloader, model, loss_fn, optimizer):
    for X, y in tqdm(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, f'{100*correct:.2f}%'

# hyper parameters
learning_rate = 1e-2
batch_size = 64
epochs = 10

# data loaders
train_dataloader = DataLoader(dataset_train, batch_size=64, shuffle=True)
test_dataloader = DataLoader(dataset_test, batch_size=64, shuffle=True)

# train ANN model
model = ANN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# init dataframe to save results
res = {'model': [], 'test_loss': [], 'test_acc': []}
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loss, test_acc = test_loop(test_dataloader, model, loss_fn)

# save ANN results
res['model'].append('ANN')
res['test_loss'].append(test_loss)
res['test_acc'].append(test_acc)

# train CNN model
model = CNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# init dataframe to save results
res = {'model': [], 'test_loss': [], 'test_acc': []}
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loss, test_acc = test_loop(test_dataloader, model, loss_fn)

# save ANN results
res['model'].append('CNN')
res['test_loss'].append(test_loss)
res['test_acc'].append(test_acc)

# save dataframe to csv
df = pd.DataFrame(res)
df.to_csv('results.csv')
