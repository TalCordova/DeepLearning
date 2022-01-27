import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

np.random.seed(2022)
A = np.array([[np.sqrt(99)/10, -0.1], [0.1, np.sqrt(99)/10]])
B = np.array([[np.sqrt(2)/2, -1], [np.sqrt(2)/2, np.sqrt(2)/2]])
Sigma_1 = 0.01
Sigma_2 = 0.2

init_state = np.array([1, 0])
process_len = 2000

# generate the data with Kalman filter
X = []
Y = []
for i in range(process_len):
    X.append(np.matmul(A, init_state) + np.random.normal(0, Sigma_1**2))
    Y.append(np.matmul(B, X[i]) + np.random.normal(0, Sigma_2 ** 2))
    init_state = X[i]

# split train & test
train_labels = np.array(X[:1000])
test_labels = np.array(X[1000:])
train_data = np.array(Y[:1000])
test_data = np.array(Y[1000:])


class lstm(nn.Module):
    def __init__(self, input_size=2, hidden_size=2):
        super().__init__()
        hidden_state = torch.randn(1, 1, hidden_size)
        cell_state = torch.randn(1, 1, hidden_size)
        self.hidden = (hidden_state, cell_state)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
    def forward(self, input):
        lstm_out, _ = self.lstm(input)
        return lstm_out[:, -1, :]

class KalmanDataset(Dataset):
    def __init__(self, data, labels):
        self.train_data = torch.Tensor(np.expand_dims(data, axis=1))
        self.train_labels = torch.Tensor(labels)

    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, idx):
        return self.train_data[idx], self.train_labels[idx]

def train_loop(dataloader, model, loss_fn, optimizer):
    for X, y in dataloader:
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test_loop(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(f"MSE: {(test_loss):>0.1f}\n")
    return test_loss

# hyper parameters
learning_rate = 1e-2
batch_size = 64
epochs = 20

# build dataset

trainDatast = KalmanDataset(train_data, train_labels)
testDatast = KalmanDataset(test_data, test_labels)

# data loaders
train_dataloader = DataLoader(trainDatast, batch_size=64, shuffle=True)
test_dataloader = DataLoader(testDatast, batch_size=64, shuffle=True)


loss = nn.MSELoss()
model = lstm()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss, optimizer)
    test_loss = test_loop(test_dataloader, model, loss)

# B - predict the next observation
print(f'The next obser{model(torch.Tensor(test_data[0].reshape(1, 1, 2))).detach().numpy()[0]}')
