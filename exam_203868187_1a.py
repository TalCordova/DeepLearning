import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.ar_model import ar_select_order
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from collections import Counter

# load dataset
f = open('atis/intent_label.txt', 'r', encoding="utf8") # opening a file
labels = f.readlines()
labels = [str(i)[:-1] for i in labels]
num_labels = len(labels)
labels = dict(zip(labels, range(1,23)))

## Question 1

le = LabelEncoder()
ohe = OneHotEncoder(sparse=False)

##load train text
f = open('atis/train/seq.in', 'r', encoding="utf8") # opening a file
lines = f.readlines()
train_set =[str(i)[:-1] for i in lines]
splitted_train_set = [str(i)[:-1].split() for i in lines]
f = open('atis/train/label', 'r', encoding="utf8") # opening a file
lines = f.readlines()
train_labels = [str(i)[:-1] for i in lines]
s = set()
for line in splitted_train_set:
    s.update(line)
voc = dict(zip(s, range(1,len(s)+1)))
ohe.fit(np.array(voc.values()).reshape(-1,1))
for key in voc.keys():
    voc[key] = ohe.transform(np.array(voc[key]).reshape(-1,1))

train_set_temp = []
for sentence in splitted_train_set:
    train_set_temp.append([voc[word] for word in sentence])


##load test text
f = open('atis/test/seq.in', 'r', encoding="utf8") # opening a file
lines = f.readlines()
test_set =[str(i)[:-1] for i in lines]
f = open('atis/test/label', 'r', encoding="utf8") # opening a file
lines = f.readlines()
test_labels = [str(i)[:-1] for i in lines]

##load dev text
f = open('atis/dev/seq.in', 'r', encoding="utf8") # opening a file
lines = f.readlines()
dev_set =[str(i)[:-1] for i in lines]
f = open('atis/dev/label', 'r', encoding="utf8") # opening a file
lines = f.readlines()
dev_labels = [str(i)[:-1] for i in lines]

class lstm(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        hidden_state = torch.randn(1, 1, hidden_size)
        cell_state = torch.randn(1, 1, hidden_size)
        self.hidden = (hidden_state, cell_state)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.lin = nn.Linear(hidden_size, num_labels)
    def forward(self, input):
        lstm_out, _ = self.lstm(input)
        x = self.lin(lstm_out[:, -1, :])
        return nn.Softmax(x)

class GDPDataset(Dataset):
    def __init__(self, data, window_size=5):
        self.train_data = []
        self.train_labels = []
        for i in range(len(data) - window_size):
            self.train_data.append(data[i:i+window_size])
            self.train_labels.append(data[i+window_size, 1])
        self.train_data = torch.Tensor(self.train_data)
        self.train_labels = torch.Tensor(self.train_labels)
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
    print(f"Test Error: \n MSE: {(test_loss):>0.1f}%")
    return test_loss


# hyper parameters
learning_rate = 1e-2
batch_size = 64
epochs = 10

# build dataset

trainDatast = GDPDataset(train_set)
evalDatast = GDPDataset(dev_set)
testDatast = GDPDataset(test_set)

# data loaders
train_dataloader = DataLoader(trainDatast, batch_size=16, shuffle=True)
eval_dataloader = DataLoader(evalDatast, batch_size=16, shuffle=True)
test_dataloader = DataLoader(testDatast, batch_size=16, shuffle=True)


loss = nn.MSELoss()
hidden_size = 32
res = {'hidden_size': [], 'test_loss': []}
model = lstm(input_size=8, hidden_size=hidden_size)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss, optimizer)
    eval_loss = test_loop(eval_dataloader, model, loss)
    test_loss = test_loop(test_dataloader, model, loss)
res['test_loss'].append(test_loss)
res = pd.DataFrame(res)
print(f"best hidden size of lstm with window size=5: {res[res['test_loss']==res['test_loss'].min()]['hidden_size'].values[0]}")
