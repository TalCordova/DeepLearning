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
from sklearn.preprocessing import StandardScaler


# load dataset
filepath = 'https://raw.githubusercontent.com/selva86/datasets/master/Raotbl6.csv'
df = pd.read_csv(filepath, parse_dates=['date'], index_col='date')

# test for causality between two time series - RGNP & GNP
test = 'ssr_chi2test'
test_result = grangercausalitytests(df[['rgnp','pgnp']], maxlag=10, verbose=False)
p_values = [round(test_result[i+1][0][test][1],4) for i in range(10)]
print('p_values = ',p_values)
print('min_p_value = ',np.min(p_values))

# test for stationarity
r = adfuller(df['rgnp'], autolag='AIC')
output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
p_value = output['pvalue']
print(output)

# difference the series to get stationarity
v = df.rgnp.values
dv = np.diff(v)
r = adfuller(dv, autolag='AIC')
output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
p_value = output['pvalue']
print(output)

# split train & test
train = dv[:-10]
test = dv[-10:]

# A - predict last 10 observations of rgnp with AR
sel = ar_select_order(train, 5, ic='aic')
res = sel.model.fit()
print(res.summary())
print(res.model.predict(params=res.params, start=len(train), end=len(train)+len(test)-1, dynamic=False))

# B - predict last 10 observations of rgnp & pgnp with VAR
v = df.loc[:, ['rgnp', 'pgnp']].values
dv = np.diff(v, axis=0)
train = dv[:-10]
test = dv[-10:]

model = VAR(train)
results = model.fit(maxlags=5, ic='aic')
print(results.summary())
lag_order = results.k_ar
print(results.forecast(train, 5))

# C - predict last 10 observations with LSTM
class lstm(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        hidden_state = torch.randn(1, 1, hidden_size)
        cell_state = torch.randn(1, 1, hidden_size)
        self.hidden = (hidden_state, cell_state)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.lin = nn.Linear(hidden_size, 1)
    def forward(self, input):
        lstm_out, _ = self.lstm(input)
        x = self.lin(lstm_out[:, -1, :])
        return x

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

# use all data to predict the rgnp - assume window_size = 5
v = df.values
dv = np.diff(v, axis=0)
ss = StandardScaler(copy=False)
dv = ss.fit_transform(dv)
train = dv[:-10]
test = dv[-10:]

# hyper parameters
learning_rate = 1e-2
batch_size = 64
epochs = 10

# build dataset

trainDatast = GDPDataset(train)
testDatast = GDPDataset(test)

# data loaders
train_dataloader = DataLoader(trainDatast, batch_size=16, shuffle=True)
test_dataloader = DataLoader(testDatast, batch_size=16, shuffle=True)


loss = nn.MSELoss()
hidden_sizes = [4, 8, 16, 32, 64, 128, 256]
res = {'hidden_size': [], 'test_loss': []}
for h in hidden_sizes:
    model = lstm(input_size=8, hidden_size=h)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss, optimizer)
        test_loss = test_loop(test_dataloader, model, loss)
    res['test_loss'].append(test_loss)
    res['hidden_size'].append(h)
res = pd.DataFrame(res)
print(f"best hidden size of lstm with window size=5: {res[res['test_loss']==res['test_loss'].min()]['hidden_size'].values[0]}")

######################################
#        best hidden size = 32
######################################