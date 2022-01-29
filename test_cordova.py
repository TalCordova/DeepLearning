## import packages
import numpy as np
import torch
import torch.nn.functional as F
import sklearn
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
import transformers
import tensorflow as tf
from tokenizers import Tokenizer

f = open('atis/intent_label.txt', 'r', encoding="utf8") # opening a file
labels = f.readlines()
labels = [str(i)[:-1] for i in labels]
num_labels = len(labels)
labels = dict(zip(labels, range(0,22)))

## Question 1

##load train text
f = open('atis/train/seq.in', 'r', encoding="utf8") # opening a file
lines = f.readlines()
train_set =[str(i)[:-1] for i in lines]
f = open('atis/train/label', 'r', encoding="utf8") # opening a file
lines = f.readlines()
train_labels = [str(i)[:-1] for i in lines]

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

## non - language model

## language model - BERT
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments

tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True)

bert_input = tokenizer(train_set,padding=True , truncation=True, return_tensors="pt")

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

class Dataset(torch.utils.data.Dataset):

    def __init__(self, text, text_labels):

        self.labels = [labels[label] for label in text_labels]
        self.texts = [tokenizer(t, padding=True, truncation=True, return_tensors="pt") for t in text]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

from torch import nn
from transformers import BertModel

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, num_labels)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


from torch.optim import Adam
from tqdm import tqdm


def train(model, train_data, train_labels, val_data, val_labels,learning_rate, epochs):
    train, val = Dataset(train_data, train_labels), Dataset(val_data, val_labels)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')


EPOCHS = 5
model = BertClassifier()
LR = 1e-6

train(model, train_set,train_labels, dev_set,dev_labels, LR, EPOCHS)