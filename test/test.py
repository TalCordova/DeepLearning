## import packages
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
import transformers
import numpy as np
import torch
from torch import nn
from transformers import BertModel
from torch.optim import Adam
from tqdm import tqdm

f = open('atis/intent_label.txt', 'r', encoding="utf8") # opening a file
labels = f.readlines()
labels = [str(i)[:-1] for i in labels]
num_labels = len(labels)
labels = dict(zip(labels, range(1, 23)))

# load train text
f = open('atis/train/seq.in')
content = f.readlines()
train_text = [str(x)[:-1] for x in content]
f = open('atis/train/label')
content = f.readlines()
train_labels = [str(x)[:-1] for x in content]
train_labels = [labels.get(label, 1) for label in train_labels]

# load dev text
f = open('atis/dev/seq.in')
content = f.readlines()
dev_text = [str(x)[:-1] for x in content]
f = open('atis/dev/label')
content = f.readlines()
dev_labels = [str(x)[:-1] for x in content]
dev_labels = [labels.get(label, 1) for label in dev_labels]


# load test text
f = open('atis/test/seq.in')
content = f.readlines()
test_text = [str(x)[:-1] for x in content]
f = open('atis/test/label')
content = f.readlines()
test_labels = [str(x)[:-1] for x in content]
test_labels = [labels.get(label, 1) for label in test_labels]


# convert text to tf-idf
vectorizer = TfidfVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(train_text)
X_test = vectorizer.transform(test_text)

# train non-LM model
# mlp = MLPClassifier(verbose=1).fit(X_train, train_labels)
# print('Train Accuracy Score: ', mlp.score(X_train, train_labels))
# print(f'Test Prediction\n {mlp.predict(X_test)}\nTest Accuracy Score: {mlp.score(X_test, test_labels)}')


# train LM-Model
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

class Dataset(torch.utils.data.Dataset):

    def __init__(self, text, text_labels):

        self.labels = text_labels
        self.texts = [tokenizer(t, padding='max_length',max_length=512,  truncation=True, return_tensors="pt") for t in text]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        # return torch.Tensor(self.labels[idx])
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert = self.bert.to("cuda")
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, num_labels)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


def train(model, train_data, train_labels, val_data, val_labels, learning_rate, epochs):
    train, val = Dataset(train_data, train_labels), Dataset(val_data, val_labels)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

    ##use_cuda = torch.cuda.is_available()
    use_cuda = False
    ##device = torch.device("cuda" if use_cuda else "cpu")
    device = "cpu"

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

            batch_loss = criterion(output, train_label.type(torch.LongTensor))
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


def evaluate(model, test_data, test_labels):
    test = Dataset(test_data, test_labels)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=10)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():

        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')
    print(f'Model Predictions: {output.argmax(dim=1)}')


EPOCHS = 1
model = BertClassifier()
model
LR = 1e-4

train(model, train_text, train_labels, dev_text, dev_labels, LR, EPOCHS)
evaluate(model, test_text, test_labels)
