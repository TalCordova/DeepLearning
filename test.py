from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
import transformers
import numpy as np
from transformers import AutoModelForSequenceClassification

f = open('atis/intent_label.txt')
content = f.readlines()
numLabels = len(content)

# load train text
f = open('atis/train/seq.in')
content = f.readlines()
train_text = [str(x)[:-1] for x in content]
f = open('atis/train/label')
content = f.readlines()
train_labels = [str(x)[:-1] for x in content]

# load dev text
f = open('atis/dev/seq.in')
content = f.readlines()
dev_text = [str(x)[:-1] for x in content]
f = open('atis/dev/label')
content = f.readlines()
dev_labels = [str(x)[:-1] for x in content]

# load test text
f = open('atis/test/seq.in')
content = f.readlines()
test_text = [str(x)[:-1] for x in content]
f = open('atis/test/label')
content = f.readlines()
test_labels = [str(x)[:-1] for x in content]

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
nlp = transformers.TFBertModel.from_pretrained('bert-base-uncased')

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch


class IntentDataset(Dataset):
    def __init__(self, text, labels):
        tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        input_ids = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        self.text = input_ids
        self.labels = labels
        # self.train_data = torch.Tensor(self.train_data)
        # self.train_labels = torch.Tensor(self.train_labels)
    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.text[idx], self.labels[idx]

train_input_ids = tokenizer(train_text, padding=True, truncation=True, return_tensors="pt")
train_embeddings = nlp(input_ids)

input_ids = tokenizer(dev_text, padding=True, truncation=True, return_tensors="pt")
dev_embeddings = nlp(input_ids)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=numLabels)

from transformers import TrainingArguments

training_args = TrainingArguments("test_trainer")

from transformers import Trainer

train_dataset = IntentDataset(train_text, train_labels)
dev_dataset = IntentDataset(dev_text, dev_labels)

trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=dev_dataset)
trainer.train()
