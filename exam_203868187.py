## import packages
import numpy as np
import torch
import torch.nn.functional as F
import sklearn
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
import transformers
import tensorflow as tf

f = open('atis/intent_label.txt', 'r', encoding="utf8") # opening a file
labels = f.readlines()
num_labels = len(labels)

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
devSet =[str(i)[:-1] for i in lines]
f = open('atis/dev/label', 'r', encoding="utf8") # opening a file
lines = f.readlines()
devLabels = [str(i)[:-1] for i in lines]

## non - language model

## language model - BERT
txt = "bank river"
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments

tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
nlp = transformers.TFBertModel.from_pretrained('bert-base-uncased')

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=num_labels)
training_args = TrainingArguments("test_trainer")

embedding = nlp(input_ids)
embedding[0][0]
