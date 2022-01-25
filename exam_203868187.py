## import packages
import numpy as np
import torch
import torch.nn.functional as F
import sklearn
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
import transformers
import tensorflow as tf

## Question 1

##load train text
f = open('atis/train/seq.in', 'r', encoding="utf8") # opening a file
lines = f.readlines()
corpus = f.read()
corpus1 = set(corpus)
trainSet =[str(i)[:-1].split() for i in lines]
f = open('atis/label/seq.in', 'r', encoding="utf8") # opening a file
lines = f.readlines()
trainLabels = [str(i)[:-1].split() for i in lines]

integer_encoded = []
for i in corpus.split():
    v = np.where(np.array(corpus1) == i)[0][0]
    integer_encoded.append(v)
print("\ninteger encoded: ", integer_encoded)


def get_vec(len_doc, word):
    empty_vector = [0] * len_doc
    vect = 0
    find = np.where(np.array(doc1) == word)[0][0]
    empty_vector[find] = 1
    return empty_vector


def get_matrix(doc1):
    mat = []
    len_doc = len(doc1)
    for i in docs:
        vec = get_vec(len_doc, i)
        mat.append(vec)

    return np.asarray(mat)


##load test text
f = open('atis/test/seq.in', 'r', encoding="utf8") # opening a file
lines = f.readlines()
testSet =[str(i)[:-1].split() for i in lines]
f = open('atis/test/seq.in', 'r', encoding="utf8") # opening a file
lines = f.readlines()
testLabels = [str(i)[:-1].split() for i in lines]

##load dev text
f = open('atis/dev/seq.in', 'r', encoding="utf8") # opening a file
lines = f.readlines()
devSet =[str(i)[:-1].split() for i in lines]
f = open('atis/dev/seq.in', 'r', encoding="utf8") # opening a file
lines = f.readlines()
devLabels = [str(i)[:-1].split() for i in lines]

## non - language model

## language model - using transformers
txt = "bank river"

tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
input_ids = tokenizer("Hello, my dog is cute", return_tensors="tf")
nlp = transformers.TFBertModel.from_pretrained('bert-base-uncased')
embedding = nlp(input_ids)
embedding[0][0]
