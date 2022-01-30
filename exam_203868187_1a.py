
## ------------------------------------------------ Import  Packages-------------------------------------------------
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers import Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

## ------------------------------------Label Dictionary------------------------------------------------

file = open('atis/intent_label.txt', 'r', encoding="utf8") # opening a file
labels = file.readlines()
labels = [str(i)[:-1] for i in labels]
num_labels = len(labels)
labels = dict(zip(labels, range(0, 22)))

indices = list(labels.values())
depth = len(labels)
one_hot_labels = tf.one_hot(indices, depth)
one_hot_labels = np.array(one_hot_labels)
one_hot_dict = dict.fromkeys(labels.keys())
i = 0
for key in one_hot_dict.keys():
    one_hot_dict[key] = one_hot_labels[i]
    i = i+1

## ------------------------------------------- Data from Atis --------------------------------------------

# train data
file = open('atis/train/seq.in', 'r', encoding="utf8") # opening a file
lines = file.readlines()
train_set = [str(x)[:-1] for x in lines]
file = open('atis/train/label', 'r', encoding="utf8") # opening a file
lines = file.readlines()
train_labels = [str(line)[:-1] for line in lines]
train_labels = [one_hot_dict.get(label, one_hot_dict.get("UNK")) for label in train_labels]

# validation data
file = open('atis/dev/seq.in', 'r', encoding="utf8") # opening a file
lines = file.readlines()
val_set = [str(line)[:-1] for line in lines]
file = open('atis/dev/label', 'r', encoding="utf8") # opening a file
lines = file.readlines()
val_labels = [str(line)[:-1] for line in lines]
val_labels = [one_hot_dict.get(label, one_hot_dict.get("UNK")) for label in val_labels]

# test data
file = open('atis/test/seq.in', 'r', encoding="utf8") # opening a file
lines = file.readlines()
test_set = [str(line)[:-1] for line in lines]
file = open('atis/test/label', 'r', encoding="utf8") # opening a file
lines = file.readlines()
test_labels = [str(line)[:-1] for line in lines]
test_labels = [one_hot_dict.get(label, one_hot_dict.get("UNK")) for label in test_labels]

maxlen = max([len(i) for i in train_set])

## ------------------------------------------------ Prepare Data -----------------------------------------------------------------------

num_words = 1000
oov_token = '<UNK>'
pad_type = 'post'
trunc_type = 'post'

tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
tokenizer.fit_on_texts(train_set)

word_index = tokenizer.word_index
train_sequences = tokenizer.texts_to_sequences(train_set)

max_len = max([len(x) for x in train_sequences])
train_padded = pad_sequences(train_sequences, padding=pad_type, truncating=trunc_type, maxlen=max_len)
train_padded = np.array(train_padded)

val_sequence = tokenizer.texts_to_sequences(val_set)
val_padded = pad_sequences(val_sequence, padding=pad_type, truncating=trunc_type, maxlen=max_len)
val_padded = np.array(val_padded)

test_sequence = tokenizer.texts_to_sequences(test_set)
test_padded = pad_sequences(test_sequence, padding=pad_type, truncating=trunc_type, maxlen=max_len)
test_padded = np.array(test_padded)

train_labels = np.array(train_labels, dtype=object).astype(int)
val_labels = np.array(val_labels, dtype=object).astype(int)
test_labels = np.array(test_labels, dtype=object).astype(int)


## --------------------------------------------------------Train Model LSTM -------------------------------------------------------------------

max_sentence_length = max_len
embedding_vector_length = len(train_padded)
model = Sequential()
model.add(Embedding(input_length=max_sentence_length, output_dim=num_labels, input_dim=embedding_vector_length))
model.add(LSTM(200))
model.add(Dense(num_labels, activation='softmax'))
optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
print(model.summary())

num_epochs = 30
batch_size = 64
model.fit(train_padded, train_labels, validation_data=(val_padded, val_labels), epochs=num_epochs, batch_size=batch_size, verbose=1)

## --------------------------------------------------------Evaluate LSTM -------------------------------------------------------------------
scores = model.evaluate(test_padded, test_labels, verbose=1)
print("\nTest Accuracy: %.2f%%" % (scores[1]*100))