
## import packages
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


##label dictionary
f = open('atis/intent_label.txt', 'r', encoding="utf8") # opening a file
labels = f.readlines()
labels = [str(i)[:-1] for i in labels]
num_labels = len(labels)
labels = dict(zip(labels, range(0, 22)))


# load train text
f = open('atis/train/seq.in')
content = f.readlines()
train_text = [str(x)[:-1].split() for x in content]
f = open('atis/train/label')
content = f.readlines()
train_labels = [str(x)[:-1] for x in content]
train_labels = [labels.get(label, 0) for label in train_labels]


# load dev text
f = open('atis/dev/seq.in')
content = f.readlines()
dev_text = [str(x)[:-1] for x in content]
f = open('atis/dev/label')
content = f.readlines()
dev_labels = [str(x)[:-1] for x in content]
dev_labels = [labels.get(label, 0) for label in dev_labels]

# load test text
f = open('atis/test/seq.in')
content = f.readlines()
test_text = [str(x)[:-1] for x in content]
f = open('atis/test/label')
content = f.readlines()
test_labels = [str(x)[:-1] for x in content]
test_labels = [labels.get(label, 0) for label in test_labels]

## prepare data
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
num_words = 1000
oov_token = '<UNK>'
pad_type = 'post'
trunc_type = 'post'

tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
tokenizer.fit_on_texts(train_text)

word_index = tokenizer.word_index
train_sequences = tokenizer.texts_to_sequences(train_text)

maxlen = max([len(x) for x in train_sequences])
train_padded = pad_sequences(train_sequences, padding=pad_type, truncating=trunc_type, maxlen=maxlen)
train_padded = np.array(train_padded)

val_sequence = tokenizer.texts_to_sequences(dev_text)
val_padded = pad_sequences(val_sequence, padding=pad_type, truncating=trunc_type, maxlen=maxlen)
val_padded = np.array(val_padded)

test_sequence = tokenizer.texts_to_sequences(test_text)
test_padded = pad_sequences(test_sequence, padding=pad_type, truncating=trunc_type, maxlen=maxlen)
test_padded = np.array(test_padded)

train_labels = np.array(train_labels)
dev_labels = np.array(dev_labels)
test_labels = np.array(test_labels)


## train model
max_sentence_length = maxlen
embedding_vector_length = len(train_padded)
model = Sequential()
model.add(Embedding(input_length=max_sentence_length, output_dim=num_labels, input_dim=embedding_vector_length))
model.add(Dropout(0.2))
model.add(LSTM(num_labels))
model.add(Dropout(0.2))
model.add(Dense(1, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(train_padded, train_labels.reshape(-1,1), validation_data=(val_padded, dev_labels.reshape(-1,1)), epochs=10, batch_size=64, verbose=1)

scores = model.evaluate(test_padded, test_labels.reshape(-1, 1), verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))