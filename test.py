from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
import transformers
import numpy as np

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
# input_ids = np.array(tokenizer.encode(train_text))
input_ids = tokenizer("Hello, my dog is cute", return_tensors="tf")
model = transformers.TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
outputs = model(input_ids)