#importing necessary libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.style.use('dark_background')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, GlobalMaxPooling1D, SpatialDropout1D
from string import punctuation

#importing the train dataset
trainDF = pd.read_csv('training_queries.csv')

#extracting test data from text file and appending it to a dataframe
file = open("TestData.txt","r+")
TestText=[]
TestIndex=[]
x=file.readlines()
for i in range(1,426):
    index, text=x[i].split(",", 1)
    TestText.append(text)
    TestIndex.append(int(index))
     
#creating a dataframe using texts and lables
testDF = pd.DataFrame()
testDF['index'] = TestIndex
testDF['text'] = TestText

#defining data pre-processing function of text data
def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in punctuation])
    text = ' '.join(text.split())
    return text

X_train = trainDF['text'].apply(lambda p: clean_text(p))
y_train = trainDF['label']
X_test = testDF['text'].apply(lambda p: clean_text(p))

#Let’s look at the individual length of each phrase in the corpus.
phrase_len = X_train.apply(lambda p: len(p.split(' ')))
max_phrase_len = phrase_len.max()
print('max phrase len: {0}'.format(max_phrase_len))
plt.figure(figsize = (10, 8))
plt.hist(phrase_len, alpha = 0.2, density = True)
plt.xlabel('phrase len')
plt.ylabel('probability')
plt.grid(alpha = 0.25)

#tokenizer to parse the phrases
max_words = 8192
tokenizer = Tokenizer(
    num_words = max_words,
    filters = '"#$%&()*+-/:;<=>@[\]^_`{|}~'
)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
#pad_sequence is used to ensure that all the phrase are the same length
X_train = pad_sequences(X_train, maxlen = max_phrase_len)
X_test = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(X_test, maxlen = max_phrase_len)
y_train = to_categorical(y_train)
 

#model using a LSTM layer
model_lstm = Sequential()
model_lstm.add(Embedding(input_dim = max_words, output_dim = 256, input_length = max_phrase_len))
model_lstm.add(SpatialDropout1D(0.3))
model_lstm.add(LSTM(256, dropout = 0.3, recurrent_dropout = 0.3))
model_lstm.add(Dense(256, activation = 'relu'))
model_lstm.add(Dropout(0.3))
model_lstm.add(Dense(10, activation = 'softmax'))
model_lstm.compile(
    loss='categorical_crossentropy',
    optimizer='Adam',
    metrics=['accuracy']
)
 
history = model_lstm.fit(
    X_train,
    y_train,
    validation_split = 0.1,
    epochs = 50,
    batch_size = 512
)

#plot of training and validation loss with each epoch
plt.clf()
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'y', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


#plot of training and validation accuracy with each epoch
plt.clf()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'g', label='Training acc')
plt.plot(epochs, val_acc, 'y', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#predicting the test set results
y_pred = model_lstm.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)






