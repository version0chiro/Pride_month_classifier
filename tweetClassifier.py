from __future__ import print_function
import os
import zipfile
import plaidml.keras
plaidml.keras.install_backend()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from scipy.special import expit
from keras.models import Sequential
from keras import layers
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# filepath_dict = {'yelp':   'data/sentiment_analysis/yelp_labelled.txt',
#                  'amazon': 'data/sentiment_analysis/amazon_cells_labelled.txt',
#                  'imdb':   'data/sentiment_analysis/imdb_labelled.txt'}
#
# df_list = []
#
# for source,filepath in filepath_dict.items():
#     df = pd.read_csv(filepath,names=['sentence','label'],sep='\t')
#     df['source'] = source #Add another column filld with the source name
#     df_list.append(df)
#
# df = pd.concat(df_list)
# print(df.iloc[0])
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
df = pd.read_csv("data/DataFiles/sentiment140-subset.csv", nrows=30000)
# print(df.head())



sentences = df['text'].values

y = df['polarity'].values

sentences_train, sentences_text, y_train, y_test = train_test_split(sentences,y,test_size=0.25,random_state=1000)

vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)

X_train = vectorizer.transform(sentences_train)
X_test = vectorizer.transform(sentences_text)

encoder = LabelEncoder()

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)

X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_text)

vocab_size = len(tokenizer.word_index)  + 1

print(sentences_train[2])

print(X_train[2])

maxlen=140

X_train=pad_sequences(X_train,padding='post',maxlen=maxlen)
X_test=pad_sequences(X_test,padding='post',maxlen=maxlen)

embedding_dim = 50

model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size,
                           output_dim=embedding_dim,
                           input_length=maxlen))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train,
                    epochs=5,
                    validation_data=(X_test, y_test),
                    batch_size=50)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(history)

# #
# classifier = LogisticRegression(max_iter=10000)
# classifier.fit(X_train,y_train)
# score = classifier.score(X_test,y_test)
# #
# print("Accuracy:",score)

# input_dim = X_train.shape[1] #Number of Features
#
# model = Sequential()
# model.add(layers.Dense(10,input_dim=input_dim,activation='relu'))
# model.add(layers.Dense(1,activation='sigmoid'))
#
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
#
# print(model.summary())
#
# history = model.fit(X_train,y_train,
#                     epochs=10,
#                     validation_data=(X_test,y_test),
#                     batch_size=10)
#
# loss,accuracy = model.evaluate(X_train,y_train,verbose=False)
# print("Training Accuracy : {:.4f}".format(accuracy))
# loss,accuracy = model.evaluate(X_test,y_test,verbose=False)
# print("Testing Accuracy: {:.4f}".format(accuracy))
#
# model.save('SavedModel/3rdModel')
#
# def plot_history(history):
#     acc = history.history['acc']
#     val_acc = history.history['val_acc']
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
#     x = range(1, len(acc) + 1)
#
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(x, acc, 'b', label='Training acc')
#     plt.plot(x, val_acc, 'r', label='Validation acc')
#     plt.title('Training and validation accuracy')
#     plt.legend()
#     plt.subplot(1, 2, 2)
#     plt.plot(x, loss, 'b', label='Training loss')
#     plt.plot(x, val_loss, 'r', label='Validation loss')
#     plt.title('Training and validation loss')
#     plt.legend()
#     plt.show()
#
#
#
# plot_history(history)
