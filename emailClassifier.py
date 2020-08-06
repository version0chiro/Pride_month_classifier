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


filepath_dict = {'yelp':   'data/sentiment_analysis/yelp_labelled.txt',
                 'amazon': 'data/sentiment_analysis/amazon_cells_labelled.txt',
                 'imdb':   'data/sentiment_analysis/imdb_labelled.txt'}

df_list = []

for source,filepath in filepath_dict.items():
    df = pd.read_csv(filepath,names=['sentence','label'],sep='\t')
    df['source'] = source #Add another column filld with the source name
    df_list.append(df)

df = pd.concat(df_list)
print(df.iloc[0])

sentences1 = ['John likes ice cream', 'John hates chocolate.']
# vectorizer = CountVectorizer(min_df=0, lowercase=False)
# vectorizer.fit(sentences)
# print(vectorizer.vocabulary_)

df_yelp = df[df['source']=='yelp']

sentences = df_yelp['sentence'].values

y = df_yelp['label'].values

sentences_train, sentences_text, y_train, y_test = train_test_split(sentences,y,test_size=0.25,random_state=1000)

vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)

X_train = vectorizer.transform(sentences_train)
X_test = vectorizer.transform(sentences_text)

tp_test = vectorizer.transform(sentences1)

# classifier = LogisticRegression()
# classifier.fit(X_train,y_train)
# score = classifier.score(X_test,y_test)
# # print(sentences_text[2:10])
# print(classifier.predict(tp_test))
#
# print("Accuracy:",score)
#
# for source in df['source'].unique():
#     df_source = df[df['source']==source]
#     sentences = df_source['sentence'].values
#     y = df_source['label'].values
#
#     sentences_train, sentences_test, y_train, y_test = train_test_split(
#         sentences, y, test_size=0.25, random_state=1000)
#
#     vectorizer = CountVectorizer()
#     vectorizer.fit(sentences_train)
#     X_train = vectorizer.transform(sentences_train)
#     X_test = vectorizer.transform(sentences_test)
#
#     classifier = LogisticRegression()
#     classifier.fit(X_train, y_train)
#     score = classifier.score(X_test, y_test)
#     print('Accuracy for {} data: {:.4f}'.format(source, score))

input_dim = X_train.shape[1] #Number of Features

model = Sequential()
model.add(layers.Dense(10,input_dim=input_dim,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())

history = model.fit(X_train,y_train,
                    epochs=20,
                    validation_data=(X_test,y_test),
                    batch_size=10)

loss,accuracy = model.evaluate(X_train,y_train,verbose=False)
print("Training Accuracy : {:.4f}".format(accuracy))
loss,accuracy = model.evaluate(X_test,y_test,verbose=False)
print("Testing Accuracy: {:.4f}".format(accuracy))

model.save('SavedModel/2ndModel')

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
    plt.show()



plot_history(history)
