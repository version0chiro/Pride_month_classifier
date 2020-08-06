from __future__ import print_function
import os
import zipfile
import plaidml.keras
plaidml.keras.install_backend()
import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from scipy.special import expit
from keras.models import Sequential
from keras import layers
from PIL import Image
import pytesseract
import cv2
import os
import pickle
file_path= "data/DataFiles/Dataset/"


import os


 #2.To rename files
file_list=os.listdir(file_path)
df = pd.read_csv("data/DataFiles/sentiment140-subset.csv", nrows=30000)



sentences = df['text'].values

y = df['polarity'].values

sentences_train, sentences_text, y_train, y_test = train_test_split(sentences,y,test_size=0.25,random_state=1000)

vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)
testing=["Find out who you are and be that person. That's what your soul Neie Niee RU toy to be. Find that truth. live that truth, and everything else","I apologize for anything negative I've said towards gays and for that matter anyone.","Bitch get lost","That's so gay","Retarded"]

X_train = vectorizer.transform(sentences_train)
X_test = vectorizer.transform(sentences_text)
testingX = vectorizer.transform(testing)
# tp_test = vectorizer.transform(sentences1)

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

new_model = keras.models.load_model('SavedModel/3rdModel')

print(new_model.predict(testingX))

retrive_path="data/DataFiles/below_1000/"
retrive_list = os.listdir(retrive_path)
text_file = open("data/DataFiles/predictions.txt","w")

with open('textString3.pkl','rb') as handle:
    textFull=pickle.load(handle)

pridictions=[]
textX = vectorizer.transform(textFull)
# print(textFull[2:100])
for _ in range(len(textFull)):
    if new_model.predict(textX[_])>0.70:
        print("Positive")
        print(new_model.predict(textX[_]))
        print(textFull[_])
        pridictions.append("Positive\n")

    elif new_model.predict(textX[_])<0.30:
        print("negative")
        print(new_model.predict(textX[_]))
        print(textFull[_])
        pridictions.append("Negative\n")

    else:
        print("neutral")
        print(new_model.predict(textX[_]))
        print(textFull[_])
        pridictions.append("Random\n")


text_file = open("data/DataFiles/predictionSet.txt","w")
for _ in pridictions:
    text_file.write(_)
text_file.close()
# textFull=[]
# for _ in range(len(retrive_list)):
#     text = pytesseract.image_to_string(Image.open(retrive_path+retrive_list[_]))
#     textFull.append(text)

# with open('textString.pkl','wb') as output:
#     pickle.dump(textFull,output)
# print(sentences_text[62:70])
# for _ in X_test[62:70]:
#     print(new_model.predict(_))
#     if(new_model.predict(_)>.60):
#         print("positive")
#     elif((new_model.predict(_)<.60) & (new_model.predict(_)>.20)):
#         print("neutral")
#     else:
#         print("negative")

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
#                     epochs=20,
#                     validation_data=(X_test,y_test),
#                     batch_size=10)
#
# loss,accuracy = model.evaluate(X_train,y_train,verbose=False)
# print("Training Accuracy : {:.4f}".format(accuracy))
# loss,accuracy = model.evaluate(X_test,y_test,verbose=False)
# print("Testing Accuracy: {:.4f}".format(accuracy))
#
# model.save('SavedModel/2ndModel')

