import pickle

from PIL import Image
import pytesseract
import cv2
import os



retrive_path="data/DataFiles/below_1000/"
retrive_list = os.listdir(retrive_path)

textFull=[]
for _ in range(len(retrive_list)):
    text = pytesseract.image_to_string(Image.open(retrive_path+retrive_list[_]))
    textFull.append(text)

with open('textString3.pkl','wb') as output:
    pickle.dump(textFull,output)
