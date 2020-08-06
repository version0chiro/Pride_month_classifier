from PIL import Image
import pytesseract
import cv2
import os

file_path= "data/DataFiles/Dataset/"

import os


 #2.To rename files
file_list=os.listdir(file_path)

for _ in range(len(file_list)):
    image = cv2.imread(file_path+file_list[_])
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray,0,255,
                         cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
    # gray = cv2.medianBlur(gray,3)
    path = file_list[_].replace(".jpg","")
    filename="data/DataFiles/sac/"+path+".png"
    cv2.imwrite(filename,gray)

# retrive_path="data/DataFiles/sac/"
# retrive_list = os.listdir(retrive_path)
# text_file = open("data/DataFiles/textSet.txt","w")
# textFull=[]
# for _ in range(len(retrive_list)):
#     text = pytesseract.image_to_string(Image.open(retrive_path+retrive_list[_]))
#     print(text)
#     textFull.append(text)
#
# for _ in textFull:
#     text_file.write(_)
#     text_file.write("\n")
# text_file.close()