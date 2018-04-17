# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 11:47:59 2018

@author: zeynab
"""

# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import pandas as pd
import numpy as np
import cv2



def pixel_extract(frames):
    
    face_cascade = cv2.CascadeClassifier('C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    
    detect_face = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
    
   
    
    (x, y, w, h) = detect_face[0]
    
    
    #horizontal_offset = int(0.15 * w)
    #vertical_offset = int(0.15 * h)
    gray = gray[y:y+h, x:x+w]
    #extracted_face =gray[y+vertical_offset:y+h, x+horizontal_offset:x-horizontal_offset+w]
    zoomed_face=zoom(gray, (128. / gray.shape[0], 
                                           128. / gray.shape[1]))
    #change to zoomed_face
    return zoomed_face

def add_data(X, y, path, file, a, b, em_num, im1):
    #[a,b)
    data = pd.read_excel(path+file)
    img = data.path
    for i in range(a,b):
        #print(img.iloc[i-a+im1])
        X[i] = pixel_extract(cv2.imread(path + img.iloc[i-a+im1]))
        y[i] = em_num
        #print(img.iloc[i])
    return X, y



'''

dim=128


path_f, file_f="C:/Users/user/Desktop/ubuntu/female/","female_path.xlsx"

path_m, file_m="C:/Users/user/Desktop/ubuntu/male/","male_path.xlsx"


X_train=np.empty((4,dim,dim))
y_train=np.empty(4)

X_test=np.empty((2,dim,dim))
y_test=np.empty(2)


add_data(X_train, y_train, path_f, file_f, 0, 3, 0,0)
print("-----------------------------------------")
add_data(X_train, y_train, path_m, file_m, 3, 6, 1,0)
print("-----------------------------------------")
add_data(X_test, y_test, path_f, file_f, 0, 2, 0, 3)
print("-----------------------------------------")
add_data(X_test, y_test, path_m, file_m, 2, 4, 1, 3)






path_f, file_f="C:/Users/user/Desktop/ubuntu/female/","female_path.xlsx"

path_m, file_m="C:/Users/user/Desktop/ubuntu/male/","male_path.xlsx"


X_train=np.empty((2000,32,32))
y_train=np.empty(2000)

X_test=np.empty((500,32,32))
y_test=np.empty(500)

#add_data(X_train, y_train, path_f, file_f, 0, 100, 0, 0)

#path_h, file_h="C:/Users/user/Desktop/ubuntu/jaffe/","jaffe.xlsx"

data = pd.read_excel(path_m+file_m)
img = data.path
for i in range(0,3010):
    print(img.iloc[i])
    pixel_extract(cv2.imread(path_m + img.iloc[i]))

#add_data(X_train, y_train, path_m, file_m, 1000, 2000, 1, 1000)
#pixel_extract(cv2.imread("F:/1_raw/00abe0e1583750059de100f709ea06bb18508ac21b5e2d42ec488426.jpg"))

X=np.zeros((10, 32, 32))
y=np.zeros(10)
add_data(X, y, "C:/Users/user/Desktop/sdp/AngryData_new/angry/", "angry_angry_path.xlsx", 2, 5, 2)
print(X)
print(y)

####TRAIN#####################################################################
X_train=np.empty((1200,32,32))

#angry
angry_path="C:/Users/user/Desktop/sdp/AngryData_new/angry/"
data_angry=pd.read_excel(angry_path+"angry_angry_path.xlsx")
img_angry=data_angry.path

#print("------------------------------------------------")
print("angry_test",angry_path+img_angry.iloc[399])

for i in range(800,1200):
    X_train[i]=isSmiling1(cv2.imread(angry_path+img_angry.iloc[i-800]))
    #print(i)
    #print("----------------------------------------------------")
    #print(i-800)

----------------------------------------------------
397
1198
----------------------------------------------------
398
1199
----------------------------------------------------
399


#print(X_train[799])
#print("----------------------------------------------------")
#print(X_train[800])    
#print("----------------------------------------------------")    
#print(X_train[1199])
print("----------------------------------------------------")
#print(X_train[1200])    
#print("----------------------------------------------------")     
 
####TEST#####################################################################
X_test=np.empty((300,32,32))

#angry
#print("------------------------------------------------")
print("angry_test",angry_path+img_angry.iloc[400])

for i in range(200,230):
    X_train[i]=pixel_extract(cv2.imread(angry_path+img_angry.iloc[i+400-200]))
    #print(i)
    #print("----------------------------------------------------")
    #print(i+400-200)    

'''