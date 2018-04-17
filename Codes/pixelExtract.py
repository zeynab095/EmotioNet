# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 10:25:55 2018

@author: zeynab
"""

import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import pandas as pd
import cv2
import csv

#svc_1=svc_1
'''


def isSmiling(frame):
    
    face_cascade = cv2.CascadeClassifier('C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detect_face = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
    
    (x, y, w, h) = detect_face[0]
    print("------------------------------------------------------")
    
    gray = gray[y:y+h, x:x+w]
    #print(detect_face.shape)
    print("gray", gray.shape)
    out = cv2.resize(gray, (350, 350))
    print("shape",out.shape)
    horizontal_offset = int(0.15 * w)
    vertical_offset = int(0.15 * h)
    extracted_face =gray[y+vertical_offset:y+h, x+horizontal_offset:x-horizontal_offset+w]
    zoomed_face=zoom(extracted_face, (64. / extracted_face.shape[0], 
                                           64. / extracted_face.shape[1]))
    print("extract",extracted_face.shape)
    print("zoomed",zoomed_face.shape)
    print("zoomedre",zoomed_face.reshape(-1).shape)
    #smileInt=svc_1.predict([zoomed_face.reshape(-1)])
    
    plt.subplot(122)
    plt.imshow(frame, cmap='gray')
    return 0

'''
def isSmiling1(frames):
    
    print("aaa")
    
   
       
    
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
    zoomed_face=zoom(gray, (64. / gray.shape[0], 
                                           64. / gray.shape[1]))
    #zoomed_face=zoom(extracted_face, (extracted_face.shape[0], extracted_face.shape[1]))
    zoom1=zoomed_face.reshape(-1)
    print('aaaa')
    
    print(zoom1.shape)
    '''
    smileInt=svc_1.predict([zoomed_face.reshape(-1)])
    
    sub=100+l*10+k
    '''
    
    #plt.imshow(gray, cmap='gray')
    plt.imshow(zoomed_face, cmap='gray')
    '''
    if(smileInt==1):print('Smiling')
    else: print('Not Smiling')
    k=k+1
    '''
    return zoom1

la=[]
print("a")
data=pd.read_excel("happy_neutral_path.xlsx")
data_img=data.path
size=data_img.size

for i in range(size):
    print(data_img.iloc[i])
    la.append(isSmiling1(cv2.imread(data_img.iloc[i])))
    
with open("happy_neutral_pixels.csv", "w") as f:
    csv_writer = csv.writer(f)
    for i in la:
        
        
        csv_writer.writerow(i)
    
    


