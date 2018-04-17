# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 13:40:23 2018

@author: zeynab
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 09:40:58 2018

@author: zeynab
"""

import matplotlib.pyplot as plt
from scipy.ndimage import zoom

import cv2
import numpy as np
import cv2
import dlib
import imutils
import numpy as np
import math
import csv
import pandas as pd
import os;

#svc_1=svc_1
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
curdir = os.path.abspath(os.path.dirname(__file__))

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
    #h=int(h*1.15);
    #w=int(w*1.15);
    gray = gray[y:y+h, x:x+w]
    detections = detector(gray, 1)
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(gray, d) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        xcentral=[]
        ycentral=[]
        for i in range(1,68): #Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        xmean = np.mean(xlist) #Find both coordinates of centre of gravity
        ymean = np.mean(ylist)
        xcentral = [(x-xmean) for x in xlist] #Calculate distance centre <-> other points in both axes
        ycentral = [(y-ymean) for y in ylist]    
        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            #landmarks_vectorised.append(w)
            #landmarks_vectorised.append(z)
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, x)*360)/(2*math.pi))

        data= landmarks_vectorised
        #print(data)
    
        
    if len(detections) < 1: 
        print("a")
    return data    

print("a")


la=[]
print(isSmiling(cv2.imread("img/face2.png")))

data=pd.read_excel("happy_happy_path.xlsx")
data_img=data.path

size=data_img.size




for i in range(size):
    print(data_img.iloc[i])
    la.append(isSmiling(cv2.imread(data_img.iloc[i])))
    

with open("happy_happy_landmark_extracted_face.csv", "w") as f:
    
    for i in la:
        
        
        csv_writer = csv.writer(f)
        csv_writer.writerow(i)


