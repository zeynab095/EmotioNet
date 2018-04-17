# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 09:58:54 2018

@author: zeynab
"""

import cv2
import dlib
import imutils
import numpy as np
import math
import csv
import pandas as pd

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")



def get_landmarks(image):
    image = cv2.imread(image)
    image = imutils.resize(image, width=235)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections = detector(gray, 1)
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
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

print(get_landmarks("face2.png"))   


    

