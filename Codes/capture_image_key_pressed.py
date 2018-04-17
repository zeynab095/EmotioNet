# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 03:45:08 2018

@author: zeynab
"""

import cv2
from scipy.ndimage import zoom

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('C://opencv/build/etc/haarcascades/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('C://opencv/build/etc/haarcascades/haarcascade_eye.xml')
img_counter = 0
frame_counter = 0

while(True):
    # Capture frame-by-frame
   
    ret, frame = cap.read()
    cv2.imshow("test", frame)
    
    frame_counter += 1
    
    k = cv2.waitKey(1)
    if k%256 == 27:
        print("{} written!")
            
        cap.release()
        cv2.destroyAllWindows()
    
             
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()