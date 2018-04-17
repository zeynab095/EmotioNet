# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 08:08:59 2018

@author: zeynab
"""

from scipy.ndimage import zoom
from keras.models import model_from_json
import cv2
import numpy as np


face_cascade = cv2.CascadeClassifier('C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')



cap = cv2.VideoCapture(0)

json_file = open('models/model_conv_gender.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("models/model_conv_gender.h5")
print("Loaded model from disk")

loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

while(True):
    count=0
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        if w > 200:
         count=+1   
         cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
         (x, y, w, h) = faces[0]
         horizontal_offset = int(0.15 * w)
         vertical_offset = int(0.15 * h)
         extracted_face = gray[y+vertical_offset:y+h, x+horizontal_offset:x-horizontal_offset+w]
         new_extracted_face = zoom(extracted_face, (64. / extracted_face.shape[0], 64. / extracted_face.shape[1]))
         new_extracted_face = new_extracted_face.astype(float)
         new_extracted_face /= float(new_extracted_face.max())
         #predict=svc_1.predict([new_extracted_face.ravel()])

    '''
    for (x,y,w,h) in faces:
        X1=np.zeros((1,32,32))
        if w > 200:
         count=+1   
         cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
         (x, y, w, h) = faces[0]
         gray = gray[y:y+h, x:x+w]
    #extracted_face =gray[y+vertical_offset:y+h, x+horizontal_offset:x-horizontal_offset+w]
         zoomed_face=zoom(gray, (32. / gray.shape[0], 32. / gray.shape[1]))
         X1[0]=zoomed_face
         X1 = X1.reshape(1, 32, 32,1).astype('float32')
         X1/=255
         predict=loaded_model.predict(X1, verbose=1).round()
         
         if predict[0][0] == 1:
                cv2.putText(frame, "female",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, 55, 4)
         else:
                cv2.putText(frame, "male",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, 55, 4)
      '''

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
    
