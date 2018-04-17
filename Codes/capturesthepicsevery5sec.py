#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
from keras.models import model_from_json
from angry_pixel import add_data, pixel_extract
import numpy as np
import cv2

c_1 =0
c_2 =0
c_3 =0
c_4 =0
c_5 =0
c_6 =0

emotion_json="C:/Users/user/Desktop/ubuntu/models/model_emotion1.json"
emotion_h5="C:/Users/user/Desktop/ubuntu/models/model_emotion1.h5"
gender_json="C:/Users/user/Desktop/ubuntu/models/modelg.json"
gender_h5="C:/Users/user/Desktop/ubuntu/models/modelg.h5"

dim=128

##---------------Emotion--------------------
json_file = open(emotion_json, 'r')
loaded_model_json = json_file.read()
json_file.close()
model_emotion = model_from_json(loaded_model_json)
model_emotion.load_weights(emotion_h5)
model_emotion.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


##---------------Gender--------------------
json_file = open(gender_json, 'r')
loaded_model_json = json_file.read()
json_file.close()
model_gender = model_from_json(loaded_model_json)
model_gender.load_weights(gender_h5)
model_gender.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


print(cv2.__version__)
vidcap = cv2.VideoCapture('video.avi')
success,image = vidcap.read()
count = 0
success = True
while success:
  vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*5000))    # added this line 
  cv2.imwrite("video_frames/framenew%d.jpg" % count, image)
  X1=np.zeros((1,dim,dim))
  X1[0]=pixel_extract(cv2.imread("video_frames/framenew%d.jpg" % count))
  X1 = X1.reshape(1, dim, dim,1).astype('float32')
  X1/=255
  predicted_emotion=model_emotion.predict(X1, verbose=1).round() 
  if predicted_emotion[0][0]==1: c_1 += 1
  elif predicted_emotion[0][1]==1: c_2 += 1
  elif predicted_emotion[0][2]==1: c_3 += 1
  elif predicted_emotion[0][3]==1: c_4 += 1
  elif predicted_emotion[0][4]==1: c_5 += 1
  elif predicted_emotion[0][5]==1: c_6 += 1
  print(predicted_emotion) 
  print("framenew%d.jpg" % count)   # save frame as JPEG file
  success,image = vidcap.read()
  print ('Read a new frame: ', success)
  count += 1
'''
X2=np.zeros((1,dim,dim))   
X2[0]=pixel_extract(cv2.imread("video_frames/framenew2.jpg"))  
X2 = X2.reshape(1, dim, dim,1).astype('float32')
X2/=255
predicted_gender=model_gender.predict(X2, verbose=1).round() 
print("gender",predicted_gender)
'''

print("-----------------------------------")
print("happy: ",c_1/count)
print("sad: ",c_2/count)
print("angry",c_3/count) 
print("surprise: ",c_4/count)
print("disgust: ",c_5/count)
print("neutral",c_6/count) 


# When everything done, release the capture
vidcap.release()
cv2.destroyAllWindows()