# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 06:36:37 2018

@author: zeynab

"""

from angry_pixel import add_data, pixel_extract
from keras.models import model_from_json
from PIL import Image
import numpy as np
import os.path
import shutil
import cv2
import sys



def predict_emotion(path):
   
            
    json_file = open('models/model_conv_cnn.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("models/model_conv_cnn.h5")
    print("Loaded model from disk")
    
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    X1=np.zeros((1,32,32))
    X1[0]=pixel_extract(cv2.imread(path))
    X1 = X1.reshape(1, 32, 32,1).astype('float32')
    X1/=255
    predicted_gender=loaded_model.predict(X1, verbose=1).round()
    
    return predicted_gender 


print(predict_emotion("img/30127981_2340019522891222_8394871466771873792_o.jpg"))