# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 02:20:07 2018

@author: zeynab
"""
import os.path
import sys
from PIL import Image
from angry_pixel import add_data, pixel_extract
import cv2
from keras.models import model_from_json
import numpy as np

state=0
value=0

def value_check(value):
    
    if(value=='quit'): 
        print("You quitted!")  
        sys.exit()
    else: 
        print("Your input: ", value)
        
    return value    

def image_check(path):
    try:
        Image.open(path)
    except IOError:
        print("Please enter image path")
        return False
    return True

def capture_image():
    img_name=""
    cam = cv2.VideoCapture(0)
    
    cv2.namedWindow("test")
    
    img_counter = 0
    
    while True:
        ret, frame = cam.read()
        cv2.imshow("test", frame)
        if not ret:
            break
        k = cv2.waitKey(1)
    
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            
            img_name = "image.png"
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
            break
    
    cam.release()
    
    cv2.destroyAllWindows()
    return img_name



while True:
    print("To quit write 'quit'")
    if state==0:
        print("Welcom\n")
        print("1. Find emotion in image")
        print("2. Find emotion in image")
        print("3. Find emotion in image")
        print("4. Find emotion in image")
        print("To use please write corresponding number")
        value=value_check(input())
        if int(value)==1: state=1
        elif int(value)==2: state=2
        elif int(value)==3: state=3
        elif int(value)==4: state=4

    elif state==1:
        
        print("5. Choose image")
        print("6. Take a picture")
        value=value_check(input())
        if int(value)==5: state=5
        elif int(value)==6: state=6
        
        
        
    elif state==2:
        print("HI 2")
        break
    
    elif state==3:
        print("HI 3")
        break
    elif state==4:
        print("HI 4") 
        break
        
    elif state==5:
        print("Write image source: ")
        value=value_check(input())
        if image_check(value):
            
            json_file = open('model_conv.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights("model_conv.h5")
            print("Loaded model from disk")
            
            loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
            X1=np.zeros((1,32,32))
            X1[0]=pixel_extract(cv2.imread(value))
            X1 = X1.reshape(1, 32, 32,1).astype('float32')
            X1/=255
            print(loaded_model.predict(X1, verbose=1).round())
            
            
            print("True")
            break
        
    
    

print("You quitted!")    
    


