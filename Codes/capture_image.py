# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 03:38:14 2018

@author: zeynab
"""

import cv2


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

print(capture_image())