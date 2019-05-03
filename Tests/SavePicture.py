#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 13:23:42 2019

@author: maurusmanatschal
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:06:43 2019

@author: David
"""
import numpy as np
import cv2
import random


if __name__ == "__main__":
    point = (150, 150)
    cap = cv2.VideoCapture(0)
   
    ret, frame = cap.read()
    if ret == True: 

        cv2.imwrite("{}.jpg".format(str(random.random())[4:8]), frame);
        
                
            
        
        