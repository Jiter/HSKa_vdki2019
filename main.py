# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:06:43 2019

@author: David
"""
import numpy as np
import cv2

if __name__ == "__main__":
    point = (150, 150)
    cap = cv2.VideoCapture(0)
   
    while True:
        ret, frame = cap.read()
        if ret == True: 
            color = frame[150, 150]
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame,'Classifier MTM2019',(1,50), font, 2,(0,255,0),2,cv2.LINE_AA)
            cv2.circle(frame,point,10,(0,255,0))   
            cv2.putText(frame,str(color),(1,100), font, 0.5,(int(color[0]),int(color[1]),int(color[2])),2,cv2.LINE_AA)
                    
            cv2.imshow("frame", frame)
            if cv2.waitKey(20) & 0xFF == ord("q"):
                break
        else:
            print("shit")
            break