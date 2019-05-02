# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:06:43 2019

@author: David
"""
import numpy as np
import cv2

# definiere Farb-Ranges
lower_black = (0, 0, 40)
upper_black = (180, 255, 255)


if __name__ == "__main__":
    point = (150, 150)
    cap = cv2.VideoCapture(0)
   
    while True:
        ret, frame = cap.read()
        if ret == True: 
            #color = frame[150, 150]
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame,'Classifier VdKI',(1,50), font, 2,(0,255,0),2,cv2.LINE_AA)
            #cv2.circle(frame,point,10,(0,255,0))   
            #cv2.putText(frame,str(color),(1,100), font, 0.5,(int(color[0]),int(color[1]),int(color[2])),2,cv2.LINE_AA)
            
            mask = cv2.inRange(frame, lower_black, upper_black)
           
            
            # finde Konturen in der Maske, die nur noch zeigt, wo gelbe Pixel sind:
            _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE)
            
          
            
            cv2.drawContours(frame, contours, -1, (0,255,0))
            cv2.imshow("im2", im2) #Zeigt die Maske als Bild.


            if len(contours) > 0:
                objekt = max(contours, key=cv2.contourArea)
         
                # zeichne die Bounding box des Tennisballs in das Video-Bild ein:
                x, y, w, h = cv2.boundingRect(objekt)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), thickness=3)
                cv2.putText(frame,"Width: {}, Height: {}".format(w, h),(1,100), font, 0.5,(int(155),int(155),int(155)),2,cv2.LINE_AA)
                
                #rect = cv2.minAreaRect(cnt)
                #box = cv2.boxPoints(rect)
                #box = np.int0(box)
                #cv2.drawContours(frame,[box],0,(0,255,255),2)

            cv2.imshow("frame", frame)
            
            if cv2.waitKey(20) & 0xFF == ord("q"):
                break
            
            
            
        else:
            print("shit")
            break