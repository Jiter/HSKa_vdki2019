# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:06:43 2019

@author: David
"""
import numpy as np
import cv2
import random as rng
import glob

# definiere Farb-Ranges
lower_black = (0, 0, 40)
upper_black = (180, 255, 255)

use_tilted_rect = True

point = (150, 150)

threshold = 70

#do_live = False # Schalter zwischen LiveKamera und Übungsbildern
do_live = True
  

def detect(frame):    

    font = cv2.FONT_HERSHEY_SIMPLEX
        

    frame = cv2.blur(frame,(3,3))
    
    # Nutze Canny Filter zum detektieren von Kanten
    edges = cv2.Canny(frame, 100, 255)

    # finde Konturen
    _, contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    hull_list = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        hull_list.append(hull)
        
    cont = np.vstack(contours[i] for i in range(len(contours)))
    hull_all = []
    
    hull_all.append(cv2.convexHull(cont))
    
    # Draw contours + hull results
    #frame = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv2.drawContours(frame, hull_list, i, color)
            
    cv2.imshow('file', edges) #Zeigt die Maske als Bild.

    if len(contours) > 0:
        objekt = max(hull_all, key=cv2.contourArea)
 
        if use_tilted_rect:
            rect = cv2.minAreaRect(objekt)
            box = cv2.boxPoints(rect)
           
            box = np.intp(box)
            
            _, (w,h), _ = rect;

            cv2.drawContours(frame,[box],0,(0,0,255),2)

            cv2.putText(frame,"Width: {:f}, Height: {:f}".format(w, h),(1,100), font, 0.5,(int(155),int(155),int(155)),2,cv2.LINE_AA)

        else:    
            # zeichne die Bounding box des Tennisballs in das Video-Bild ein:
            x, y, w, h = cv2.boundingRect(objekt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), thickness=3)
            cv2.putText(frame,"Width: {}, Height: {}".format(w, h),(1,100), font, 0.5,(int(155),int(155),int(155)),2,cv2.LINE_AA)

    return frame, edges


if __name__ == "__main__":
    
    cnt = 0
    
    
    if do_live:
        cap = cv2.VideoCapture(1); #Initialisiere die Kamera
    else:
        fnames = glob.glob("_Data/*.jpg")
    
    while True: #Schleife 
        
        if do_live:
            ret, frame = cap.read()
        else:
            ret = True
            print(fnames[cnt])
            frame = cv2.imread(fnames[cnt])
            
        
        if ret == True: # Falls gültiges Bild gelesen
            frame, edges  = detect(frame)
            
            cv2.imshow(str("orig"), frame)
            
            #cv2.imwrite("C:/Users/David/Documents/GitHub/HSKa_vdki2019/_Data/Puit/{}.jpg", frame);
            #cv2.imwrite("C:/Users/David/Documents/GitHub/HSKa_vdki2019/_Data/Puit/{}_canny.jpg".format(fnames[cnt][6:10]), edges);
            
            if cv2.waitKey(20) & 0xFF == ord("q"):
                break
            
            cnt = cnt + 1
            if do_live == False and cnt >= len(fnames):
                break

            
            
        else: # Falls Bild ungültig, Kamera nicht bereit oÄ
            print("shit")
            break