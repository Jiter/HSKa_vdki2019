# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 21:13:02 2019

@author: David
"""
import numpy as np
import cv2

# definiere Region of Interest
x, y, w, h = 200, 180, 200, 150
#x, y, w, h = 300, 230, 200, 150
# definiere Farb-Ranges
lower_yellow = (0, 0, 40)
upper_yellow = (180, 255, 255)


if __name__ == "__main__":
    while True:        
        # lese Bild von Festplatte
        image = cv2.imread("../_Data/2.jpg")
        # konvertiere Frame in HSV-Farbraum, um besser nach Farb-Ranges filtern zu können
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(image, lower_yellow, upper_yellow)
         
        # finde Konturen in der Maske, die nur noch zeigt, wo gelbe Pixel sind:
        _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            tennis_ball = max(contours, key=cv2.contourArea)
     
            # zeichne die Bounding box des Tennisballs in das Video-Bild ein:
            x, y, w, h = cv2.boundingRect(tennis_ball)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), thickness=3)

        # zeichne Rechteck in Bild
        #cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), thickness=2)

        # gebe Hue-Wert an der linken oberen Ecke der ROI aus, um Farbwerte des Tennis balls zu ermitteln:
        cv2.putText(image, "HSV: {0}".format(image[y+10, x+10]), (x, y - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0, 0), thickness=2)    
    
        # zeige das Bild an
        cv2.imshow("Bild modifiziert", image)
         
        # warte auf Tastendruck (wichtig, sonst sieht man das Fenster nicht)
        if cv2.waitKey(20) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            #cam.release()
            break
