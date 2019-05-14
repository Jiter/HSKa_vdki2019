# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:06:43 2019

@author: David
"""
import numpy as np
import cv2
import random as rng
import glob

use_tilted_rect = True

# do_live = False # Schalter zwischen LiveKamera und Übungsbildern
do_live = True


def detect(frame):

    font = cv2.FONT_HERSHEY_SIMPLEX

    frame = cv2.blur(frame, (3, 3))

    # Nutze Canny Filter zum detektieren von Kanten
    edges = cv2.Canny(frame, 100, 255)

    # finde Konturen
    _, contours, _ = cv2.findContours(edges, cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)
    hull_list = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])  # konvexe Hülle aller Konturen
        hull_list.append(hull)

    # if ():
    cont = np.vstack(contours[i] for i in range(len(contours)))
    hull_all = []

    # print("contours:{}".format(contours))
    if (not contours == []):
        cont = np.vstack(contours[i] for i in range(len(contours)))
        hull_all = []

        hull_all.append(cv2.convexHull(cont))
        
        mask = edges
        
        cv2.drawContours(mask, hull_all, -1, (255, 255, 255), -1)
        # cv2.imshow("Mask", mask)
        
        b, g, r, _ = np.uint8(cv2.mean(frame, mask))
        
        cv2.putText(frame, "Color: {},{},{}".format(b, g, r),
            (10, 40), font, 0.5,
            (int(b), int(g), int(r)),
            2, cv2.LINE_AA)
        
        # Draw contours + hull results
        # frame = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
        for i in range(len(contours)):
            color = (rng.randint(0, 256),
                     rng.randint(0, 256),
                     rng.randint(0, 256))

        print('{:d},{:d},{:d}'.format(b, g, r))
            #cv2.drawContours(frame, hull_list, i, color)

        cv2.imshow('Canny', edges)  # Zeigt die Maske als Bild.

        if len(contours) > 0:
            objekt = max(hull_all, key=cv2.contourArea)
            
            
            if use_tilted_rect:
                rect = cv2.minAreaRect(objekt)
                box = cv2.boxPoints(rect)

                box = np.intp(box)

                _, (a, b), _ = rect

                if a > b:
                    h = b
                    w = a
                else:
                    h = a
                    w = b

                cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)

                cv2.putText(frame, "Width: {:f}, Height: {:f}".format(w, h),
                            (10, 20), font, 0.5,
                            (int(155), int(155), int(155)),
                            2, cv2.LINE_AA)

                # print('{:f},{:f}'.format(w, h))

            else:
                # Zeichne das BoundingRect des Objekts in das Video-Bild ein:
                x, y, w, h = cv2.boundingRect(objekt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255),
                              thickness=3)
                cv2.putText(frame, "Width: {}, \n Height: {}".format(w, h),
                            (10, 20), font, 0.5, (int(155), int(155),
                            int(155)), 2, cv2.LINE_AA)

    return frame, edges


def makeWindows():
    cv2.namedWindow("Canny", cv2.WINDOW_NORMAL)
    cv2.waitKey(10)
    cv2.moveWindow("Canny", 500, 0)
    cv2.waitKey(10)
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.waitKey(10)
    cv2.moveWindow("Original", 0, 0)
    cv2.waitKey(10)


if __name__ == "__main__":

    cnt = 0

#    for i in range(1,1499) :
#        cap = cv2.VideoCapture(i)
#        if (cap.isOpened()):
#            print("Found camera {}\n".format(i))
#            break;
 
    makeWindows()

    if do_live:
        cap = cv2.VideoCapture(0)  # Initialisiere die Kamera
    else:
        fnames = glob.glob("_Data/*.jpg")

    while True:  # Schleife

        if do_live:
            ret, frame = cap.read()
        else:
            ret = True
            print(fnames[cnt])
            frame = cv2.imread(fnames[cnt])

        if ret:  # Falls gültiges Bild gelesen
            frame, edges = detect(frame)

            cv2.imshow("Original", frame)

#            cv2.imwrite("_Data/Puit/{}.jpg", frame)
#            cv2.imwrite("_Data/Puit/{}_canny.jpg".format(fnames[cnt][6:10]), mask)

            if (cv2.waitKey(20) & 0xFF) == ord("q"):
                break

            if (not do_live) and cnt >= len(fnames):
                break
            
            cnt = cnt + 1
            if cnt > 11:
                break

        else:  # Falls Bild ungültig, Kamera nicht bereit oÄ
            print("Could not retrieve any Picture... Sad...")
            break

    # Release Handle on CAP and destroy all Windows
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(10)
    print("TEST")
