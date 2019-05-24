# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:06:43 2019

@author: David
"""
import numpy as np
import cv2
import random as rng
import glob
import math

use_tilted_rect = True

do_live = False  # Schalter zwischen LiveKamera und Übungsbildern
# do_live = True


def detect(frame):

    feat = []  # Feature Array

    font = cv2.FONT_HERSHEY_SIMPLEX

    frame = cv2.blur(frame, (3, 3))

    # Nutze Canny Filter zum detektieren von Kanten
    edges = cv2.Canny(frame, 100, 255)

    cv2.imshow("Canny", edges)

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
        cv2.drawContours(frame, hull_list, i, color)

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

                print('{:f},{:f}'.format(w, h))

            else:
                # Zeichne das BoundingRect des Objekts in das Video-Bild ein:
                x, y, w, h = cv2.boundingRect(objekt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255),
                              thickness=3)
                cv2.putText(frame, "Width: {}, \n Height: {}".format(w, h),
                            (10, 20), font, 0.5, (int(155), int(155),
                            int(155)), 2, cv2.LINE_AA)

    cv2.imshow("Original", frame)

    feat.append(w)
    feat.append(h)
    feat.append(color)

    return frame, edges, feat


def makeWindows():
    cv2.namedWindow("Canny", cv2.WINDOW_NORMAL)
    cv2.waitKey(10)
    cv2.moveWindow("Canny", 500, 0)
    cv2.waitKey(10)
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.waitKey(10)
    cv2.moveWindow("Original", 0, 0)
    cv2.waitKey(10)


def loadProbabilityArrays(h):
    prob = []
    #       Klasse, Schmetterling, Hase, Schaf, Küken
    prob = np.array([
            [0,     0,      0.16,   0,      0],      # 290 - 295
            [1,     0,      0.595,  0,      0],      # 295 - 300
            [2,     0.093,  0.238,  0,      0],      # 300 - 305
            [3,     0.240,  0,      0,      0],      # 305 - 310
            [4,     0.2,    0,      0,      0],      # 310 - 315
            [5,     0.053,  0,      0,      0],      # 315 - 320
            [6,     0,      0,      0,      0],      # 320 - 325
            [7,     0.253,  0,      0,      0],      # 325 - 330
            [8,     0.16,   0,      0.079,  0],      # 330 - 335
            [9,     0,      0,      0.571,  0],      # 335 - 340
            [10,    0,      0,      0.19,   0],      # 340 - 345
            [11,    0,      0,      0.158,  0],      # 345 - 350
            [12,    0,      0,      0,      0],      # 350 - 355
            [13,    0,      0,      0,      0],      # 355 - 360
            [14,    0,      0,      0,      0],      # 360 - 365
            [15,    0,      0,      0,      0],      # 365 - 370
            [16,    0,      0,      0,      0],      # 370 - 375
            [17,    0,      0,      0,      0],      # 375 - 380
            [18,    0,      0,      0,      0.067],  # 380 - 385
            [19,    0,      0,      0,      0.284],  # 385 - 390
            [20,    0,      0,      0,      0.243],  # 390 - 395
            [21,    0,      0,      0,      0.365],  # 395 - 400
            [22,    0,      0,      0,      0.040],  # 400 - 405
            ])

    return prob[height2Class(h), :]


# Function to calculate the given Class from the Height
def height2Class(h):
    c = math.floor(((h - 290) / 5))
    if c > 22:
        c = 22
    elif c < 0:
        c = 0

    return c


# Calculate the Probability for all Classes on one Height-Class
def calcProb(prob, klasse):
    return ((prob[klasse]) / 4) / ((prob[1] + prob[2] + prob[3] + prob[4]) / 4)


# Give Back an Vector with probabilitys of each Teached Class
def probabilityMatrix(classprob):
    v = []
    for i in range(4):
        v.append(calcProb(classprob, i + 1))
    return v


def thisIsWhereTheMagicHappens(h):
    classprob = loadProbabilityArrays(h)
    heightprob = probabilityMatrix(classprob)

    print(heightprob)


def rmseClassifier(feat):

    klasse = "Unknown"
    rmse = []

    # Mittelwerte Breite Höhe Farbe
    yK = [393.568, 226.339, 114.51]  # Küken
    yH = [269.751, 215.735, 126.42]  # Hasen
    yS = [339.887, 237.996, 149.33]  # Schafe
    yP = [318.529, 239.186, 111.40]  # Schmetterlinge
    cl = ["Küken", "Hase", "Schaf", "Schmetterling"]

    n = len(yK)  # Anzahl Merkmale

    w = feat[0]
    h = feat[1]
    c = sum(map(float, filter(None, feat[2][1:])))/(len(feat[2])-1)

    rmse.append(math.sqrt((1 / n) * (pow((yK[0] - w), 2) + pow((yK[1] - h), 2) + pow((yK[2] - c), 2))))
    rmse.append(math.sqrt((1 / n) * (pow((yH[0] - w), 2) + pow((yH[1] - h), 2) + pow((yH[2] - c), 2))))
    rmse.append(math.sqrt((1 / n) * (pow((yS[0] - w), 2) + pow((yS[1] - h), 2) + pow((yS[2] - c), 2))))
    rmse.append(math.sqrt((1 / n) * (pow((yP[0] - w), 2) + pow((yP[1] - h), 2) + pow((yP[2] - c), 2))))

    klasse = cl[rmse.index(min(rmse))]

    print(rmse)

    print(klasse)
    return klasse


if __name__ == "__main__":

    cnt = 0

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
            frame, edges, feat = detect(frame)

            rmseklasse = rmseClassifier(feat)

            cv2.putText(frame, "RMSE: {}".format(rmseklasse),
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (int(155), int(155), int(155)),
                        2, cv2.LINE_AA)

            cv2.imwrite("_Data/Puit/{}.jpg".format(fnames[cnt][6:10]), frame)
#            cv2.imwrite("_Data/Puit/{}_canny.jpg".format(fnames[cnt][6:10]), edges)

            if (cv2.waitKey(20) & 0xFF) == ord("q"):
                break

            cnt = cnt + 1
            if (not do_live) and cnt >= len(fnames):
                break


        else:  # Falls Bild ungültig, Kamera nicht bereit oÄ
            print("Could not retrieve any Picture... Sad...")
            break

    # Release Handle on CAP and destroy all Windows
    if do_live:
        cap.release()

    cv2.destroyAllWindows()
    cv2.waitKey(10)
    print("Bye Bye")
