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
import pdb

use_tilted_rect = True

#do_live = False  # Schalter zwischen LiveKamera und Übungsbildern
do_live = True


def detect(frame):

    w = 0
    h = 0
    color = (rng.randint(0, 256),
             rng.randint(0, 256),
             rng.randint(0, 256))
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
        
        

        print('Color: {:d},{:d},{:d}'.format(b, g, r))
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

                print('Width, Heigth: {:f},{:f}'.format(w, h))

            else:
                # Zeichne das BoundingRect des Objekts in das Video-Bild ein:
                x, y, w, h = cv2.boundingRect(objekt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255),
                              thickness=3)
                cv2.putText(frame, "Width: {}, \n Height: {}".format(w, h),
                            (10, 20), font, 0.5, (int(155), int(155),
                            int(155)), 2, cv2.LINE_AA)

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


def loadProbabilityArrays(w, h, b, g, r,):
    prob = []

    # WIDTH
    #       Klasse, Schmetterling, Hase, Schaf, Küken
    prob.append(np.array([
            [0,     0,      0.591,  0,      0],      # 210 - 215
            [1,     0,      0.159,  0,      0.026],  # 215 - 220
            [2,     0,      0.159,  0,      0.299],  # 220 - 225
            [3,     0.240,  0,      0.106,  0.597],  # 225 - 230
            [4,     0.2,    0,      0.227,  0.078],  # 230 - 235
            [5,     0.053,  0,      0.197,  0],      # 235 - 240
            [6,     0,      0,      0.469,  0],      # 240 - 245
            [7,     0.253,  0,      0,      0],      # 245 - 250
            [8,     0.16,   0,      0,      0],      # 250 - 255
            ]))

    # HEIGHT
    #       Klasse, Schmetterling, Hase, Schaf, Küken
    prob.append(np.array([
            [0,     0,      0.16,   0,      0],      # 290 - 295
            [1,     0,      0.595,  0,      0],      # 295 - 300
            [2,     0.091,  0.238,  0,      0],      # 300 - 305
            [3,     0.233,  0,      0,      0],      # 305 - 310
            [4,     0.195,  0,      0,      0],      # 310 - 315
            [5,     0.078,  0,      0,      0],      # 315 - 320
            [6,     0,      0,      0,      0],      # 320 - 325
            [7,     0.247,  0,      0,      0],      # 325 - 330
            [8,     0.156,  0,      0.079,  0],      # 330 - 335
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
            ]))

    # BLAU
    #       Klasse, Schmetterling, Hase, Schaf, Küken
    prob.append(np.array([
            [0,     0,      0,      0,      0.013],      # 75  - 79
            [1,     0,      0,      0,      0.065],      # 80  - 84
            [2,     0.078,  0,      0,      0.104],      # 85  - 89
            [3,     0.208,  0,      0,      0.143],      # 90  - 94
            [4,     0.312,  0,      0,      0.104],      # 95  - 99
            [5,     0.065,  0,      0,      0.039],      # 100 - 104
            [6,     0.117,  0.045,  0,      0.117],      # 105 - 109
            [7,     0.091,  0.159,  0,      0.130],      # 110 - 114
            [8,     0,      0.318,  0,      0.195],      # 115 - 119
            [9,     0.026,  0.386,  0,      0.078],      # 120 - 124
            [10,    0.065,  0.091,  0,      0.013],      # 125 - 129
            [11,    0.039,  0,      0,      0],          # 130 - 134
            [12,    0,      0,      0.288,  0],          # 135 - 139
            [13,    0,      0,      0.379,  0],          # 140 - 144
            [14,    0,      0,      0.166,  0],          # 145 - 149
            [15,    0,      0,      0.166,  0],          # 150 - 154
            ]))

    # GRÜN
    #       Klasse, Schmetterling, Hase, Schaf, Küken
    prob.append(np.array([
            [0,     0,      0,      0,      0],      # 75  - 79
            [1,     0.208,  0,      0,      0],      # 80  - 84
            [2,     0.078,  0,      0,      0],      # 85  - 89
            [3,     0.286,  0,      0,      0],      # 90  - 94
            [4,     0.065,  0,      0,      0],      # 95  - 99
            [5,     0,      0,      0,      0],      # 100 - 104
            [6,     0,      0,      0,      0],      # 105 - 109
            [7,     0.065,  0,      0,      0.273],  # 110 - 114
            [8,     0.078,   0,      0,      0.156],  # 115 - 119
            [9,     0.117,  0.182,  0,      0.026],  # 120 - 124
            [10,    0.026,  0.818,  0,      0.182],  # 125 - 129
            [11,    0,      0,      0,      0.234],  # 130 - 134
            [12,    0,      0,      0,      0.130],  # 135 - 139
            [13,    0,      0,      0,      0],      # 140 - 144
            [14,    0,      0,      0.333,  0],      # 145 - 149
            [15,    0,      0,      0.257,  0],      # 150 - 154
            [16,    0,      0,      0.318,  0],      # 155 - 159
            [17,    0,      0,      0.091,  0],      # 160 - 164
            [18,    0.143,  0,      0,      0],      # 165 - 169
            ]))

    # RED
    #       Klasse, Schmetterling, Hase, Schaf, Küken
    prob.append(np.array([
            [0,     0,      0,      0,      0],      # 75  - 79
            [1,     0,      0,      0,      0],      # 80  - 84
            [2,     0,      0,      0,      0],      # 85  - 89
            [3,     0,      0,      0,      0],      # 90  - 94
            [4,     0,      0,      0,      0.013],  # 95  - 99
            [5,     0,      0,      0,      0.078],  # 100 - 104
            [6,     0,      0,      0,      0.182],  # 105 - 109
            [7,     0.091,  0,      0,      0.195],  # 110 - 114
            [8,     0.052,  0,      0,      0.078],  # 115 - 119
            [9,     0.506,  0.182,  0,      0.130],  # 120 - 124
            [10,    0.156,  0.818,  0,      0.221],  # 125 - 129
            [11,    0.052,  0.523,  0,      0.104],  # 130 - 134
            [12,    0.143,  0.477,  0.030,  0],      # 135 - 139
            [13,    0,      0,      0.136,   0],      # 140 - 144
            [14,    0,      0,      0.166,  0],      # 145 - 149
            [15,    0,      0,      0.257,  0],      # 150 - 154
            [16,    0,      0,      0.409,  0],      # 155 - 159
            ]))

    ret = []

    ret.append(prob[0][width2Class(w), :])
    ret.append(prob[1][height2Class(h), :])
    ret.append(prob[2][blue2Class(b), :])
    ret.append(prob[3][green2Class(g), :])
    ret.append(prob[4][red2Class(r), :])

    return ret


# Function to calculate the given Class from the Width
def width2Class(w):
    c = math.floor(((w - 210) / 5))
    if c > 8:
        c = 8
    elif c < 0:
        c = 0
    return c


# Function to calculate the given Class from the Height
def height2Class(h):
    c = math.floor(((h - 290) / 5))
    if c > 22:
        c = 22
    elif c < 0:
        c = 0
    return c


# Function to calculate the given Class from the Blue
def blue2Class(b):
    c = math.floor(((b - 75) / 5))
    if c > 15:
        c = 15
    elif c < 0:
        c = 0
    return c


# Function to calculate the given Class from the Green
def green2Class(g):
    c = math.floor(((g - 75) / 5))
    if c > 18:
        c = 18
    elif c < 0:
        c = 0
    return c


# Function to calculate the given Class from the Red
def red2Class(r):
    c = math.floor(((r - 75) / 5))
    if c > 16:
        c = 16
    elif c < 0:
        c = 0
    return c


# Calculate the Probability for all Classes on one Height-Class
def calcProb(prob):
    ret = []
    for i in range(4):
        su = ((prob[1] + prob[2] + prob[3] + prob[4]))
        if (su == 0):
            ret.append(0)
        else:
            ret.append((prob[i+1]) / (prob[1] + prob[2] + prob[3] + prob[4]))
            
    return ret

# Give Back an Vector with probabilitys of each Teached Class
def probabilityMatrix(classprob):
    ret = []
    for i in range(1,len(classprob)):
        ret.append(calcProb(classprob[i]))
        
    return ret


# No Really... This is really where the magic happens
def thisIsWhereTheMagicHappens(feat):
    
    klasse = "Unknown"
    
    w = feat[0]
    h = feat[1]
    b = feat[2][0]
    g = feat[2][1]
    r = feat[2][2]
    cl = ["Kueken", "Hase", "Schaf", "Schmetterling"]
    
    classprob = loadProbabilityArrays(h, w, b, g, r)
    prob = probabilityMatrix(classprob)
      
    summe = []
    
    for i in range(len(prob[1])):
        s = 0
        for j in range(len(prob)):
            s = prob[j][i] + s
        summe.append(s)
    
    klasse = cl[summe.index(max(summe))]
    
    return klasse


def rmseClassifier(feat):

    klasse = "Unknown"
    rmse = []

    # Mittelwerte Breite Höhe Farbe
    yK = [393.568, 226.339, 114.51]  # Küken
    yH = [269.751, 215.735, 126.42]  # Hasen
    yS = [339.887, 237.996, 149.33]  # Schafe
    yP = [318.529, 239.186, 111.40]  # Schmetterlinge
    cl = ["Kueken", "Hase", "Schaf", "Schmetterling"]

    n = len(yK)  # Anzahl Merkmale

    w = feat[0]
    h = feat[1]
    c = sum(map(float, filter(None, feat[2][1:])))/(len(feat[2])-1)

    rmse.append(math.sqrt((1 / n) * (pow((yK[0] - w), 2) + pow((yK[1] - h), 2) + 0*pow((yK[2] - c), 2))))
    rmse.append(math.sqrt((1 / n) * (pow((yH[0] - w), 2) + pow((yH[1] - h), 2) + 0*pow((yH[2] - c), 2))))
    rmse.append(math.sqrt((1 / n) * (pow((yS[0] - w), 2) + pow((yS[1] - h), 2) + 0*pow((yS[2] - c), 2))))
    rmse.append(math.sqrt((1 / n) * (pow((yP[0] - w), 2) + pow((yP[1] - h), 2) + 0*pow((yP[2] - c), 2))))

    klasse = cl[rmse.index(min(rmse))]

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
            
            bayesklasse = thisIsWhereTheMagicHappens(feat)
            
            print("RMSE: {}, Bayess: {}".format(rmseklasse, bayesklasse))

            cv2.putText(frame, "RMSE: {}".format(rmseklasse),
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (int(155), int(155), int(155)),
                        2, cv2.LINE_AA)
            
            cv2.putText(frame, "Bayes: {}".format(bayesklasse),
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (int(155), int(155), int(155)),
                        2, cv2.LINE_AA)

            cv2.imshow("Original", frame)

            if (not do_live):
                cv2.imwrite("_Data/Puit/{}.jpg".format(fnames[cnt][6:10]), frame)
#               cv2.imwrite("_Data/Puit/{}_canny.jpg".format(fnames[cnt][6:10]), edges)

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
