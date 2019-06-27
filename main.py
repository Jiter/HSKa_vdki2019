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

#do_live = False  # Schalter zwischen LiveKamera und Übungsbildern
do_live = True


def detect(frame):

    w = 0
    h = 0
    bgr = []
    hsv = []
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
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        h, s, v, _ = np.uint8(cv2.mean(hsv, mask))
 
        bgr = (b,g,r)
        hsv = (h,s,v)
        
        cv2.putText(frame, "Mean Color of Object: RGB: {},{},{} HSV: {},{},{}".format(r, g, b, h, s, v),
                    (10, 40), font, 0.5,
                    (int(b), int(g), int(r)),
                    2, cv2.LINE_AA)

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
                            (int(0), int(155), int(0)),
                            2, cv2.LINE_AA)

                print('Width, Heigth: {:f},{:f}'.format(w, h))

            else:
                # Zeichne das BoundingRect des Objekts in das Video-Bild ein:
                x, y, w, h = cv2.boundingRect(objekt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255),
                              thickness=3)
                cv2.putText(frame, "Width: {}, Height: {}".format(w, h),
                            (10, 20), font, 0.5, (int(0), int(155),
                            int(0)), 2, cv2.LINE_AA)

    feat.append(w)
    feat.append(h)
    feat.append(bgr)
    feat.append(hsv)

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


def loadProbabilityArrays(w, h, ch, s):
    prob = []

    # WIDTH
    #       Klasse, Schmetterling, Hase, Schaf, Küken
    prob.append(np.array([
            [0,     0,      0.023,  0,      0],      # 210 - 215
            [1,     0,      0.045,  0,      0],  # 215 - 220
            [2,     0,      0.772,  0,      0.5],  # 220 - 225
            [3,     0,      0.159,  0,      0.469],  # 225 - 230
            [4,     0,      0,      0.561,  0.030],  # 230 - 235
            [5,     0.143,  0,      0.439,  0],      # 235 - 240
            [6,     0.364,  0,      0.,     0],      # 240 - 245
            [7,     0.493,  0,      0,      0],      # 245 - 250
            [8,     0,      0,      0,      0],      # 250 - 255
            ]))

    # HEIGHT
    #       Klasse, Schmetterling, Hase, Schaf, Küken
    prob.append(np.array([
            [0,     0,      0.045,  0,      0],      # 290 - 295
            [1,     0,      0.613,  0,      0],      # 295 - 300
            [2,     0,      0.341,  0,      0],      # 300 - 305
            [3,     0.143,  0,      0,      0],      # 305 - 310
            [4,     0.005,  0,      0,      0],      # 310 - 315
            [5,     0.393,  0,      0,      0],      # 315 - 320
            [6,     0.005,  0,      0.215,  0],      # 320 - 325
            [7,     0.311,  0,      0.154,  0],      # 325 - 330
            [8,     0.143,  0,      0.446,  0],      # 330 - 335
            [9,     0,      0,      0.184,  0],      # 335 - 340
            [10,    0,      0,      0,      0],      # 340 - 345
            [11,    0,      0,      0,      0],      # 345 - 350
            [12,    0,      0,      0,      0],      # 350 - 355
            [13,    0,      0,      0,      0],      # 355 - 360
            [14,    0,      0,      0,      0],      # 360 - 365
            [15,    0,      0,      0,      0],      # 365 - 370
            [16,    0,      0,      0,      0],      # 370 - 375
            [17,    0,      0,      0,      0],      # 375 - 380
            [18,    0,      0,      0,      0.242],  # 380 - 385
            [19,    0,      0,      0,      0.616],  # 385 - 390
            [20,    0,      0,      0,      0.005],  # 390 - 395
            [21,    0,      0,      0,      0.091],  # 395 - 400
            [22,    0,      0,      0,      0.045],  # 400 - 405
            ]))

    # H-Value (alt:BLAU)
    #       Klasse, Schmetterling, Hase, Schaf, Küken
    prob.append(np.array([
            [0,     0,      0.068,  0,      0],      # 24  - 25
            [1,     0.072,  0.773,  0,      0],      # 26  - 27
            [2,     0.127,  0.159,  0,      0],      # 28  - 29
            [4,     0,      0,      0,      0],     # 30 - 31
            [5,     0,      0,      0,      0],     # 32 - 33
            [6,     0,      0,      0,      0],     # 34 - 35
            [7,     0,      0,      0,      0],     # 36 - 37
            [8,     0,      0,      0,      0],     # 38 - 39
            [9,     0,      0,      0,      0],     # 40 - 41
            [10,     0,      0,      0,      0],     # 42 - 43
            [11,     0,      0,      0,      0],     # 44 - 45
            [12,     0,      0,      0,      0],     # 46 - 47
            [3,     0,      0,      0,      0.473],      # 48  - 49
            [4,     0,      0,      0,      0.127],      # 50  - 51
            [5,     0,      0,      0,      0],      # 52 - 53
            [6,     0,      0,      0,      0.4],      # 54 - 55
            [7,     0,      0,      0,      0],      # 56 - 57
            [8,     0,      0,      0.151,  0],      # 58 - 59
            [9,     0.145,  0,      0.515,  0],      # 60 - 61
            [10,    0.054,  0,      0.242,  0],      # 62 - 63
            [11,    0,      0,      0.091,  0],          # 64 - 65
            [12,     0,      0,      0,      0],     # 66 - 67
            [13,     0,      0,      0,      0],     # 68 - 69
            [14,     0,      0,      0,      0],     # 70 - 71
            [15,     0,      0,      0,      0],     # 72 - 73
            [16,     0,      0,      0,      0],     # 74 - 75
            [17,     0,      0,      0,      0],     # 76 - 77
            [18,     0,      0,      0,      0],     # 78 - 79
            [19,     0,      0,      0,      0],     # 80 - 81
            [20,     0,      0,      0,      0],     # 82 - 83
            [21,     0,      0,      0,      0],     # 84 - 85
            [22,     0,      0,      0,      0],     # 86 - 87
            [23,     0,      0,      0,      0],     # 88 - 89
            [24,     0,      0,      0,      0],     # 90 - 91
            [25,     0,      0,      0,      0],     # 92 - 93
            [26,     0,      0,      0,      0],     # 94 - 95
            [27,     0,      0,      0,      0],     # 96 - 97
            [28,     0,      0,      0,      0],     # 98 - 99
            [29,     0,      0,      0,      0],     # 100 - 101
            [30,     0,      0,      0,      0],     # 102 - 103
            [31,     0,      0,      0,      0],     # 104 - 105
            [32,     0,      0,      0,      0],     # 106 - 107
            [33,     0,      0,      0,      0],     # 108 - 109
            [34,     0,      0,      0,      0],     # 110 - 111
            [35,     0,      0,      0,      0],     # 112 - 113
            [36,     0,      0,      0,      0],     # 114 - 115
            [37,     0,      0,      0,      0],     # 116 - 117
            [38,     0,      0,      0,      0],     # 118 - 119
            [39,     0,      0,      0,      0],     # 120 - 121
            [40,     0,      0,      0,      0],     # 122 - 123
            [41,     0,      0,      0,      0],     # 124 - 125
            [12,    0.036,  0,      0.288,  0],          # 126 - 127
            [13,    0,      0,      0.379,  0],          # 128 - 129
            [14,    0,      0,      0.166,  0],          # 130 - 131
            [15,    0,      0,      0.166,  0],          # 132 - 133
            [16,    0.018,  0,      0,      0],      # 134  - 135
            [17,    0.182,  0,      0,      0],      # 136  - 137
            [18,    0.018,  0,      0,      0],      # 138  - 139
            [19,    0.036,  0,      0,      0],      # 140  - 141
            [20,    0.018,  0,      0,      0],      # 142  - 143
            [21,    0.236,  0,      0,      0],      # 144 - 145
            [22,    0.054,  0,      0,      0],      # 146 - 147
            ]))

    # S-Value
    #       Klasse, Schmetterling, Hase, Schaf, Küken
    prob.append(np.array([
           
            [0,     0,      0,      0,      0],     # 48  - 49
            [1,     0,      0,      0,      0],     # 50  - 51
            [2,     0,      0,      0,      0],     # 52  - 53
            [3,     0,      0,      0,      0],     # 54  - 55
            [4,     0,      0,      0.257,  0],     # 56 - 57
            [5,     0,      0,      0.273,  0],     # 58 - 59
            [6,     0,      0,      0.454,  0],     # 60 - 61
            [7,     0,      0,      0.015,  0],     # 62 - 63
            [8,     0,      0,      0,      0],     # 64 - 65
            [9,     0,      0,      0,      0],     # 66 - 67
            [10,     0,      0,      0,      0],     # 68 - 69
            [11,     0,      0,      0,      0],     # 70 - 71
            [12,     0,      0,      0,      0],     # 72 - 73
            [13,     0,      0,      0,      0],     # 74 - 75
            [14,     0,      0,      0,      0],     # 76 - 77
            [15,     0,      0,      0,      0],     # 78 - 79
            [16,     0,      0,      0,      0],     # 80 - 81
            [17,     0,      0,      0,      0],     # 82 - 83
            [18,     0,      0,      0,      0],     # 84 - 85
            [19,     0,      0,      0,      0],     # 86 - 87
            [20,     0.045,  0,      0,      0],     # 88 - 89
            [21,     0.106,  0,      0,      0],     # 90 - 91
            [22,    0.015,  0,      0,      0],     # 92 - 93
            [23,    0.015,  0,      0,      0],     # 94 - 95
            [24,     0,      0,      0,      0],     # 96 - 97
            [25,     0,      0,      0,      0],     # 98 - 99
            [26,     0,      0,      0,      0],     # 100 - 101
            [27,     0,      0,      0,      0],     # 102 - 103
            [28,     0,      0,      0,      0],     # 104 - 105
            [29,     0,      0,      0,      0],     # 106 - 107
            [30,    0.091,  0,      0,      0],      # 108 - 109
            [31,    0.075,  0,      0,      0],      # 110 - 111
            [32,     0,      0,      0,      0],     # 112 - 113
            [33,     0,      0,      0,      0],     # 114 - 115
            [34,     0,      0,      0,      0],     # 116 - 117
            [35,    0.257,  0.25,   0,      0],      # 118 - 119
            [36,    0.075,  0.091,  0,      0],      # 120 - 121
            [37,    0,      0.272,  0,      0],      # 122 - 123
            [38,    0,      0.227,  0,      0],      # 124 - 125
            [39,    0,      0.068,  0,      0],     # 126 - 127
            [19,    0,      0.091,  0,      0],     # 128 - 129
            [20,    0,      0,      0,      0.109],  # 130 - 131
            [21,    0,      0,      0,      0.091],  # 132 - 133
            [22,    0,      0,      0,      0],      # 134 - 135
            [23,    0,      0,      0,      0.054],  # 136 - 137
            [24,    0,      0,      0,      0.145],  # 138 - 139
            [25,    0,      0,      0,      0.091],  # 140 - 141
            [26,    0,      0,      0,      0.145],      # 142 - 143
            [27,    0,      0,      0,      0.072],      # 144 - 145
            [28,    0,      0,      0,      0.109],      # 146 - 147
            [29,    0.015,  0,      0,      0.127],      # 148 - 149
            [30,    0.151,  0,      0,      0.054],      # 150 - 151
            [31,    0.061,  0,      0,      0],      # 152 - 153
            [32,    0.061,  0,      0,      0],  # 154 - 155
            [33,    0.030,  0,      0,      0],      # 156 - 157
            ]))

    ret = []

    ret.append(prob[0][width2Class(w), :])
    ret.append(prob[1][height2Class(h), :])
    ret.append(prob[2][H2Class(ch), :])
    ret.append(prob[3][S2Class(s), :])

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
def H2Class(b):
    c = math.floor(((b - 24) / 2))
    if c > 61:
        c = 61
    elif c < 0:
        c = 0
    return c


# Function to calculate the given Class from the Saturation
def S2Class(g):
    c = math.floor(((g - 48) / 2))
    if c > 54:
        c = 54
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
    for i in range(0, len(classprob)):
        ret.append(calcProb(classprob[i]))
        
    return ret


# No Really... This is really where the magic happens
def thisIsWhereTheMagicHappens(feat):
    
    klasse = "Unknown"
    
    w = feat[0]
    h = feat[1]
    hc = feat[3][0]
    v = feat[3][1]
    cl = ["Schmetterling", "Hase", "Schaf", "Kueken"]
    
    classprob = loadProbabilityArrays(h, w, hc, v)
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
    yK = [393.568, 220.339, 114.51]  # Küken
    yH = [305.751, 230.735, 117.42]  # Hasen
    yS = [339.887, 237.996, 149.33]  # Schafe
    yP = [327.529, 245.186, 121.40]  # Schmetterlinge
    cl = ["Kueken", "Hase", "Schaf", "Schmetterling"]

    n = len(yK)  # Anzahl Merkmale

    w = feat[0]
    h = feat[1]
    c = sum(map(float, filter(None, feat[2][1:])))/(len(feat[2])-1)

    rmse.append(math.sqrt((1 / n) * (pow((yK[0] - w), 2) + pow((yK[1] - h), 2) + pow((yK[2] - c), 2))))
    rmse.append(math.sqrt((1 / n) * (pow((yH[0] - w), 2) + pow((yH[1] - h), 2) + pow((yH[2] - c), 2))))
    rmse.append(math.sqrt((1 / n) * (pow((yS[0] - w), 2) + pow((yS[1] - h), 2) + pow((yS[2] - c), 2))))
    rmse.append(math.sqrt((1 / n) * (pow((yP[0] - w), 2) + pow((yP[1] - h), 2) + pow((yP[2] - c), 2))))

    klasse = cl[rmse.index(min(rmse))]

    return klasse


def rmseHSVClassifier(feat):

    klasse = "Unknown"
    rmse = []

    # Mittelwerte Breite Höhe H , S
    yK = [377.200, 223.600, 51.091, 141.018]  # Küken
    yH = [299.200, 222.220, 26.568, 122.545]  # Hasen
    yS = [330.440, 234.310, 60.863, 59.000]  # Schafe
    yP1 = [324.070, 244.550, 61.450, 90.720]  # Schmetterlinge
    yP2 = [324.070, 244.550, 140.000, 116.000]  # Schmetterlinge 
    yP3 = [324.070, 244.550, 28.000, 150.000]  # Schmetterlinge
    cl = ["Kueken", "Hase", "Schaf", "Schmetterling", "Schmetterbling", "Schmetterlingling"]

    n = len(yK)  # Anzahl Merkmale

    w = feat[0]
    h = feat[1]
    hc = feat[3][0]
    v = feat[3][1]
    
    print(str(hc) + " , " + str(v))

    rmse.append(math.sqrt((1 / n) * (pow((yK[0] - w), 2) + pow((yK[1] - h), 2) + pow((yK[2] - hc), 2) + pow((yK[3] - v), 2))))
    rmse.append(math.sqrt((1 / n) * (pow((yH[0] - w), 2) + pow((yH[1] - h), 2) + pow((yH[2] - hc), 2) + pow((yH[3] - v), 2))))
    rmse.append(math.sqrt((1 / n) * (pow((yS[0] - w), 2) + pow((yS[1] - h), 2) + pow((yS[2] - hc), 2) + pow((yS[3] - v), 2))))
    rmse.append(math.sqrt((1 / n) * (pow((yP1[0] - w), 2) + pow((yP1[1] - h), 2) + pow((yP1[2] - hc), 2) + pow((yP1[3] - v), 2))))
    rmse.append(math.sqrt((1 / n) * (pow((yP2[0] - w), 2) + pow((yP2[1] - h), 2) + pow((yP2[2] - hc), 2) + pow((yP2[3] - v), 2))))
    rmse.append(math.sqrt((1 / n) * (pow((yP3[0] - w), 2) + pow((yP3[1] - h), 2) + pow((yP3[2] - hc), 2) + pow((yP3[3] - v), 2))))

    klasse = cl[rmse.index(min(rmse))]

    return klasse


def TreeClassifier(feat):
    
    w = feat[0]
    h = feat[1]
    hue = feat[3][0]
    sat = feat[3][1]
    
    if ((w > 365) & (w < 420)):
        if ((h > 212) & (h < 238)):
            if ((hue > 40) & (hue < 60) & (sat > 120) & (sat < 160)): 
                return "Kueken"
            else:
                return "Stoerobjekt1"
    elif (((h * w) > 60000) & ((h * w) < 75000)):
        if ((hue > 15) & (hue < 38) & (sat > 107) & (sat < 150)): 
            return "Hase"
        else:
            return "Stoerobjekt2"
    elif (not((h > 220) & (h < 260))):
        return "Stoerobjekt3"
    elif (not((w > 280) & (w < 370))):
        return "Stoerobjekt4"
    elif ((hue > 50) & (hue < 70) & (sat > 50) & (sat < 75)):
        return "Schaf"
    else:
        if ((hue > 45) & (hue < 75) & (sat > 78) & (sat < 115)): 
            return "Schmetterbling" #grün
        elif ((hue > 90) & (hue < 160) & (sat > 110) & (sat < 135)):
            return "Schmetterling" #rot
        elif ((hue > 15) & (hue < 40) & (sat > 135) & (sat < 170)):
            return "Schmetterling" #gäl
        else:
            return "Stoerobjekt5"
        
    return "ERROR in Class"


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
            rmsehsvklasse = rmseHSVClassifier(feat)
            
            bayesklasse = thisIsWhereTheMagicHappens(feat)
            
            treeklasse = TreeClassifier(feat)
            
            print("RMSE: {}, Bayess: {}, Tree: {}".format(rmseklasse, bayesklasse, treeklasse))

            cv2.putText(frame, "RMSE (RGB): {}".format(rmseklasse),
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (int(0), int(155), int(0)),
                        2, cv2.LINE_AA)
            
            cv2.putText(frame, "RMSE (HSV): {}".format(rmsehsvklasse),
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (int(0), int(155), int(0)),
                        2, cv2.LINE_AA)
            
            cv2.putText(frame, "Bayes: {}".format(bayesklasse),
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (int(0), int(155), int(0)),
                        2, cv2.LINE_AA)
            
            cv2.putText(frame, "Tree: {}".format(treeklasse),
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (int(0), int(155), int(0)),
                        2, cv2.LINE_AA)

            cv2.imshow("Original", frame)
            if (not do_live):
                cv2.imwrite("_Data/Puit/{}.jpg".format(fnames[cnt][6:10]), frame)
#               cv2.imwrite("_Data/Puit/{}_canny.jpg".format(fnames[cnt][6:10]), edges)

            if (cv2.waitKey(20) & 0xFF) == ord("q"):
                break
            
            # Take Picture with Classname
            if (cv2.waitKey(20) & 0xFF) == ord("p"):
                cv2.imwrite("_Data/Test/{}.jpg".format(rmseklasse), frame)
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
