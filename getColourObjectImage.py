import cv2
import numpy as np
import find_contour as fc   # importation de ma bibliotheque
"""
Nous allons appprendre  comment detecter la couleur sur une image
"""

path = r'/Users/frankgiressegadjuinianga/Documents/Master2TSI_UBa/Dossier_projet/SortingBeansSystem_Using_ANN/imageAcquisition/imageTest/rc5.jpg'
img = cv2.imread(path)
cv2.namedWindow("barre guidage")   # nouvelle fenetre windows
cv2.resizeWindow("barre guidage", 640, 240)  # redimensionnement de la nouvelle fenetre windows
cv2.createTrackbar("Teinte min", "barre guidage", 0, 179, fc.empty)  # 0, 179
cv2.createTrackbar("Teinte max", "barre guidage", 179, 179, fc.empty)  # 179, 179
cv2.createTrackbar("Sat min", "barre guidage", 53, 255, fc.empty)  # 0, 255
cv2.createTrackbar("Sat max", "barre guidage", 255, 255, fc.empty)  # 255, 255
cv2.createTrackbar("Val min", "barre guidage", 13, 255, fc.empty)  # 0, 255
cv2.createTrackbar("Val max", "barre guidage", 255, 255, fc.empty)  # 255, 255

# 2. vous allons maintenant definir les valeurs individuels de saturation,teinte permettant de detecter la couleur verte
while True:
    img = cv2.imread(path,None)
    # 1. transformation HSV (Hue Saturation Value)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgHSV = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2HSV)
    t_min = cv2.getTrackbarPos("Teinte min", "barre guidage")
    t_max = cv2.getTrackbarPos("Teinte max", "barre guidage")
    s_min = cv2.getTrackbarPos("Sat min", "barre guidage")
    s_max = cv2.getTrackbarPos("Sat max", "barre guidage")
    v_min = cv2.getTrackbarPos("Val min", "barre guidage")
    v_max = cv2.getTrackbarPos("Val max", "barre guidage")
    print(t_min, t_max, s_min, s_max, v_min, v_max) #### teinte min et max, saturation min et max, luminance min et max
    
    # 3. appliquer les differente valeur a notre image HSV
    lower = np.array([t_min, s_min, v_min])
    upper = np.array([t_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower, upper)

    # 4. detection de la couleur de l'image
    colorObjectDetected = cv2.bitwise_and(img, img, mask=mask)
    # sauvegarder l'image detcter dans la meme fenetre que l'image originale
    cv2.imwrite("imageTest/imgColorDetected.jpg", colorObjectDetected)   

    # affichage des images
    cv2.imshow("Original ", img)
    cv2.imshow("HSV image", imgHSV)
    cv2.imshow("mask", mask)
    cv2.imshow("couleur detecter", colorObjectDetected)
    key = cv2.waitKey(200)
    print(key)
    if key in [ord('a'), 1048673]:
        print('a pressed!')
    elif key in [27, 1048603]:  # ESC key to abort, close window
        cv2.destroyAllWindows()
        break
    # fin ------------------------------------------------------------------------------------------------------------------
