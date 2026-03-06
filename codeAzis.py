import cv2
import numpy as np
import find_contour as fc   # importation de ma bibliotheque
"""
Nous allons appprendre  comment detecter la couleur sur une image
"""
#
path = r'/Volumes/VAGA/Janvier 2022 (Hydrique)/Camera Thermique/Mais/Pot 25/Pot 25 1.jpg'
# ------------------------------------------------------------------------------------------------------------------
cv2.namedWindow("Barre navigation")  # nouvelle fenetre windows
cv2.resizeWindow("Barre navigation", 509, 313)  # redimensionnement de la nouvelle fenetre windows
cv2.createTrackbar("Tei. min", "Barre navigation", 10, 179, fc.empty)  # 0, 179
cv2.createTrackbar("Tei. max", "Barre navigation", 179, 179, fc.empty)  # 179, 179
cv2.createTrackbar("Sat. min", "Barre navigation", 53, 255, fc.empty)  # 0, 255
cv2.createTrackbar("Sat. max", "Barre navigation", 255, 255, fc.empty)  # 255, 255
cv2.createTrackbar("Lum. min", "Barre navigation", 13, 255, fc.empty)  # 0, 255
cv2.createTrackbar("Lum. max", "Barre navigation", 255, 255, fc.empty)  # 255, 255

# Nous allons maintenant definir les valeurs individuels de saturation,teinte permettant de detecter la couleur verte
while True:
    img = cv2.imread(path)
    # Transformation HSV (Hue Saturation Value)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
    t_min = cv2.getTrackbarPos("Tei. min", "Barre navigation")
    t_max = cv2.getTrackbarPos("Tei. max", "Barre navigation")
    s_min = cv2.getTrackbarPos("Sat. min", "Barre navigation")
    s_max = cv2.getTrackbarPos("Sat. max", "Barre navigation")
    v_min = cv2.getTrackbarPos("Lum. min", "Barre navigation")
    v_max = cv2.getTrackbarPos("Lum. max", "Barre navigation")
    # determination des valeurs du masques
    lower = np.array([t_min, s_min, v_min])
    upper = np.array([t_max, s_max, v_max])
    mask = cv2.inRange(img_hsv, lower, upper)
    # Detection couleur de l'objet par application du masque
    color_object_detected = cv2.bitwise_and(img, img, mask=mask)
    #cv2.imshow("couleur objet ", color_object_detected)
    # affichage des images
    cv2.imshow("Original ", img_rgb)
    cv2.imshow("HSV image", img_hsv)
    cv2.imshow("mask", mask)
    cv2.imshow("couleur detecter", color_object_detected)
    #cv2.imwrite('/Users/frankgiressegadjuinianga/Documents/Master2TSI_UBa/Dossier_projet/SortingBeansSystem_Using_ANN/imageAcquisition/imageTest', color_object_detected)
    key = cv2.waitKey(1)
    if key in [ord('a'), 1048673]:
        print('a pressed!')
    elif key in [27, 1048603]:  # ESC key to abort, close window
        cv2.destroyAllWindows()
        break