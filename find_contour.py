"""
--------------------------------- Bibliotheque permettant de:
    >> detecter le contour des objets
    >> dessiner ce contour de ces objets --------------------
    >> calculer les oefficients de la transformer de fourier elliptique sur contour fermee (EFD)
                                                                                            -------------------------------------
"""

import cv2
import numpy as np
from pyefd import elliptic_fourier_descriptors, calculate_dc_coefficients     # eFD

" >> Cette fonction permet de trouver et de dessiner le contour des objets sur une image pour un seuil entre (0, 255) "

def ContourObject(imageCanny):
    # ----------------------------------------------------------------------------------------------------------------------
    contours, hierarchy = cv2.findContours(imageCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # TREE cherchons le contour grace a la fonction "findContour" d'opencv
    imageDrawing = np.zeros((imageCanny.shape[0], imageCanny.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        cv2.drawContours(imageDrawing, contours, i, (0, 255, 0), 3, cv2.LINE_8, hierarchy, 0)  # # dessinons tous les coutours en vert (0, 255, 0)
    # fin ------------------------------------------------------------------------------------------------------------------
    # Show in a window
    cv2.imshow("contour image", imageDrawing)
    # prevoir l'ajouter de la sortie normale
    return contours

"""
    >> calculer les harmonique ou coefficients de fourier
      h0 --> fondamentale : << permet de caracteriser la taille de l'objet >>
      hi --> harmobique : << permet de caracteriser ls differentes formes de l'objet >>
"""
def coeffs_closeEllipticFourierDescriptors(cont):
    # ----------------------------------------------------------------------------------------------------------------------
    val_coeffs = elliptic_fourier_descriptors(np.squeeze(cont), order=5, normalize=True)  # cherchons les coefficient sur l'ensemble des contour de la EFD
    return val_coeffs

def fondamental_coeffs_closeEllipticFD(cont):
    # ----------------------------------------------------------------------------------------------------------------------
    val1, val2 = calculate_dc_coefficients(cont)
    return val1, val2

def empty(a):
    # ----------------------------------------------------------------------------------------------------------------------
    pass


