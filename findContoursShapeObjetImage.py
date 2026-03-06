"""
---------------------------------------- Calcul des coefficients de la efd pour la detection de :
            >> la taille de la graine {h0}: fondammentale
            >> les differentes formes {h1,.....h9}
                                                                        --------------------------------------------------
"""
import cv2
import numpy as np
import find_contour as fc  # importation de notre  bibliotheque find_contour
import chargement_images as cimg
# import enregistrer_donnees as engd

# ------ ce code est valide tout aussi bien pour un tableau d'image que pour une image individuel

# ------------- constante
MIN_S, MAX_S = 30, 255  # la valeur min doit etre > 3

# debut de detection de contour
path = "imageAcquisition/haricotRouge_charancon_105ech/imgRougeCh7.jpg"
path = 'D://PROJECT//SortingSystem_Using_ANN//imageAcquisition//haricotBlanc_charancon_105ech'
imgArray = cimg.loadingImages_callback(path)
harmonical = []
for cs in range(0, len(imgArray)):
    cv2.namedWindow("original")  # nouvelle fenetre windows
    cv2.createTrackbar("seuil", "original", MIN_S, MAX_S, fc.empty) # creation d'une barre de defilement
    while True:
        print("image: ", cs)  # afficher en temps reelles la position de l'image correspondant
        img = imgArray[cs]  # lecture individuelles de chaque image
        cv2.imshow("original", img)
        imgBlur = cv2.GaussianBlur(img, (3, 3), 2)  # elimination des bruit sur l'image original
        imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)  # passage de l'image BGR en image de gris (0, 255)
        val_seuil = int(cv2.getTrackbarPos("seuil", "original"))  # valeur du seuil pour une meilleur detection  du contour
        #
        print("seuil detection:", val_seuil)
        imgCanny = cv2.Canny(imgGray, val_seuil, val_seuil * 2)
        val_contour = fc.ContourObject(imgCanny)  # appel de notre fonction permettant de dessiner le contour detecter
        print("taille contour:", len(val_contour))  # elle varie selon le seuil de detection
        # 
        """
        Dans cette partir nous avons effectuer une transformation d'une liste multidimension en une liste 1D
        Il s'agissait principalement d'applatir les valeurs de contour 3D et une liste 1D
        """
        contour_niv1 = [cnt for i in val_contour for cnt in i]  # 1 ere reduction de la liste
        contour_niv2 = [cnt1 for i1 in contour_niv1 for cnt1 in i1]  # 2 nde reduction de la liste
        # contour_niv3 = [cnt2 for i2 in contour_niv2 for cnt2 in i2]
        # harmo = []
        harmo = fc.coeffs_closeEllipticFourierDescriptors(contour_niv2)
        
        key = cv2.waitKey(100)
        #print(key)
        if key in [ord('a'), 1048673]:
            print('a pressed!')
        elif key in [27, 1048603]:  # ESC key to abort, close window
            cv2.destroyAllWindows()
            break
        # fin ----------------------------------------------------------------------------------------------------
    #a0 = harmo[:,0]
    #b0 = harmo[:,1]
    #c0 = harmo[:,2]
    #d0 = harmo[:,3]
    harmonical.append(harmo)
    # fin ---------------------------------------------------------------------------------------------------------------
# engd.save_HarmonicalFormat_EXCEL_callback(harmonical)