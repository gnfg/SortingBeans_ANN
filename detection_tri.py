"""
--------------------------------- Bibliotheque permettant de:
        >> detecter la couleur des objets presents dans une image BGR
        >> calculer la moyenne et l'ecart-type des couleurs sur l'objet detectee
        >> d'enregistrer les donnees dans le dataset
                                                                            --------------------------------------------
"""
import cv2
import numpy as np
import find_contour as fc   # Notre bibliotheque personaliser
# import enregistrer_donnees as ed   # Notre bibliotheque pour le chargement des donnees dans le dataset

"""
Cette fonction nous permet a partir d'une une image originale de detecter la couleur specifique d'un objet a l'interieur
"""
def ObjectColorDetection_callback(originalImage):
    # ------------------------------------------------------------------------------------------------------------------
    cv2.namedWindow("Barre navigation")  # nouvelle fenetre windows
    cv2.resizeWindow("Barre navigation", 640, 240)  # redimensionnement de la nouvelle fenetre windows
    cv2.createTrackbar("Tei. min", "Barre navigation", 10, 179, fc.empty)  # 0, 179
    cv2.createTrackbar("Tei. max", "Barre navigation", 179, 179, fc.empty)  # 179, 179
    cv2.createTrackbar("Sat. min", "Barre navigation", 53, 255, fc.empty)  # 0, 255
    cv2.createTrackbar("Sat. max", "Barre navigation", 255, 255, fc.empty)  # 255, 255
    cv2.createTrackbar("Lum. min", "Barre navigation", 13, 255, fc.empty)  # 0, 255
    cv2.createTrackbar("Lum. max", "Barre navigation", 255, 255, fc.empty)  # 255, 255

    # Nous allons maintenant definir les valeurs individuels de saturation,teinte permettant de detecter la couleur verte
    while True:
        img = originalImage
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
        cv2.imshow("couleur objet ", color_object_detected)
        key = cv2.waitKey(1)
        if key in [ord('a'), 1048673]:
            print('a pressed!')
        elif key in [27, 1048603]:  # ESC key to abort, close window
            cv2.destroyAllWindows()
            break
        # fin ----------------------------------------------------------------------------------------------------------
    imgDetect = color_object_detected
    # fin --------------------------------------------------------------------------------------------------------------
    return imgDetect

"""
    >> Cette fonction nous calcule dans un premier temps la moyenne et dans un second temps l'ecart-type 
    >> espace RVB = espace de chrominance (Rouge, Vert, Bleu)
    >> e_ChR : espace de chrominance Rouge
"""

def Mean_StDev_ColorObjectDetected_callback(imgColorDetected):
    # ------------------------------------------------------------------------------------------------------------------
    #
    #liste_mean = []
    #liste_stdev = []
    #for l in range(0, len(imgColorDetected)):
        #img = imgColorDetected[l]
    img = imgColorDetected
    # img_e_ChR = img[:, :, 0]  # image original dans e_ChR
    # img_e_ChV = img[:, :, 1]   # image original dans e_ChV
    # img_e_ChB = img[:, :, 2]   # image original dans e_ChB
    rows = img.shape[1]  # ligne
    cols = img.shape[0]  # colonne
    liste_3d = []  # liste vide de stockage des valeurs ou l'objet est detecter dans (e_ChR, e_ChV, e_ChB)
    for cpt in range(0, 3):  #
        img_chrom = img[:, :, cpt]  # 3 espaces de chrominance ou l'objet est appercu
        liste = []
        for x in range(0, cols):
            for y in range(0, rows):
                if img_chrom[x][y] == 0:  # elimination des valeurs nul present dans la matrice d'image de chaque e_Ch
                    pass
                else:  # recuperation des valeurs non null de ces matrice
                    liste.append(img_chrom[x][y])  # liste permettant de stocker ces valeurs
                # fin -------------------------------------
            # fin -------------------------------------------------------
        # fin  ----------------------------------------------------------------------------------------------------
        liste_3d.append(liste)  # stockage dans liste3D des valeurs des pixels non nul(ou objet detectet) pour chaque e_Ch
    # fin ------------------------------------------------------------------------------------------------------------
    # calcul la moyenne et l'ecart-type
    mean_rvb = []  # moyenne detection objet dans e_RVB
    stdev_rvb = []  # StDev = Standard Deviation (Ecart-Type) detection objet dans e_RVB
    for i in range(0, 3):
        mean_rvb.append(np.mean(liste_3d[i]))
        stdev_rvb.append(np.std(liste_3d[i]))
    print(mean_rvb, stdev_rvb)
    # fin ----------------------------------------------------------------------------------------------------------
    # liste_mean.append(mean_rvb)
    # liste_stdev.append(stdev_rvb)
    # fin ------------------------------------------------------------------------------------------------------------
    # ed.save_MeanStDevFormat_TEXT_callback(liste_mean, liste_stdev)    # enregistrement  moyenne/ecart type dans un fichier txt
    # ed.save_MeanStDevFormat_EXCEL_callback(liste_mean, liste_stdev)   # enregistrement moyenne/ecart type dans un fichier xlxs
    return mean_rvb, stdev_rvb

"""
    >> Cette fonction nous calcule dans un premier temps la moyenne et dans un second temps l'ecart-type 
    >> Coefficients de fourier elliptique.
"""

def coeffsEFD_ObjectShapeDetected_callback(image):
    #-------------------------------------------------------------------------------------------------------------------
    cv2.namedWindow("original")  # nouvelle fenetre windows
    cv2.createTrackbar("seuil", "original", 15, 255, fc.empty) # creation d'une barre de defilement
    efd_harmo = []
    while True:
        img = image
        cv2.imshow("original", img)
        img_blur = cv2.GaussianBlur(img, (3, 3), 1)  # elimination des bruit sur l'image original
        img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)  # passage de l'image BGR en image de gris (0, 255)
        val_seuil = int(cv2.getTrackbarPos("seuil", "original"))  # valeur du seuil pour une meilleur detection  du contour
        print("seuil: ", val_seuil)
        img_canny = cv2.Canny(img_gray, val_seuil, val_seuil * 2)
        val_contour = fc.ContourObject(img_canny)  # appel de notre fonction permettant de dessiner le contour detecter
        # 
        print("longueur contour: ", len(val_contour))  # inversement proportinnel au seuil de detection
        contour_niv1 = [cnt for i in val_contour for cnt in i]  # 1 ere reduction de la taille des contour
        contour_niv2 = [cnt1 for i1 in contour_niv1 for cnt1 in i1]  # 2 nde reduction de la liste
        #
        harmonical = fc.coeffs_closeEllipticFourierDescriptors(contour_niv2) # valeurs des coefficients de fourier
        print("efd hamonical: ", harmonical)
        #
        key = cv2.waitKey(1)
        if key in [ord('a'), 1048673]:
            print('a pressed!')
        elif key in [27, 1048603]:  # ESC key to abort, close window
            cv2.destroyAllWindows()
            break
        # fin ----------------------------------------------------------------------------------------------------------
    # fin ---------------------------------------------------------------------------------------------------------------
    efd_harmo = harmonical
    return efd_harmo
