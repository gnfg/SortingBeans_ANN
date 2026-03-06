"""
Nous calculons ici la couleur moyenne et l'ecart-type de l'objet sur chacune des images
"""
# importation des Bibliotheque
import numpy as np  # manipulation des matrices
# import matplotlib.pyplot as plt
import chargement_images as cimg
import enregistrer_donnees as ed   # Notre bibliotheque pour le chargement des donnees dans le dataset

"""
path = "imageAcquisition/haricotRouge_charancon_105ech/imgCh7_Detected.jpg"  # repertoire de l'image
img = cv2.imread(path)  # lecture de l'image originale
"""
# les images utiliser ici ce sont les images dont la couleur a deja ete detecter au prealable
path = 'D:/PROJECT/SortingSystem_Using_ANN/imageAcquisition/haricotBlanc_charancon_105ech'
imgArray = cimg.loadingImages_callback(path)  # nous avons le tableau de tous les images de notre base de donnnees

"""
    >> Cette fonction nous calcule dans un premier temps la moyenne et dans un second temps l'ecart-type 
    >> espace RVB = espace de chrominance (Rouge, Vert, Bleu)
    >> e_ChR : espace de chrominance Rouge
"""
liste_mean = []
liste_stdev = []
for l in range(0, len(imgArray)):
    img = imgArray[l]
    # separation de l'image d'origine dans e_Ch[R,V,B]
    img_e_ChB = img[:, :, 0]  # image original dans e_ChB
    img_e_ChV = img[:, :, 1]  # image original dans e_ChV
    img_e_ChR = img[:, :, 2]  # image original dans e_ChR

    rows = img.shape[1]  # Width---> ligne
    cols = img.shape[0]  # Height ---> colonne

    liste3D = []  # liste de stockage des valeurs ou l'objet est de couleur [rouge, vert, bleu] dans l'image d'origine

    for cpt in range(0, 3):  # image dans les 3 espaces
        img_Chrom = img[:, :, cpt]  # nous avons ici les 3 images respectivement dans dans chaque espace de chrominance
        liste = []
        for x in range(0, cols):
            for y in range(0, rows):
                if img_Chrom[x][y] == 0:  # elimination des valeur nul present dans la matrice d'image de chrominance rouge
                    pass
                else:  # valeur ou la couleur est defini dans l'image detecter
                    liste.append(img_Chrom[x][y])  # creer une liste permettant de stocker ces valeurs
                # fin ------------------------------------------------------------------------------------------------------
            # fin ----------------------------------------------------------------------------------------------------------
        # fin --------------------------------------------------------------------------------------------------------------
        liste3D.append(liste) # nous avons la une liste a trois colonnes contenant les valeurs des pixels pour chaque zone de chrominance
    # fin ------------------------------------------------------------------------------------------------------------------
    # calcul de la moyenne et de l'ecart-type(ET) des pixel de l'objet detectee dans notre espace RVB
    mean_BVR = []  # moyenne detection objet dans l'espace RVB
    stDev_BVR = []  # SD = Standard Deviation (Ecart-Type) detection objet dans l'espace RVG
    for i in range(0, 3):
        mean_BVR.append(np.mean(liste3D[i]))
        stDev_BVR.append(np.std(liste3D[i]))
    # fin ----------------------------------------------------------------------------------------------------------------
    # print(mean_RVB, stDev_RVB)
    liste_mean.append(mean_BVR)
    liste_stdev.append(stDev_BVR)
    # a = np.array(liste_stdev)
    #
ed.save_MeanStDevFormat_EXCEL_callback(liste_mean, liste_stdev)   # enregistrement moyenne/ecart type dans un fichier xlxs

