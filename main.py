"""
-------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------
-------------------------------- << Programme principale >> -------------------------------------------
<< Thematique >>     Systeme de tri des semences aricoles a base de reseau de neuronne convolutionnel
<< Auteurs >>        M. Frank GADJUI NIANGA
                     M. Desirat TINHOUETO COMLAN
<< Enseignant >>     Pr. Ludovic JOURNAUX
<< MASTER TSI >>     Master 2 Image-Vision, Traitement de Signal et Image
<< Etablissement >>  Universite de Bourgogne
<< Annee scolaire >> 2021-2022
<< Date de debut >>  22/09/2021
<< Date de fin >>    ?
------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------
"""

"""
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Importation des bibliotheques <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
"""
import numpy as np
import detection_tri as dt  # Notre bibliotheque personaliser
import chargement_images as cimg
import enregistrer_donnees as enred

"""
Ici faisons appelle aux differentes fonctions contenues dans nos bibliotheque prealablement importer
------------------------------------------------------------------------------------------------------------------------
>>>>>>>>>>>>>>>>>>>>>>>>>>>> NB : la valeur du seul de detection est > 3 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
------------------------------------------------------------------------------------------------------------------------
"""

# ------------------- Chargement et lecture des images reelles (origines)
# D:\PROJECT\SortingSystem_Using_ANN\imageAcquisition\haricotRouge_charancon_105ech
path = r'/Users/frankgiressegadjuinianga/Documents/Master2TSI_UBa/Dossier_projet/SortingBeansSystem_Using_ANN/imageAcquisition/imageTest'
imgArray = cimg.loadingImages_callback(path)  # lecture de l'image original  cv2.imread(path)

# couleur de base:
bordeau = [128, 0, 0]

# ------------------- Initialisation des listeC:\Users\ASUS\Documents\projet\NEW\SortingBeansSystem_Using_ANN\imageAcquisition\image_haricot
list_moy = []
list_ecart = []
list_harmo = []
sortiDesire = []

# ------------------- Detecter la couleur de la graine
for i in range(0, len(imgArray)):
    img = imgArray[i]
    imgDetected = dt.ObjectColorDetection_callback(img)

    # ------------------- evaluer l'ecart-type et la moyenne de couleur
    val_mean, val_stDev = dt.Mean_StDev_ColorObjectDetected_callback(imgDetected)
    # l'ecart type permettant de calculer la dispersion autour de moyenne alors, plus elle est petite plus la couleur est uniforme
    if np.mean(val_stDev) < 10:
        y = 1  # travaillant avec des image RVB avec des composant dans le R, V, B alors la moyenne de l'ecartype de l'image RVB est calculer de cette facons
    else:
        y = 0
    sortiDesire.append(y)
    list_moy.append(val_mean)
    list_ecart.append(val_stDev)
    # ------------------- Detecter et dessiner le contour de l'pbjet ainsi que les coefficient de fourier representant la taille et la forme de la graine
    efd_coeffs = dt.coeffsEFD_ObjectShapeDetected_callback(img)
    list_harmo.append(efd_coeffs)

# ------------------- enregistrement de tous les caracteristiques
enred.save_Format_EXCEL_callback(list_moy, list_ecart, list_harmo, sortiDesire)
