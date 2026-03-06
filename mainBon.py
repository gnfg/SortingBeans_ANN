"""
-------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------
-------------------------------- << Programme principale >> -------------------------------------------
<< Thematique >>     Systeme de tri des semences aricoles a base de reseau de neuronne convolutionnel
<< Auteurs >>        M. Frank GADJUI NIANGA
                     M. Desirat TINHOUETO COMLAN
<< Enseignant >>     Pr. Ludovic JOURNAUX
<< MASTER TSI >>     Master 2 Traitement de Signal et Image
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

import detection_tri as dt          # Notre bibliotheque personaliser
import chargement_images as cimg

"""
Ici faisons appelle aux differentes fonctions contenues dans nos bibliotheque prealablement importer
------------------------------------------------------------------------------------------------------------------------
>>>>>>>>>>>>>>>>>>>>>>>>>>>> NB : la valeur du seul de detection est > 3 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
------------------------------------------------------------------------------------------------------------------------
"""

# ------------------- Chargement et lecture des images reelles (origines)
path1 = 'D:\PROJECT\SortingSystem_Using_ANN\imageAcquisition\haricotBlanc_bonne_105ech' # repertoire de l'image "imageAcquisition/haricotRouge_charancon_105ech/imgRougeCh7.jpg" 
imgArray1 = cimg.loadingImages_callback(path1) # lecture de l'image original  cv2.imread(path)

# ------------------- Detecter la couleur de la graine
list_moy_bon = []
list_ecart_bon = []
list_harmo_bon = []
for i in range(0, len(imgArray1)):
    img1 = imgArray1[i]
    imgDetected1 = dt.ObjectColorDetection_callback(img1)

# ------------------- evaluer l'ecart-type et la moyenne de couleur
    val_mean, val_stDev = dt.Mean_StDev_ColorObjectDetected_callback(imgDetected1)
    list_moy_bon.append(val_mean)
    list_ecart_bon.append(val_stDev)
    # print(val_mean, val_stDev)
# ------------------- Detecter et dessiner le contour de l'pbjet ainsi que les coefficient de fourier representant la taille et la forme de la graine

    efd_coeffs = dt.coeffsEFD_ObjectShapeDetected_callback(img1)
    list_harmo_bon.append(efd_coeffs)
    #print("efd harmonical: ", efd_coeffs)