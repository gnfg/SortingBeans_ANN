"""
--------------------------------- Bibliotheque permettant de:
        >> Lire l'ensemble des image d'un dossier et la sauvegarder dans un tableau d'image
                                                                            --------------------------------------------
"""
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2

"""
mypath ='D:/PROJECT/SortingSystem_Using_ANN/imageAcquisition/haricotBlanc_charancon_105ech'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
Tab_images = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
  Tab_images[n] = cv2.imread(join(mypath, onlyfiles[n]))
"""  

def loadingImages_callback(Pathfolder): # le chemin d'access au dossier des images
  fichier_img = [f for f in listdir(Pathfolder) if isfile(join(Pathfolder, f))]  
  Tab_images = np.empty(len(fichier_img), dtype=object)  # initialisation vide du tableau des images
  for n in range(0, len(fichier_img)):
    Tab_images[n] = cv2.imread(join(Pathfolder, fichier_img[n])) # chargement des images dans le tableau
  return Tab_images

