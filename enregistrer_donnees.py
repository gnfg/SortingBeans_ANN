"""
--------------------------------- Bibliotheque permettant d'enregistrer les donnees moyenne, ecart-type, coeffsEFD:
        >> au format de fichier "dat" ou "txt" dans le dataset
        >> au format de fichier 'xlxs' ou 'csv' dans notre dataset
                                                                            --------------------------------------------
"""
import pandas as pd
import numpy as np
from openpyxl import load_workbook


# enregistrement au format ".txt" / ".dat"
def save_MeanStDevFormat_TEXT_callback(image_mean, image_stdev):
    # -------------------------------------------------------------------------------------------------------------------
    moy = np.array(image_mean)  # nous transformons en matrice colonne notre vecteur moyenne de donnee
    stdev = np.array(image_stdev)  # nous transformons en matrice colonne notre vecteur moyenne de donnee
    path_saving = r'/Users/frankgiressegadjuinianga/Documents/Master2TSI_UBa/Dossier_projet/SortingBeansSystem_Using_ANN/BeansDataset.txt'
    tableVal = [moy[:,0], moy[:,1], moy[:,2], stdev[:,0], stdev[:,1], stdev[:,2]]
    np.savetxt(path_saving, np.column_stack(tableVal), fmt='%.3e', delimiter="  ", header='Moy_ChR   Moy_ChV   Moy_ChB   ET_ChR  ET_ChV  ET_ChB')

# enregistrement au format ".xlsx" / ".csv"
def save_MeanStDevFormat_EXCEL_callback(image_mean, image_stdev):
    # -------------------------------------------------------------------------------------------------------------------
    #
    moy = np.array(image_mean)   # nous transformons en matrice colonne notre vecteur moyenne de donnee
    stdev = np.array(image_stdev) # nous transformons en matrice colonne notre vecteur moyenne de donnee
    path_saving = r'/Users/frankgiressegadjuinianga/Documents/Master2TSI_UBa/Dossier_projet/SortingBeansSystem_Using_ANN/BeansDataset.xlsx'
    writer = pd.ExcelWriter(path_saving, engine='openpyxl')
    wb = writer.book
    df = pd.DataFrame({'Moy_ChB': moy[:,0],
                       'Moy_ChV': moy[:,1],
                       'Moy_ChR': moy[:,2],
                       'ET_ChB': stdev[:,0],
                       'ET_ChV': stdev[:,1],
                       'ET_ChR': stdev[:,2]})
    df.to_excel(writer, index=False)
    wb.save(path_saving)
    
# enregistrement au format des donnes dans notre dataset ".xlsx" / ".csv"
def save_Format_EXCEL_callback(liste_mean, liste_stdev, liste_harm, liste_sortiDesire):
    # -------------------------------------------------------------------------------------------------------------------
    vala0 = []
    valb0 = []
    vala1 = []
    valb1 = []
    vala2 = []
    valb2 = []
    vala3 = []
    valb3 = []
    vala4 = []
    valb4 = []
    for i in range(0, len(liste_harm)):
        dataHarOnlyImg = liste_harm[i]
        h0 = np.array(dataHarOnlyImg[0])
        h1 = np.array(dataHarOnlyImg[1])
        h2 = np.array(dataHarOnlyImg[2])
        h3 = np.array(dataHarOnlyImg[3])
        h4 = np.array(dataHarOnlyImg[4])
        vala0.append(np.sqrt(np.square(h0[0]) + np.square(h0[2])))
        valb0.append(np.sqrt(np.square(h0[1]) + np.square(h0[3])))
        vala1.append(np.sqrt(np.square(h1[0]) + np.square(h1[2])))
        valb1.append(np.sqrt(np.square(h1[1]) + np.square(h1[3])))
        vala2.append(np.sqrt(np.square(h2[0]) + np.square(h2[2])))
        valb2.append(np.sqrt(np.square(h2[1]) + np.square(h2[3])))
        vala3.append(np.sqrt(np.square(h3[0]) + np.square(h3[2])))
        valb3.append(np.sqrt(np.square(h3[1]) + np.square(h3[3])))
        vala4.append(np.sqrt(np.square(h4[0]) + np.square(h4[2])))
        valb4.append(np.sqrt(np.square(h4[1]) + np.square(h4[3])))
       # harm0.append(h0)
    moy = np.array(liste_mean)     # nous transformons en matrice colonne notre vecteur moyenne de donnee
    stdev = np.array(liste_stdev)  # nous transformons en matrice colonne notre vecteur moyenne de donnee
    path_saving = r'/Users/frankgiressegadjuinianga/Documents/Master2TSI_UBa/Dossier_projet/SortingBeansSystem_Using_ANN/myBeansDataset.xlsx'
    writer = pd.ExcelWriter(path_saving, engine='openpyxl')
    wb = writer.book
    df = pd.DataFrame({'Moy_ChB': moy[:,0],
                       'Moy_ChV': moy[:,1],
                       'Moy_ChR': moy[:,2],
                       'ET_ChB': stdev[:,0],
                       'ET_ChV': stdev[:,1],
                       'ET_ChR': stdev[:,2],
                       'coeff_a0': vala0,
                       'coeff_b0': valb0,
                       'coeff_a1': vala1,
                       'coeff_b1': valb1,
                       'coeff_a2': vala2,
                       'coeff_b2': valb2,
                       'coeff_a3': vala3,
                       'coeff_b3': valb3,
                       'coeff_a4': vala4,
                       'coeff_b4': valb4,
                       'sortie': liste_sortiDesire})
    df.to_excel(writer, index=False)
    wb.save(path_saving)
