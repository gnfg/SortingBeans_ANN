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
# ----------------- importation des données
import pandas
import performance_metrique as perfm
from sklearn import model_selection
#from sklearn.neural_network import MLPClassifier
from sklearn import svm

# ----------------- lescture de notre base de donnees
path = "/Users/frankgiressegadjuinianga/Documents/Master2TSI_UBa/Dossier_projet/SortingBeansSystem_Using_ANN/myBeansDataset.xlsx"
data_classA = pandas.read_excel(path)
X = data_classA.values[:, 0:14] # features(descripteurs)
Y = data_classA.values[:, 15] # targets(cible)
"""
Important :  nous utilisons dans cette partie une technique de classification pour separer au mieux nos echantillons
-----> le perceptron multicouche (PMC)
            >> utiliser dans les espaces de dimension faible (2 caracteristiques / features )
-----> les machines a vecteur de support (SVM) 
            >> efficace dans les espaces de grande dimension ( >2 features)
-----> etc..
"""
# ---------------- subdivision en donnee d'apprentissage et de test
XTrain, XTest, YTrain, YTest = model_selection.train_test_split(X, Y, train_size=60)
rna = svm.SVC(kernel='linear', degree=3, gamma='scale')  # noyau lineair
#rna = svm.SVC(kernel='rbf', random_state=0, gamma=0.20, C=10.0) # noyau rbf
#rna = MLPClassifier(hidden_layer_size=(5,), activation = "logistic", solver="lbfgs")  # perceptron multicouche avec 5 neurones dans la cc

rna.fit(XTrain, YTrain) #  apprentissage de notre systeme
pred = rna.predict(XTest) #  prédiction sur l'échantillon test
Yreel = YTest
print(" pred:", pred)

#---------------- mesure des performances
perfm.Perfermance(Yreel, pred)
