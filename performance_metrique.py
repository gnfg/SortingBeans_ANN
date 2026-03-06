
# ----------------- importation des données
import pandas
import matplotlib.pyplot as plt
from sklearn import metrics

def Perfermance(val_reel, val_pred):
    print(metrics.confusion_matrix(val_reel, val_pred))
    print("Taux erreur = " + str(1 - metrics.accuracy_score(val_reel, val_pred)))
    # ---------------- affichage des donnees
    plt.plot(val_reel, 'b', val_pred, 'r')
    plt.show()