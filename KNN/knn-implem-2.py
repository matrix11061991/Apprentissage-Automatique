import numpy as np

# chargement des données d'entraînement et de test
X_train = ...
X_test = ...
y_train = ...
y_test = ...

# fonction de calcul de distance Euclidienne
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# initialisation des variables pour stocker les résultats
y_pred = []

# boucle sur chaque point de données de test
for x_test in X_test:
    # calcul de la distance entre le point de données de test et chaque point de données d'entraînement
    distances = [euclidean_distance(x_test, x_train) for x_train in X_train]
    
    # tri des distances en ordre croissant
    nearest_indices = np.argsort(distances)
    
    # sélection des k plus proches voisins
    k = 3
    nearest_indices = nearest_indices[:k]
    
    # utilisation de la majorité des labels de classe des k plus proches voisins pour prédire la classe
    nearest_labels = [y_train[i] for i in nearest_indices]
    y_pred.append(np.bincount(nearest_labels).argmax())
