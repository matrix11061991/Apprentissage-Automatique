# Fiche d'Activité Pratique : K Plus Proches Voisins (KNN) avec Scikit-Learn
## Objectif
Comprendre et mettre en œuvre l'algorithme des **K Plus Proches Voisins (KNN)** à l'aide de la bibliothèque Scikit-Learn en Python.
## Prérequis  
* Python installé sur la machine
* Les bibliothèques Scikit-Learn, NumPy et Matplotlib.
## Introduction
L'algorithme des **K Plus Proches Voisins (KNN)** est une méthode d'apprentissage supervisé utilisée pour la classification et la régression. Il se base sur le principe que des points similaires se trouvent généralement dans des zones proches les unes des autres. KNN attribue une classe à un point de données en fonction des classes majoritaires parmi ses k voisins les plus proches.
**I. Qu'est-ce que les K Plus Proches Voisins (KNN) ?**
L'idée fondamentale de KNN est de trouver les k voisins les plus proches d'un point de données donné et de prendre une décision en fonction des classes majoritaires parmi ces voisins. La distance entre les points peut être mesurée de différentes manières, souvent par la distance euclidienne.

L'algorithme KNN peut être utilisé pour la classification et la régression :
- **Classification :** La classe majoritaire parmi les k voisins est attribuée au point de données.
- **Régression :** La valeur moyenne ou médiane des k voisins est attribuée au point de données.
  
**II. Mise en Pratique avec Scikit-Learn :**  
**Installation de Scikit-Learn :**
  Assurez-vous d'avoir Scikit-Learn installé. Sinon, installez-le via la commande : 
```sh
pip install scikit-learn
```
**Importation des Bibliothèques :**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
```
**Génération de Données :**
```python
np.random.seed(42)
X = 2 * np.random.rand(100, 2)
y = (X[:, 0] + X[:, 1] > 1).astype(int)
```
**Entraînement du Modèle :**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

k_value = 3
model = KNeighborsClassifier(n_neighbors=k_value)
model.fit(X_train, y_train)
```
**Prédiction et Évaluation :**
```python
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{confusion_mat}")
```
## Conclusion
Les **K Plus Proches Voisins** sont une méthode simple mais efficace pour la classification et la régression. Scikit-Learn offre des outils puissants pour mettre en œuvre cet algorithme en Python. En pratiquant cette activité, vous avez maintenant les bases pour explorer davantage les concepts de KNN et son application dans le domaine de l'apprentissage automatique.
## License
```sh
                                                     __  __       _        _          _______             
                                                    |  \/  |     | |      (_)        |__   __|            
                                                    | \  / | __ _| |_ _ __ ___  __      | | ___ _ __ __ _ 
                                                    | |\/| |/ _` | __| '__| \ \/ /      | |/ _ \ '__/ _` |
                                                    | |  | | (_| | |_| |  | |>  <       | |  __/ | | (_| |
                                                    |_|  |_|\__,_|\__|_|  |_/_/\_\      |_|\___|_|  \__,_|   🇲🇬
```
                                                       




