# Fiche d'Activité Pratique : Support Vector Machines (SVMs) avec Scikit-Learn
## Objectif
 Comprendre et mettre en œuvre l'algorithme des Support Vector Machines (SVMs) à l'aide de la bibliothèque Scikit-Learn en Python.
## Prérequis  
* Python installé sur la machine
* Les bibliothèques Scikit-Learn, NumPy et Matplotlib.
## Introduction
Les Support Vector Machines (SVMs) sont une classe d'algorithmes d'apprentissage supervisé utilisés pour la classification et la régression. L'objectif principal de SVM est de trouver un hyperplan optimal dans un espace de caractéristiques qui sépare le mieux possible les différentes classes.
**I. Qu'est-ce que les Support Vector Machines (SVMs) ?**
L'idée centrale des SVMs est de trouver un hyperplan qui maximise la marge entre les classes. La marge est définie comme la distance entre l'hyperplan et les points les plus proches de chaque classe, appelés vecteurs de support. Les SVMs sont particulièrement efficaces dans des espaces de grande dimension et sont capables de gérer des données non linéaires grâce à l'utilisation de noyaux.

En termes simples, pour un problème de classification binaire, l'hyperplan est une frontière de décision qui sépare les classes de manière optimale.
  
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
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
```
**Génération de Données :**
```python
np.random.seed(42)
X, y = datasets.make_classification(n_samples=100, n_features=2, n_classes=2, n_clusters_per_class=1, n_redundant=0)
```
**Entraînement du Modèle :**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(kernel='linear')
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
Les **Support Vector Machines** sont des outils puissants pour la classification et la régression, en particulier dans des contextes de données complexes. Scikit-Learn fournit une implémentation robuste de SVMs en Python. En pratiquant cette activité, vous avez maintenant les bases pour explorer davantage les concepts de SVM et son application dans le domaine de l'apprentissage automatique.
## License
```sh
                                                     __  __       _        _          _______             
                                                    |  \/  |     | |      (_)        |__   __|            
                                                    | \  / | __ _| |_ _ __ ___  __      | | ___ _ __ __ _ 
                                                    | |\/| |/ _` | __| '__| \ \/ /      | |/ _ \ '__/ _` |
                                                    | |  | | (_| | |_| |  | |>  <       | |  __/ | | (_| |
                                                    |_|  |_|\__,_|\__|_|  |_/_/\_\      |_|\___|_|  \__,_|   🇲🇬
```
                                                       




