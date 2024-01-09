# Fiche d'Activité Pratique : Régression Logistique avec Scikit-Learn
## Objectif
Comprendre et mettre en œuvre la régression logistique à l'aide de la bibliothèque Scikit-Learn en Python.
## Prérequis  
* Python installé sur la machine
* Les bibliothèques Scikit-Learn, NumPy et Matplotlib.
## Introduction
La régression logistique est une technique d'apprentissage supervisé utilisée pour la classification. Contrairement à la régression linéaire qui prédit des valeurs continues, la régression logistique est utilisée pour prédire des probabilités d'appartenance à une classe. Elle est largement utilisée dans les problèmes de classification binaire.
**I. Qu'est-ce que la Régression Logistique ?**
La régression logistique modélise la probabilité qu'une instance appartienne à une classe particulière. L'équation de la régression logistique est donnée par la fonction logistique (ou sigmoïde) :
P(Y=1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 \cdot X_1 + \beta_2 \cdot X_2 + \ldots + \beta_n \cdot X_n)}}
Où :
- P(Y=1) est la probabilité que l'instance appartienne à la classe 1.
- e est la base du logarithme naturel.
- \beta_0, \beta_1,..., \beta_n ont les coefficients du modèle.
- X_1, X_2,..., X_n  sont les caractéristiques de l'instance.
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
```
**Génération de Données :**
```python
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = (4 + 3 * X + np.random.randn(100, 1) > 6).astype(int)
```
**Entraînement du Modèle :**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
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
La **régression logistique** est une méthode puissante pour les problèmes de classification. Scikit-Learn fournit des outils efficaces pour mettre en œuvre cette technique en Python. En pratiquant cette activité, vous avez maintenant les bases pour explorer davantage les concepts de la régression logistique et son application dans le domaine de l'apprentissage automatique.
## License
```sh
                                                     __  __       _        _          _______             
                                                    |  \/  |     | |      (_)        |__   __|            
                                                    | \  / | __ _| |_ _ __ ___  __      | | ___ _ __ __ _ 
                                                    | |\/| |/ _` | __| '__| \ \/ /      | |/ _ \ '__/ _` |
                                                    | |  | | (_| | |_| |  | |>  <       | |  __/ | | (_| |
                                                    |_|  |_|\__,_|\__|_|  |_/_/\_\      |_|\___|_|  \__,_|   🇲🇬
```
                                                       




