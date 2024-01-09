# Fiche d'Activité Pratique : Travaux Pratiques sur la régression linéaire avec scikit-learn
## Objectif
Mettre en œuvre la régression linéaire en utilisant la bibliothèque Scikit-Learn pour résoudre des problèmes réels.

## Prérequis  
* Python installé sur la machine
* Les bibliothèques Scikit-Learn, NumPy et Matplotlib.
## Travail Pratique : Prédiction des Prix Immobiliers
**Étape 1: Introduction**  
Dans ce TP, nous allons travailler sur un cas d'utilisation pratique : prédire les prix immobiliers en fonction de certaines caractéristiques. 
Pour cela, nous utiliserons la régression linéaire avec Scikit-Learn.  
**Étape 2: Tâches à Réaliser**  
**a. Importation des Bibliothèques :**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
```
**b. Chargement des Données :**
```python
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
```
**c. Exploration des Données :**
```python
print(housing.feature_names)
print(housing.data.shape)
print(housing.target.shape)
```
**d. Prétraitement des Données :**
```python
# Ajouter le code pour le prétraitement des données ici
```
**e. Division des Données :**
```python
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target, test_size=0.2, random_state=42)
```
**f. Entraînement du Modèle :**
```python
model = LinearRegression()
model.fit(X_train, y_train)
```
**g. Évaluation du Modèle :**
```python
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse}")
```
**h. Prédiction :**
```python
# Ajouter le code pour la prédiction sur de nouvelles données ici
```
**i. Visualisation des Résultats :**
```python
plt.scatter(y_test, y_pred)
plt.xlabel("Prix Réel")
plt.ylabel("Prix Prédit")
plt.title("Régression Linéaire - Prédiction des Prix Immobiliers")
plt.show()
```
## Conclusion
En réalisant ces travaux pratiques, vous aurez acquis une expérience pratique de l'application de la régression linéaire avec Scikit-Learn sur un problème réel de prédiction des prix immobiliers. Cette activité vous fournira des compétences précieuses dans l'utilisation de modèles linéaires pour résoudre des problèmes du monde réel.

## License
```sh
                                                     __  __       _        _          _______             
                                                    |  \/  |     | |      (_)        |__   __|            
                                                    | \  / | __ _| |_ _ __ ___  __      | | ___ _ __ __ _ 
                                                    | |\/| |/ _` | __| '__| \ \/ /      | |/ _ \ '__/ _` |
                                                    | |  | | (_| | |_| |  | |>  <       | |  __/ | | (_| |
                                                    |_|  |_|\__,_|\__|_|  |_/_/\_\      |_|\___|_|  \__,_|   🇲🇬
```
                                                       



