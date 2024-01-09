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
## Le code complet:
```python
# Importation des bibliothèques nécessaires
import numpy as np  # NumPy est utilisé pour les opérations numériques
import matplotlib.pyplot as plt  # Matplotlib est utilisé pour la visualisation
from sklearn.model_selection import train_test_split  # Fonction pour diviser les données en ensembles d'entraînement et de test
from sklearn.linear_model import LinearRegression  # Modèle de régression linéaire de scikit-learn
from sklearn.metrics import mean_squared_error  # Métrique d'erreur pour évaluer les performances du modèle

# Génération de données synthétiques pour l'exemple
np.random.seed(42)  # Fixer la graine aléatoire pour la reproductibilité
X = 2 * np.random.rand(100, 1)  # Générer 100 valeurs aléatoires entre 0 et 2
y = 4 + 3 * X + np.random.randn(100, 1)  # Générer des étiquettes y avec une relation linéaire et du bruit gaussien

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialiser le modèle de régression linéaire
model = LinearRegression()

# Entraîner le modèle sur l'ensemble d'entraînement
model.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Calculer l'erreur quadratique moyenne entre les prédictions et les vraies valeurs
mse = mean_squared_error(y_test, y_pred)

# Afficher les résultats
print("Coefficient (pente) du modèle :", model.coef_[0][0])
print("Terme constant du modèle :", model.intercept_[0])
print("Erreur quadratique moyenne sur l'ensemble de test :", mse)

# Tracer la ligne de régression sur les données
plt.scatter(X_test, y_test, color='black', label='Données réelles')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Régression linéaire')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Régression linéaire avec scikit-learn')
plt.legend()
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
                                                       



