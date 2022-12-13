import numpy as np

def knn(x, data, k):

  # calculer la distance entre x et chaque point de données

  distances = []

  for point in data:

    # utiliser la distance Euclidienne comme mesure de distance

    distance = np.linalg.norm(x - point)

    distances.append(distance)

  # trier les points de données en fonction de leur distance à x

  # et prendre les k plus proches voisins

  k_nearest = np.argsort(distances)[:k]
  # calculer la classe majoritaire parmi les k plus proches voisins

  classes = [data[i][1] for i in k_nearest]

  return max(set(classes), key=classes.count)
  
