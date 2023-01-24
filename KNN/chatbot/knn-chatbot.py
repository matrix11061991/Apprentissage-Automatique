from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Charger les données d'entraînement
data = pd.read_csv("chatbot_data.csv")
X = data["Question"]
y = data["Réponse"]

# Tokenization et Vectorisation
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Entraînement 
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

# Prédiction
def predict_response(question):
    question = vectorizer.transform([question])
    prediction = knn.predict(question)
    return prediction[0]

# Interaction avec l'utilisateur
while True:
    question = input("Vous : ")
    if question.lower() == "quit":
        break
    print("Bot : ", predict_response(question))
