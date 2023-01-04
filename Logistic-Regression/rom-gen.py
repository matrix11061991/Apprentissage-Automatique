from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Collectez et prétraitez les données de texte
texts = ['mon livre est un roman', 'j'aime lire des romans', 'je suis un grand fan de roman']
features = CountVectorizer().fit_transform(texts)

# Sélectionnez et entraînez un modèle d'apprentissage automatique
model = Pipeline([('vectorizer', CountVectorizer()), ('classifier', LogisticRegression())])
model.fit(features, labels)

# Générez de nouveaux textes de romans en utilisant le modèle entraîné
num_words = 10 # nombre de mots à générer
seed_text = "Je suis un grand fan de" # séquence de départ
generated_text = seed_text
for i in range(num_words):
    x = model.predict([generated_text])[0]
    generated_text += " " + x
print(generated_text)
