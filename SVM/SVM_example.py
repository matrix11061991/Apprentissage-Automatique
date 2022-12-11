# Import the necessary modules

from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

# Load the iris dataset

iris = datasets.load_iris()

# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)

# Create an SVC object with the default hyperparameters

model = SVC()

# Train the model on the training data

model.fit(X_train, y_train)

# Use the trained model to make predictions on the test data

y_pred = model.predict(X_test)

# Evaluate the model's performance

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

