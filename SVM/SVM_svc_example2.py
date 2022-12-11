from sklearn import svm

# Load the data

X = [[0, 0], [1, 1]]

y = [0, 1]

# Create the SVM model

model = svm.SVC()

# Train the model

model.fit(X, y)

# Make predictions

predictions = model.predict([[2, 2], [3, 3]])

