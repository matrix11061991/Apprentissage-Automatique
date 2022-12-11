from sklearn import svm

# Sample data (X) and labels (y)

X = [[0, 0], [1, 1]]

y = [0, 1]

# Create a SVM model with linear kernel

model = svm.SVC(kernel='linear')

# Train the model on the sample data

model.fit(X, y)

# Use the trained model to make predictions on new data

predictions = model.predict([[2, 2], [3, 3]])

