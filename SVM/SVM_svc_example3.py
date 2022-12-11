from sklearn.svm import SVC

# Load data

X_train, y_train, X_test, y_test = ...

# Create an instance of the SVM class

svm = SVC()

# Fit the model on the training data

svm.fit(X_train, y_train)

# Make predictions on the test data

predictions = svm.predict(X_test)

