import matplotlib.pyplot as plot
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

d_x, d_y = datasets.load_diabetes(return_X_y=True)

d_x = d_x[:, np.newaxis, 2]

dc_x_train = d_x[:-20]
d_x_test = d_x[-20:]

d_y_train = d_y[:-20]
d_y_test = d_y[-20:]

regression = linear_model.LinearRegression()

regression.fit(diabetes_x_train, diabetes_y_train)

predictiion = regression.predict(diabetes_x_test)

