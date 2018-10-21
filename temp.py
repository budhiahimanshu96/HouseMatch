from sklearn import utils
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline



X = pd.read_excel('Dataset.xlsx')
x = X.values
x_final = x[1:,:4]
y = x[1:,5:]
y = utils.column_or_1d(y.ravel(), warn=True)

#x_final = (x_final - x_final.mean())/x_final.std()
#y = (y - y.mean())/y.std()

regressor = LinearRegression()  
regressor.fit(x_final, y)

#y_pred = regressor.predict([[24172,5801.28,5559.56,28]])

x_final_normalized = (x_final - x_final.mean())/x_final.std()
y_normalized = (y - y.mean())/y.std()
regressor_norm = LinearRegression()  
regressor_norm.fit(x_final_normalized, y_normalized)

X_test = pd.read_excel('Dataset_test.xlsx')
x_test = X_test.values
x_final_test = x_test[1:,:4]
y_test = x_test[1:,5:]
y_test = utils.column_or_1d(y_test.ravel(), warn=True)


y_pred_test = regressor.predict([[69137,14518.8,33877.1,26]])
print(y_pred_test - y_test[0])

