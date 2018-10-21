from flask import Flask, request, flash, url_for, redirect, render_template, abort, jsonify, make_response
import pandas as pd
from sklearn import utils
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from flask_cors import CORS


# TO START APP, TYPE: gunicorn test:app

app = Flask(__name__)
# define a predict function as an endpoint 

CORS(app)

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


app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict', methods=['GET'])
def predict():
    if not request.args:
        abort(400)

    print(request.args)
    income = request.args.get("income")
    taxesAndInsurance = request.args.get("taxesAndInsurance")
    monthlyExpenses = request.args.get("monthlyExpenses")
    age = request.args.get("age")
    print(income)
    print(taxesAndInsurance)
    print(monthlyExpenses)
    print(age)

    prediction = regressor.predict([[float(income), float(taxesAndInsurance), float(monthlyExpenses), float(age)]])
    print("prediction", predict)
    
    return jsonify({"prediction" : prediction[0]})

