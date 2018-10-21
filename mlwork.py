from sklearn import utils
import pandas as pd
import numpy as np
#from sklearn.neural_network import MLPClassifier
#from sklearn import linear_model
#from sklearn.preprocessing import PolynomialFeatures
#from sklearn.linear_model import LinearRegression
#from sklearn.pipeline import Pipeline
# import statsmodels.api as sm



X = pd.read_excel('Dataset.xlsx')
x = X.values
x_final = x[1:,:4]
y = x[1:,5:]
y = utils.column_or_1d(y.ravel(), warn=True)

x_final = (x_final - x_final.mean())/x_final.std()
y = (y - y.mean())/y.std()

print(x_final[:10])
print(y[:10])

# regressor = LinearRegression()  
# regressor.fit(x_final, y)

# y_pred = regressor.predict([[0.23705,-0.658374,-0.670155,-0.939774]])


#x_final = sm.add_constant(x_final)
##--------##

#clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(4, 1), random_state=1)
#clf = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',beta_1=0.9, beta_2=0.999, 
              #early_stopping=False,epsilon=1e-08, hidden_layer_sizes=(5, 2),learning_rate='constant', 
              #learning_rate_init=0.001,max_iter=200, momentum=0.9, n_iter_no_change=10,nesterovs_momentum=True,
              #power_t=0.5, random_state=1,shuffle=True, solver='lbfgs', tol=0.0001,validation_fraction=0.1, 
              #verbose=False, warm_start=False)
#model = Pipeline([('poly', PolynomialFeatures(degree=3)),('linear', LinearRegression(fit_intercept=False))])
#clf = MLPClassifier(hidden_layer_sizes=(15,), random_state=1, max_iter=1, warm_start=True)
#x_partitioned = [[i] for i in x_final][1:10]
#y_partitioned = [[j] for j in y][1:10]

#print(x_partitioned)
#print(y_partitioned)  
#clf.fit(, )
             
# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy

from flask import Flask
app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello World!"

# fix random seed for reproducibility
seed = 7
#numpy.random.seed(seed)
# load pima indians dataset
#dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = x_final #dataset[:,0:8]
Y = y #dataset[:,8]
# create model
model = Sequential()
model.add(Dense(3, input_dim=4, init='uniform', activation='relu'))
model.add(Dense(2, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=930, batch_size=1,  verbose=2)
# calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [x for x in predictions]
print(rounded)
