from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

def predict_poly_regression(timeseries, days_forward, deg):
    x = create_timeline(len(timeseries))

    poly_model = PolynomialFeatures(degree=deg)
    X_poly = poly_model.fit_transform(x)
    lin_model = LinearRegression()
    lin_model.fit(X_poly, timeseries)

    #predict first val after end of timeseries
    X_predict = len(timeseries)-1 + days_forward
    X_predict = np.asarray(X_predict).reshape(-1,1)
    y_predict = np.asarray(lin_model.predict(poly_model.fit_transform(X_predict)), 'int32')
    return y_predict

def create_timeline(length):
    x = []  # put your dates in here
    for i in range(length):
        x.append(i)
    X = np.asarray(x)
    X = X.reshape(-1,1)
    return X
