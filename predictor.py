from sklearn.linear_model import LinearRegression
import numpy as np
from scipy import stats
from pandas import Series
from statsmodels.tsa.arima_model import ARMA

def predict_next_value(timeseries, days_forward):
    x = create_timeline(len(timeseries))

    model = LinearRegression()
    model.fit(x, timeseries)

    #predict first val after end of timeseries
    X_predict = len(timeseries)-1 + days_forward
    X_predict = np.asarray(X_predict).reshape(-1,1)
    y_predict = np.asarray(model.predict(X_predict), 'int32')
    return y_predict

def predict_with_arma_model(timeseries, days_forward):
    X_predict = len(timeseries)-1 + days_forward
    timeseries_list = np.array(timeseries).tolist()
    model = ARMA(timeseries_list, order=(3, 1))
    model_fit = model.fit()
    y_predict = np.asarray(model_fit.predict(X_predict), 'int32')
    return y_predict

def create_timeline(length):
    x = []  # put your dates in here
    for i in range(length):
        x.append(i)
    X = np.asarray(x)
    X = X.reshape(-1,1)
    return X