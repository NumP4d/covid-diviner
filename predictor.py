from sklearn.linear_model import LinearRegression
import numpy as np

def predict_next_value(timeseries):
   
    x = create_timeline(len(timeseries))
    y = np.asarray(timeseries)
    y=y.reshape(-1,1)


    model = LinearRegression()
    model.fit(x, y)

    #predict first val after end of timeseries 
    X_predict = len(timeseries)
    X_predict = np.asarray(X_predict).reshape(-1,1)
    y_predict = np.asarray(model.predict(X_predict), 'int32')
    return y_predict


def create_timeline(length):
    x = []  # put your dates in here
    for i in range(length):
        x.append(i)
    X = np.asarray(x)
    X = X.reshape(-1,1)
    return X