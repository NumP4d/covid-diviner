from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from random import random
import numpy as np

def predict_next_value(timeseries, days_forward):
    x = create_timeline(len(timeseries))

    model = LinearRegression()
    model.fit(x, timeseries)

    #predict first val after end of timeseries
    X_predict = len(timeseries)-1 + days_forward
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

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, Y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        Y.append(seq_y)
    return np.array(X), np.array(Y)

def create_train_test_set(X, Y, probability_threshold):
    i_learn = []
    i_test  = []
    for i in range(len(Y)):
        if random() < probability_threshold:
            i_learn.append(i)
        else:
            i_test.append(i)
    #X_learn = [X[i] for i in i_learn]
    #Y_learn = [Y[i] for i in i_learn]
    #X_test  = [X[i] for i in i_test]
    #Y_test  = [Y[i] for i in i_test]
    return X[i_learn], Y[i_learn], X[i_test], Y[i_test]

# Create neural network model and train it
def lstm_model_create(n_neurons, n_steps, n_features):
    # define model
    model = Sequential()
    model.add(LSTM(n_neurons, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def lstm_model_train(model, X_learn, Y_learn):
    model.fit(X_learn, Y_learn, epochs=200, verbose=0)
    return model