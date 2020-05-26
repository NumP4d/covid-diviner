from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.optimizers import RMSprop
import random
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
def split_sequence(sequence, n_steps_backward, n_steps_forward):
    X, Y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_backward + n_steps_forward
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x = sequence[i:(end_ix - n_steps_forward)]
        seq_y = sequence[(end_ix - n_steps_forward):end_ix]
        X.append(seq_x)
        Y.append(seq_y)
    return np.array(X), np.array(Y)

def create_train_test_set(X, Y, probability_threshold):
    i_learn = []
    i_test  = []
    #random.seed(7)
    for i in range(len(Y)):
        if random.random() < probability_threshold:
            i_learn.append(i)
        else:
            i_test.append(i)
    return X[i_learn], Y[i_learn], X[i_test], Y[i_test]

# Create neural network model and train it
def lstm_model_create(n_neurons, n_steps, n_features, n_future):
    second_layer_neurons = (n_neurons / 2)
    second_layer_neurons = np.int32(second_layer_neurons)
    # define model
    #model = Sequential()
    #model.add(LSTM(n_neurons, return_sequences=True, input_shape=(n_steps, n_features)))
    #model.add(LSTM(second_layer_neurons, activation='relu'))
    #model.add(Dense(n_future, activation='relu'))
    #model.add(Dense(n_future))
    #model.compile(optimizer=RMSprop(clipvalue=1.0), loss='mae')
    #model.compile(optimizer='adam', loss='mse')
    # define model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(300, activation='relu'))
    model.add(Dense(n_future))
    model.compile(optimizer='adam', loss='mse')
    return model

def lstm_model_train(model, X_learn, Y_learn):
    model.fit(X_learn, Y_learn, epochs=200)
    return model
