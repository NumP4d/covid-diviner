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

def predict_linear(timeseries, days_forward):
    for row in timeseries:
        x = create_timeline(len(timeseries))

        model = LinearRegression()
        model.fit(x, timeseries)

        #predict first val after end of timeseries
        X_predict = []
        for i in range(days_forward):
            X_predict.append(len(timeseries) + i)
        X_predict = np.asarray(X_predict).reshape(-1,1)
        Y_predict = np.asarray(model.predict(X_predict), 'int32')
    return Y_predict


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

def create_countries_train_test_set(countries, cases_c, cases_d, cases_r, probability_threshold, N_STEPS_BACKWARDS, N_STEPS_FORWARD):
    N_FEATURES = 6
    # Dataset preparation
    X_learn = []
    Y_learn = []
    X_test  = []
    Y_test  = []
    for country in countries:
        # Differentiate signal
        dataset_c   = np.diff(cases_c[country])
        dataset_d   = np.diff(cases_d[country])
        dataset_r   = np.diff(cases_r[country])
        dataset_ca  = cases_c[country][1:]
        dataset_da  = cases_d[country][1:]
        dataset_ra  = cases_r[country][1:]
        # Diff values
        X_c, Y_c = split_sequence(dataset_c, N_STEPS_BACKWARDS, N_STEPS_FORWARD)
        X_d, Y_d = split_sequence(dataset_d, N_STEPS_BACKWARDS, N_STEPS_FORWARD)
        X_r, Y_r = split_sequence(dataset_r, N_STEPS_BACKWARDS, N_STEPS_FORWARD)
        # Accumulated values
        X_ca, Y_ca = split_sequence(dataset_ca, N_STEPS_BACKWARDS, N_STEPS_FORWARD)
        X_da, Y_da = split_sequence(dataset_da, N_STEPS_BACKWARDS, N_STEPS_FORWARD)
        X_ra, Y_ra = split_sequence(dataset_ra, N_STEPS_BACKWARDS, N_STEPS_FORWARD)
        # Reshape to 3D signals
        X_c = X_c.reshape((X_c.shape[0], X_c.shape[1], 1))
        X_d = X_d.reshape((X_d.shape[0], X_d.shape[1], 1))
        X_r = X_r.reshape((X_r.shape[0], X_r.shape[1], 1))
        X_ca = X_ca.reshape((X_ca.shape[0], X_ca.shape[1], 1))
        X_da = X_da.reshape((X_da.shape[0], X_da.shape[1], 1))
        X_ra = X_ra.reshape((X_ra.shape[0], X_ra.shape[1], 1))
        # Build one huge 3D array of super-hyper-parameters for CNN
        X = np.concatenate((X_c, X_d, X_r, X_ca, X_da, X_ra), axis=2)
        # Output (response) is diff values of confirmed cases
        Y = Y_c

        # Split country to training set or test set
        if (random.random() > probability_threshold):
            # Put at the end of test set
            X_test.append(X)
            Y_test.append(Y)
        else:
            # Put at the end of training set
            X_learn.append(X)
            Y_learn.append(Y)

    # Stack lists for one big matrices
    X_learn = np.vstack(X_learn)
    Y_learn = np.vstack(Y_learn)
    if (len(X_test) > 0):
        X_test = np.vstack(X_test)
        Y_test = np.vstack(Y_test)

    # Datasets reshaping
    X_learn = X_learn.reshape((X_learn.shape[0], X_learn.shape[1], N_FEATURES))
    if (len(X_test) > 0):
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], N_FEATURES))

    #np.random.shuffle(X_countries)

    return X_learn, Y_learn, X_test, Y_test

def create_countries_predition_set(countries, cases_c, cases_d, cases_r, N_STEPS_BACKWARDS):
    N_FEATURES = 6
    # Dataset preparation
    X_predict = dict()
    for country in countries:
        # Differentiate signal
        dataset_c   = np.diff(cases_c[country])
        dataset_d   = np.diff(cases_d[country])
        dataset_r   = np.diff(cases_r[country])
        dataset_ca  = cases_c[country][1:]
        dataset_da  = cases_d[country][1:]
        dataset_ra  = cases_r[country][1:]

        # Diff values
        X_c = dataset_c[-N_STEPS_BACKWARDS:]
        X_d = dataset_d[-N_STEPS_BACKWARDS:]
        X_r = dataset_r[-N_STEPS_BACKWARDS:]
        # Accumulated values
        X_ca = dataset_ca[-N_STEPS_BACKWARDS:]
        X_da = dataset_da[-N_STEPS_BACKWARDS:]
        X_ra = dataset_ra[-N_STEPS_BACKWARDS:]

        # Reshape to 3D signals
        X_c = X_c.reshape((1, X_c.shape[0], 1))
        X_d = X_d.reshape((1, X_d.shape[0], 1))
        X_r = X_r.reshape((1, X_r.shape[0], 1))
        X_ca = X_ca.reshape((1, X_ca.shape[0], 1))
        X_da = X_da.reshape((1, X_da.shape[0], 1))
        X_ra = X_ra.reshape((1, X_ra.shape[0], 1))

        # Build one huge 2D array of super-hyper-parameters for CNN
        X = np.concatenate((X_c, X_d, X_r, X_ca, X_da, X_ra), axis=2)
        X_predict[country] = X.reshape((X.shape[0], X.shape[1], N_FEATURES))

    return X_predict

def translate_prediction(X, Y_predict):
    prediction = np.zeros(len(Y_predict))
    for i in range(len(Y_predict)):
        c_last_acc      = X[i, -1, 3]
        prediction[i]   = c_last_acc + np.sum(Y_predict[i])

    return prediction


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
    model.add(Dense(50 * n_features, activation='relu'))
    model.add(Dense(n_future, activation='softplus'))
    model.compile(optimizer='adam', loss='mse')
    return model

def lstm_model_train(model, X_learn, Y_learn):
    model.fit(X_learn, Y_learn, epochs=200)
    return model
