#!/usr/bin/python3
# Seed value
# Apparently you may use different seed values at each stage
seed_value= 2

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
#tf.random.set_seed(seed_value)
# for later versions:
tf.compat.v1.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
# for later versions:
# session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
# tf.compat.v1.keras.backend.set_session(sess)

from datetime import date
import numpy as np
import matplotlib.pyplot as plt

import data_reader
import predictor
import prediction_quality

print('Hello mister! I\'m your diviner')

#list of countries
countries = data_reader.european_countries()

#analysed dates
START_DATE  = date(2020, 2, 1)
END_DATE    = date(2020, 5, 26)


# Data set splitting for learn as 67%
SET_SPLIT_THRESHOLD = 0.67

# Model parameters
N_STEPS_BACKWARDS   = 7
N_STEPS_FORWARD     = 7
N_FEATURES          = 6
N_NEURONS           = 32

date_list   = data_reader.date_set_preparation(START_DATE, END_DATE)

(cases_c, cases_d, cases_r) = data_reader.read_covid_file(countries, date_list)

#date_test = data_reader.test_date(date(2020, 5, 26))

#(cases_c_v, cases_d_v, cases_r_v) = data_reader.read_covid_file(countries, date_test)

#cases_t = cases_c_v

print('Prediction:')
cases_p         = dict()
cases_p_linear  = dict()
for country in countries:
    print(country)
    # Differentiate signal
    dataset     = np.diff(cases_c[country])
    dataset_d   = np.diff(cases_d[country])
    dataset_r   = np.diff(cases_r[country])
    dataset_ca  = cases_c[country][1:]
    dataset_da  = cases_d[country][1:]
    dataset_ra  = cases_r[country][1:]
    # Diff values
    X_c, Y_c = predictor.split_sequence(dataset, N_STEPS_BACKWARDS, N_STEPS_FORWARD)
    X_d, Y_d = predictor.split_sequence(dataset_d, N_STEPS_BACKWARDS, N_STEPS_FORWARD)
    X_r, Y_r = predictor.split_sequence(dataset_r, N_STEPS_BACKWARDS, N_STEPS_FORWARD)
    # Accumulated values
    X_ca, Y_ca = predictor.split_sequence(dataset_ca, N_STEPS_BACKWARDS, N_STEPS_FORWARD)
    X_da, Y_da = predictor.split_sequence(dataset_da, N_STEPS_BACKWARDS, N_STEPS_FORWARD)
    X_ra, Y_ra = predictor.split_sequence(dataset_ra, N_STEPS_BACKWARDS, N_STEPS_FORWARD)
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
    # Split sequences for test as only last sequence
    #X_learn, Y_learn = X[:-1, :], Y[:-1, :]
    #X_test, Y_test = X[-1:, :], Y[-1:, :]
    # Prepare model, train it and make test prediction
    model = predictor.lstm_model_create(N_NEURONS, N_STEPS_BACKWARDS, N_FEATURES, N_STEPS_FORWARD)
    X = X.reshape((X.shape[0], X.shape[1], N_FEATURES))
    model.fit(X, Y, epochs=200, verbose=0)
    # Data for prediction for future
    X_pred = np.array(X[-1, :, :])
    X_pred = X_pred.reshape((1, X_pred.shape[0], N_FEATURES))
    Y_predict = model.predict(X_pred, verbose=0)
    Y_predict = Y_predict.clip(min=0)

    # Plot results
    dataset = cases_c[country]
    last_value = dataset[-1]
    prediction = np.zeros(N_STEPS_FORWARD)
    Y_predict = Y_predict.flatten()
    Y_predict = np.round(Y_predict)
    for i in range(N_STEPS_FORWARD):
        prediction[i] = last_value + Y_predict[i]
        last_value = prediction[i]

    t1 = np.zeros_like(dataset)
    for i in range(len(dataset)):
        t1[i] = i

    t2 = np.zeros_like(prediction)
    t2[0] = t1[-1] + 1
    for i in range(len(prediction)):
        if i != 0:
            t2[i] = t2[i - 1] + 1


    # Calculate linear response
    pred_linear = np.zeros_like(prediction)
    dataset_linear = dataset[-10:]
    for i in range(len(prediction)):
        pred_linear[i] = predictor.predict_next_value(dataset_linear, i + 1)

    plt.plot(t1, dataset, 'b.')
    plt.plot(t2, prediction, 'r.')
    plt.plot(t2, pred_linear, 'g.')
    plt.show()
    dataset = np.diff(cases_c[country])
    prediction2 = Y_predict
    t1 = t1[1:]
    plt.plot(t1, dataset, 'b.')
    plt.plot(t2, prediction2, 'r.')
    plt.show()

    print('Prediction CNN: ', prediction[-1])
    print('Prediction linear: ', pred_linear[-1])

    cases_p[country]        = prediction[-1]
    cases_p_linear[country] = pred_linear[-1]

#print('Prediction quality CNN:')

#quality_poland  = prediction_quality.quality_country(cases_p['Poland'], cases_t['Poland'])
#quality_europe  = prediction_quality.quality_all(cases_p, cases_t)

#print('for Poland: ', quality_poland, '%, for Europe: ', quality_europe, '%')

#print('Prediction quality Linear:')

#quality_poland  = prediction_quality.quality_country(cases_p_linear['Poland'], cases_t['Poland'])
#quality_europe  = prediction_quality.quality_all(cases_p_linear, cases_t)

#print('for Poland: ', quality_poland, '%, for Europe: ', quality_europe, '%')

print("Done!")