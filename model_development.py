#!/usr/bin/python3
# Seed value
# Apparently you may use different seed values at each stage
seed_value= 3

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

# Monte carlo simulation and training
N_MONTE_CARLO       = 1

# Choose model
model = 'neural_network'
#model = 'linear'

date_list   = data_reader.date_set_preparation(START_DATE, END_DATE)

(cases_c, cases_d, cases_r) = data_reader.read_covid_file(countries, date_list)

mean_quality = 0
for i in range(N_MONTE_CARLO):
    print("Iteration ", i)

    # Dataset creation and splitting
    (X_learn, Y_learn, X_test, Y_test)                              \
        = predictor.create_countries_train_test_set                 \
        (countries, cases_c, cases_d, cases_r, SET_SPLIT_THRESHOLD, \
        N_STEPS_BACKWARDS, N_STEPS_FORWARD)

    if (model == 'neural_network'):
        # Model creation, training and prediction for test sequence
        model = predictor.lstm_model_create(N_NEURONS, N_STEPS_BACKWARDS, N_FEATURES, N_STEPS_FORWARD)
        model.fit(X_learn, Y_learn, epochs=200, verbose=1)
        Y_predict = model.predict(X_test, verbose=0)
        # Limit values in prediction
        Y_predict = Y_predict.clip(min=0)
    elif (model == 'linear'):
        timeseries = X_test[:, 3, :]
        timeseries = timeseries.reshape(X_test.shape[0], X_test.shape[2], 1)
        acc_predict = np.zeros(len(Y_test))
        for i in range(len(acc_predict)):
            acc_predict[i] = predictor.predict_next_value(timeseries[i], N_STEPS_FORWARD)

    # Translate prediction for absolute value 7 days in future (accumulated)
    if (model != 'linear'):
        acc_predict = predictor.translate_prediction(X_test, Y_predict)
    acc_test    = predictor.translate_prediction(X_test, Y_test)

    # Round accumulated prediction
    acc_predict = np.round(acc_predict)

    # Calculate quality index for model
    quality = prediction_quality.calculate(acc_test, acc_predict)

    # Accumulate quality index
    mean_quality += quality


# Calculate mean
mean_quality /= N_MONTE_CARLO

print("Mean quality for model is calculated as ", mean_quality)

print("Done!")