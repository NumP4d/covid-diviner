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
START_DATE  = date(2020, 4, 1)
END_DATE    = date(2020, 6, 14)

# Model parameters
N_STEPS_BACKWARDS   = 7
N_STEPS_FORWARD     = 7
N_FEATURES          = 6
N_NEURONS           = 32

N_MONTE_CARLO       = 5

# Choose model
model_type = 'neural_network'
#model_type = 'linear'

date_list   = data_reader.date_set_preparation(START_DATE, END_DATE)

(cases_c, cases_d, cases_r) = data_reader.read_covid_file(countries, date_list)

acc_predict = dict()
for country in countries:
    # Dataset creation and splitting - all for training set
    (X_learn, Y_learn, X_test, Y_test)              \
        = predictor.create_country_train_test_set   \
        (country, cases_c, cases_d, cases_r,        \
        N_STEPS_BACKWARDS, N_STEPS_FORWARD)

    acc_predict[country] = 0
    for i in range(N_MONTE_CARLO):
        if (model_type == 'neural_network'):
            # Model creation and training for 7 days ahead accumulated
            model = predictor.lstm_model_create(N_NEURONS, N_STEPS_BACKWARDS, N_FEATURES, 1)
            model.fit(X_learn, Y_learn, epochs=100, verbose=0)

        # Model prediction for country
        Y_predict = model.predict(X_test, verbose=0)
        # Limit values in prediction
        Y_predict = Y_predict.clip(min=0)

        # Translate prediction for absolute value 7 days in future (accumulated)
        predict_value = predictor.translate_prediction   \
            (X_test, Y_predict)

        # Round accumulated prediction
        predict_value = np.round(predict_value)

        # Calculate real value for calculating quality factor
        real_value = predictor.translate_prediction     \
            (X_test, Y_test)

        acc_predict[country] += sum(np.abs(predict_value - real_value) / real_value) * 100 / 5

    acc_predict[country] /= N_MONTE_CARLO

    print("Mean accuracy for ", country, ": ", acc_predict[country], "%")

model_quality = sum(acc_predict.values()) / len(acc_predict)

print("Model quality prediction: ", model_quality, "%")