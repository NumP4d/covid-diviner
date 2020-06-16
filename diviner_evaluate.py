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
END_DATE    = date(2020, 6, 7)

# Model parameters
N_STEPS_BACKWARDS   = 10
N_STEPS_FORWARD     = 8
N_FEATURES          = 6
N_NEURONS           = 32

# Choose model
model_type = 'neural_network'
#model_type = 'linear'

# enable plotting
plotting = True

date_list   = data_reader.date_set_preparation(START_DATE, END_DATE)

(cases_c, cases_d, cases_r) = data_reader.read_covid_file(countries, date_list)

prediction = dict()
for country in countries:
    # Dataset creation and splitting - all for training set
    (X_learn, Y_learn, X_predict)              \
        = predictor.create_country_prediction_set   \
        (country, cases_c, cases_d, cases_r,        \
        N_STEPS_BACKWARDS, N_STEPS_FORWARD)

    if (model_type == 'neural_network'):
        # Model creation and training for 7 days ahead accumulated
        model = predictor.lstm_model_create(N_NEURONS, N_STEPS_BACKWARDS, N_FEATURES, 1)
        model.fit(X_learn, Y_learn, epochs=100, verbose=0)

    # Model prediction for country
    Y_predict = model.predict(X_predict, verbose=0)
    # Limit values in prediction
    Y_predict = Y_predict.clip(min=0)

    # Translate prediction for absolute value 7 days in future (accumulated)
    predict_value = predictor.translate_prediction   \
        (X_predict, Y_predict)

    # Round accumulated prediction
    prediction[country] = np.round(predict_value)

    # Print prediction result
    print(country, ": ", prediction[country])

    # Plot results
    if (plotting == True):
        dataset    = cases_c[country]

        t1 = np.zeros_like(dataset)
        for i in range(len(dataset)):
            t1[i] = i

        t2 = t1[-1] + N_STEPS_FORWARD

        # Absolute plot
        plt.plot(t1, dataset, 'b.')
        plt.plot(t2, prediction[country], 'r.')
        plt.show()
        # Diff plot
        dataset = np.diff(cases_c[country])
        plt.plot(t1[1:], dataset, 'b.')
        plt.plot(t2, (Y_predict / N_STEPS_FORWARD), 'r.')
        plt.show()

print("Done")