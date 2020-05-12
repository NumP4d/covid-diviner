#!/usr/bin/python3
from datetime import date
import numpy as np

import data_reader
import predictor
import prediction_quality


print('Hello mister! I\'m your diviner')

#list of countries
countries = data_reader.european_countries()

#analysed dates
START_DATE  = date(2020, 4, 11)
END_DATE    = date(2020, 5, 11)

# Data set splitting for learn as 67%
SET_SPLIT_THRESHOLD = 0.67

# Model parameters
N_STEPS     = 14
N_FEATURES  = 1
N_NEURONS   = 200

date_list   = data_reader.date_set_preparation(START_DATE, END_DATE)

covid_data  = data_reader.read_covid_file(countries, date_list)

print('Prediction:')
cases_p = dict()
for country in countries:
    print(country)
    X, Y = predictor.split_sequence(covid_data[country], N_STEPS)
    X_learn, Y_learn, X_test, Y_test = predictor.create_train_test_set(X, Y, SET_SPLIT_THRESHOLD)
    model = predictor.lstm_model_create(N_NEURONS, N_STEPS, N_FEATURES)
    X_learn = X_learn.reshape((X_learn.shape[0], X_learn.shape[1], N_FEATURES))
    model.fit(X_learn, Y_learn, epochs=200, verbose=0)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], N_FEATURES))
    Y_predict = model.predict(X_test, verbose=0)
    MSE = 0
    for i in range(len(Y_predict)):
        MSE += (Y_predict[i] - Y_test[i])**2
    print("MSE for test set: ", MSE)
    print("Y_predict")
    print(Y_predict)
    print("Y_test")
    print(Y_test)
    quality = prediction_quality.quality_array(Y_predict, Y_test)
    print(quality)
    #cases_p[country] = predictor.predict_next_value(covid_data[country], 7)
    #print(cases_p[country])

#print('Prediction quality:')

#quality_poland  = prediction_quality.quality_country(cases_p['Poland'], cases_t['Poland'])
#quality_europe  = prediction_quality.quality_all(cases_p, cases_t)

#print('for Poland: ', quality_poland, '%, for Europe: ', quality_europe, '%')

print("Done!")