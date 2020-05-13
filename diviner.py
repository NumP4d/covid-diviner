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
START_DATE  = date(2020, 3, 12)
END_DATE    = date(2020, 5, 12)

# Data set splitting for learn as 67%
SET_SPLIT_THRESHOLD = 0.67

# Model parameters
N_STEPS_BACKWARDS   = 14
N_STEPS_FORWARD     = 7
N_FEATURES          = 1
N_NEURONS           = 100

date_list   = data_reader.date_set_preparation(START_DATE, END_DATE)

covid_data  = data_reader.read_covid_file(countries, date_list)

print('Prediction:')
cases_p = dict()
for country in countries:
    print(country)
    X, Y = predictor.split_sequence(covid_data[country], N_STEPS_BACKWARDS, N_STEPS_FORWARD)
    #X_learn, Y_learn, X_test, Y_test = predictor.create_train_test_set(X, Y, SET_SPLIT_THRESHOLD)
    Y_predict = []
    for i in range(len(X)):
        print(X[i, :])
        prediction = predictor.predict_next_value(X[i, :], 7)
        Y_predict.append(prediction)
    Y_predict = np.array(Y_predict)
    #model = predictor.lstm_model_create(N_NEURONS, N_STEPS_BACKWARDS, N_FEATURES)
    #X_learn = X_learn.reshape((X_learn.shape[0], X_learn.shape[1], N_FEATURES))
    #model.fit(X_learn, Y_learn, epochs=200)
    #X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], N_FEATURES))
    #Y_predict = model.predict(X_test, verbose=0)
    #Y = Y_test
    MSE = 0
    for i in range(len(Y_predict)):
        MSE += ((Y_predict[i] - Y[i])**2)
    print("MSE for test set: ", MSE)
    #print("Y_predict")
    #print(Y_predict)
    #print("Y_test")
    #print(Y)
    quality = prediction_quality.quality_array(Y_predict, Y)
    print(quality)
    #cases_p[country] = predictor.predict_next_value(covid_data[country], 7)
    #print(cases_p[country])

#print('Prediction quality:')

#quality_poland  = prediction_quality.quality_country(cases_p['Poland'], cases_t['Poland'])
#quality_europe  = prediction_quality.quality_all(cases_p, cases_t)

#print('for Poland: ', quality_poland, '%, for Europe: ', quality_europe, '%')

print("Done!")