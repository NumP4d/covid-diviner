#!/usr/bin/python3
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
END_DATE    = date(2020, 5, 18)

# Data set splitting for learn as 67%
SET_SPLIT_THRESHOLD = 0.67

# Model parameters
N_STEPS_BACKWARDS   = 10
N_STEPS_FORWARD     = 7
N_FEATURES          = 1
N_NEURONS           = 64

date_list   = data_reader.date_set_preparation(START_DATE, END_DATE)

covid_data  = data_reader.read_covid_file(countries, date_list)

print('Prediction:')
cases_p = dict()
for country in countries:
    #print(country)
    dataset = np.diff(covid_data[country])
    X, Y = predictor.split_sequence(dataset, N_STEPS_BACKWARDS, N_STEPS_FORWARD)
    # Split sequences for test as only last sequence
    X_learn, Y_learn = X[:-1, :], Y[:-1, :]
    X_test, Y_test = X[-1:, :], Y[-1:, :]
    # Prepare model, train it and make test prediction
    model = predictor.lstm_model_create(N_NEURONS, N_STEPS_BACKWARDS, N_FEATURES, N_STEPS_FORWARD)
    X_learn = X_learn.reshape((X_learn.shape[0], X_learn.shape[1], N_FEATURES))
    model.fit(X_learn, Y_learn, epochs=200)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], N_FEATURES))
    Y_predict = model.predict(X_test, verbose=0)

    print("Y_test")
    print(Y_test)
    print("Y_predict")
    print(Y_predict)

    # Plot results
    dataset = covid_data[country]
    last_value = dataset[len(dataset) - 1 - N_STEPS_FORWARD]
    prediction = np.zeros(N_STEPS_FORWARD)
    Y_predict = Y_predict.flatten()
    Y_predict = np.round(Y_predict)
    for i in range(N_STEPS_FORWARD):
        prediction[i] = last_value + Y_predict[i]
        last_value = prediction[i]
    t = np.zeros_like(dataset)
    for i in range(len(dataset)):
        t[i] = i

    plt.plot(t, dataset, 'b.')
    plt.plot(t[-N_STEPS_FORWARD:], prediction, 'r.')
    plt.show()
    dataset = np.diff(covid_data[country])
    prediction = Y_predict
    t = t[:-1]
    plt.plot(t, dataset, 'b.')
    plt.plot(t[-N_STEPS_FORWARD:], prediction, 'r.')
    plt.show()

    MSE = 0
    Y_test = Y_test.flatten()
    for i in range(len(Y_predict)):
        MSE += ((Y_predict[i] - Y_test[i])**2)
    print("MSE for test set: ", MSE)

    #Y = Y_test
    #MSE = 0
    #for i in range(len(Y_predict)):
    #    MSE += ((Y_predict[i] - Y[i])**2)
    #print("MSE for test set: ", MSE)
    #print("Y_predict")
    #print(Y_predict)
    #print("Y_test")
    #print(Y)
    #quality = prediction_quality.quality_array(Y_predict, Y)
    #print(quality)
    #cases_p[country] = predictor.predict_next_value(covid_data[country], 7)
    #print(cases_p[country])
    #print("X")
    #print(X_learn)
    #print("Y")
    #print(Y_learn)

#print('Prediction quality:')

#quality_poland  = prediction_quality.quality_country(cases_p['Poland'], cases_t['Poland'])
#quality_europe  = prediction_quality.quality_all(cases_p, cases_t)

#print('for Poland: ', quality_poland, '%, for Europe: ', quality_europe, '%')

print("Done!")