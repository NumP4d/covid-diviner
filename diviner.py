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
END_DATE    = date(2020, 6, 2)

# Data set splitting for learn as 67%
SET_SPLIT_THRESHOLD = 0.67

# Model parameters
N_STEPS_BACKWARDS   = 7
N_STEPS_FORWARD     = 7
N_FEATURES          = 1
N_NEURONS           = 16

date_list   = data_reader.date_set_preparation(START_DATE, END_DATE)

covid_data  = data_reader.read_covid_file(countries, date_list)

print('Prediction:')
cases_p = dict()
for country in countries:
    print(country)
    dataset = np.diff(covid_data[country])
    X, Y = predictor.split_sequence(dataset, N_STEPS_BACKWARDS, N_STEPS_FORWARD)
    # Split sequences for test as only last sequence
    #X_learn, Y_learn = X[:-1, :], Y[:-1, :]
    #X_test, Y_test = X[-1:, :], Y[-1:, :]
    # Prepare model, train it and make test prediction
    model = predictor.lstm_model_create(N_NEURONS, N_STEPS_BACKWARDS, N_FEATURES, N_STEPS_FORWARD)
    X = X.reshape((X.shape[0], X.shape[1], N_FEATURES))
    model.fit(X, Y, epochs=200, verbose=0)
    # Data for prediction for future
    X_pred = np.array(dataset[-N_STEPS_BACKWARDS:])
    X_pred = X_pred.reshape((1, X_pred.shape[0], N_FEATURES))
    Y_predict = model.predict(X_pred, verbose=0)

    # Plot results
    dataset = covid_data[country]
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
    dataset_linear = dataset[-N_STEPS_BACKWARDS:]
    for i in range(len(prediction)):
        pred_linear[i] = predictor.predict_next_value(dataset_linear, i + 1)

    plt.plot(t1, dataset, 'b.')
    plt.plot(t2, prediction, 'r.')
    plt.plot(t2, pred_linear, 'g.')
    #plt.show()
    dataset = np.diff(covid_data[country])
    prediction2 = Y_predict
    t1 = t1[1:]
    plt.plot(t1, dataset, 'b.')
    plt.plot(t2, prediction2, 'r.')
    #plt.show()

    print('Prediction CNN: ', prediction[-1])
    print('Prediction linear: ', pred_linear[-1])

    #cases_for_linear = covid_data[country]
    #cases_for_linear = cases_for_linear[:-N_STEPS_FORWARD]

    #cases_t = covid_data[country]
    #cases_t = cases_t[-1] # last item

    #cases_p_linear = predictor.predict_next_value(cases_for_linear, N_STEPS_FORWARD)


    #cases_p_rnn = cases_for_linear[-1] + np.sum(Y_predict)

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