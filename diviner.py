#!/usr/bin/python3
from datetime import date
import numpy as np

import data_reader
import predictor
import prediction_quality


print('Hello mister! I\'m your diviner')

DAYS_FORWARD = 7

# list of countries
countries = data_reader.european_countries()

#analysed dates
start_date  = date(2020, 5, 3)
end_date    = date(2020, 5, 12)

date_list = data_reader.date_set_preparation(start_date, end_date)

covid_data  = data_reader.read_covid_file(countries, date_list)

print('Prediction:')
cases_linear = dict()
cases_poly = dict()
for country in countries:
    print(country)
    #cases_linear[country] = predictor.predict_poly_regression(covid_data[country], DAYS_FORWARD, 1)
    cases_poly[country] = predictor.predict_poly_regression(covid_data[country], DAYS_FORWARD, 2)
    #print(cases_linear[country])
    print(cases_poly[country])

print("Done!")
