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
start_date  = date(2020, 4, 28)
end_date    = date(2020, 5, 5)

date_test   = data_reader.test_date(date(2020, 5, 12))

date_list   = data_reader.date_set_preparation(start_date, end_date)

covid_data  = data_reader.read_covid_file(countries, date_list)
cases_t     = data_reader.read_covid_file(countries, date_test)

print('Prediction:')
cases_p = dict()
for country in countries:
    print(country)
    cases_p[country] = predictor.predict_next_value(covid_data[country], 7)
    print(cases_p[country])

print('Prediction quality:')

quality_poland  = prediction_quality.quality_country(cases_p['Poland'], cases_t['Poland'])
quality_europe  = prediction_quality.quality_all(cases_p, cases_t)

print('for Poland: ', quality_poland, '%, for Europe: ', quality_europe, '%')

print("Done!")