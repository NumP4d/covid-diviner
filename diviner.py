#!/usr/bin/python3
from datetime import date
import numpy as np

import data_reader
import predictor
import prediction_quality


print('Hello mister! I\'m your diviner')

# list of countries
countries = data_reader.european_countries()

# analysed dates
start_date = date(2020, 4, 12)
end_date = date(2020, 4, 26)
DAYS_FORWARD = 7

date_test = data_reader.test_date(date(2020, 5, 3))

date_list = data_reader.date_set_preparation(start_date, end_date)

covid_data = data_reader.read_covid_file(countries, date_list)
cases_t = data_reader.read_covid_file(countries, date_test)

print('Prediction:')
cases_linear = dict()
cases_poly = dict()
for country in countries:
    print(country)
    cases_linear[country] = predictor.predict_poly_regression(covid_data[country], DAYS_FORWARD, 1)
    cases_poly[country] = predictor.predict_poly_regression(covid_data[country], DAYS_FORWARD, 2)
    print(cases_linear[country])
    print(cases_poly[country])

print('Prediction quality:')

quality_poland = prediction_quality.quality_country(
    cases_linear['Poland'], cases_t['Poland'])
quality_europe = prediction_quality.quality_all(cases_linear, cases_t)

print('for Poland: ', quality_poland, '%, for Europe: ', quality_europe, '%')

print('Prediction quality for polynomial:')

quality_poland_poly = prediction_quality.quality_country(
    cases_poly['Poland'], cases_t['Poland'])
quality_europe_poly = prediction_quality.quality_all(cases_poly, cases_t)

print('for Poland: ', quality_poland_poly,
      '%, for Europe: ', quality_europe_poly, '%')

print("Done!")
