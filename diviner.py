#!/usr/bin/python3
from datetime import date
import numpy as np

import data_reader
import predictor


print('Hello mister! I\'m your diviner')

#list of countries
countries = data_reader.european_countries()

#analysed dates
start_date  = date(2020, 4, 19)
end_date    = date(2020, 5, 3)

date_list = data_reader.date_set_preparation(start_date, end_date)

covid_data = data_reader.read_covid_file(countries, date_list)

for country in countries:
    print(country)
    # print(covid_data[country])
    y = predictor.predict_next_value(covid_data[country], 7)
    print(y)

print("Done!")

# x=[]
# for i in range(10):
#     x.append(i)
# print(x)

# y = predictor.predict_next_value(x)
# print('Let me guess! Hmmmm maybe it will be:')
# print(y)
