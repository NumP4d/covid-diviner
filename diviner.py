#!/usr/bin/python3

from datetime import date
import numpy as np
import pycountry

import data_reader

countries = data_reader.european_countries()

start_date  = date(2020, 4, 19)
end_date    = date(2020, 5, 3)

date_list = data_reader.date_set_preparation(start_date, end_date)

covid_data = data_reader.read_covid_file(countries, date_list)

for country in countries:
    print(country)
    print(covid_data[country])