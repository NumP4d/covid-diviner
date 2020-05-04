#!/usr/bin/python3

from datetime import date, timedelta
import csv
import numpy as np

filepath    = 'COVID-19/csse_covid_19_data/csse_covid_19_time_series/'
filename    = 'time_series_covid19_confirmed_global.csv'

countries   = ['Poland']

start_date  = date(2020, 4, 25)
end_date    = date(2020, 5, 1)

delta = end_date - start_date

#date_list = [start_date + timedelta(days=i) for i in range(delta.days + 1)]

date_list = []
for i in range(delta.days + 1):
    date        = start_date + timedelta(days=i)
    date_str    = date.strftime('%m/%d/%y')
    date_list.append(date_str)

#date_list = date_list.strftime('%m/%d/%y')

print(', '.join(date_list))

with open(filepath + filename) as cases_file:
    csv_reader = csv.DictReader(cases_file)
    line_count = 0
    for row in csv_reader:
        if row['Country/Region'] in countries:
            print(', '.join(row))
        line_count += 1
    print(f'Processed {line_count} lines.')


def date_set_preparation(start_day, end_date, inc_days):
    dates = []
    return dates