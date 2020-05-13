#!/usr/bin/python3

from datetime import date, timedelta
import csv
import numpy as np

def date_set_preparation(start_date, end_date):
    delta = end_date - start_date
    date_list = []
    for i in range(delta.days + 1):
        date        = start_date + timedelta(days=i)
        date_str    = date.strftime('%-m/%-d/%y')
        date_list.append(date_str)
    return date_list

def test_date(date):
    test_date = [date.strftime('%-m/%-d/%y')]
    return test_date

def read_covid_file(countries, date_list):
    filepath    = 'COVID-19/csse_covid_19_data/csse_covid_19_time_series/'
    filename    = 'time_series_covid19_confirmed_global.csv'

    cases = dict()
    for country in countries:
        cases[country] = np.zeros_like(date_list)

    with open(filepath + filename) as cases_file:
        csv_reader = csv.DictReader(cases_file)
        for row in csv_reader:
            country = row['Country/Region']
            if country in countries:
                if not row['Province/State']:
                    for i in range(len(date_list)):
                        cases[country][i] = row[date_list[i]]
                    cases[country] = cases[country].astype(np.int32)
    return cases

def european_countries():
    countries   = [ 'Poland',
                'Albania',
                'Andorra',
                'Armenia',
                'Austria',
                'Azerbaijan',
                'Belarus',
                'Belgium',
                'Bosnia and Herzegovina',
                'Bulgaria',
                'Croatia',
                'Cyprus',
                'Czechia',  # Czech Republic
                'Denmark',
                'Estonia',
                'Finland',
                'France',
                'Georgia',
                'Greece',
                'Hungary',
                'Iceland',
                'Ireland',
                'Italy',
                'Kazakhstan',
                'Latvia',
                'Liechtenstein',
                'Lithuania',
                'Luxembourg',
                'Malta',
                'Moldova',
                'Monaco',
                'Montenegro',
                'Netherlands',
                'North Macedonia',
                'Norway',
                'Portugal',
                'Romania',
                'Russia',
                'San Marino',
                'Serbia',
                'Slovakia',
                'Slovenia',
                'Spain',
                'Sweden',
                'Switzerland',
                'Turkey',
                'Ukraine',
                'United Kingdom',
                'Holy See'      ]   # Vatican City
    return countries
