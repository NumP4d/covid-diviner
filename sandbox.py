#!/usr/bin/python3
from datetime import date
import numpy as np

import data_reader
import predictor
import prediction_quality

N_STEPS_BACKWARDS   = 2
N_STEPS_FORWARD     = 3

sequence = np.array([i for i in range(10)])

print(sequence)

X, Y = predictor.split_sequence(sequence, N_STEPS_BACKWARDS, N_STEPS_FORWARD)

# summarize the data
for i in range(len(Y)):
	print(X[i], Y[i])