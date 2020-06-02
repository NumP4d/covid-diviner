import numpy as np
import math

def quality_country(cases_p, cases_t):
    quality = np.absolute(cases_p - cases_t) / cases_t * 100
    return quality

def quality_all(cases_p, cases_t):
    quality = 0
    for key in cases_p:
        quality += quality_country(cases_p[key], cases_t[key])
    quality /= len(cases_p)
    return quality

def mse(Y_real, Y_predict):
    Y_real      = Y_real.flatten()
    Y_predict   = Y_predict.flatten()
    mse = 0
    for i in range(Y_real.size):
        mse += (Y_real[i] - Y_predict[i]) ** 2
    mse /= Y_real.size

    return mse

def calculate(Y_real, Y_predict):
    quality = 0
    for i in range(len(Y_real)):
        if (Y_real[i] != 0):
            quality += np.absolute(Y_real[i] - Y_predict[i]) / Y_real[i]
        else:
            quality += np.absolute(Y_real[i] - Y_predict[i]) / 0.001
    quality /= len(Y_real)
    quality *= 100

    return quality

def quality_array(cases_p, cases_t):
    quality = 0
    weights = []
    alpha = 7
    N = len(cases_p) - 1
    for i in range(len(cases_p)):
        weight = math.exp((N-i) / N)
        weights.append(weight)
    weights = np.array(weights)
    weights = weights / weights.sum()
    for i in range(len(cases_p)):
        quality += weights[i] * quality_country(cases_p[i], cases_t[i])
    #quality /= len(cases_p)
    quality = 100 - quality # translate to percentage of accuracy
    return quality