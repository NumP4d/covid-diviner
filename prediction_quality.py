import numpy as np

def quality_country(cases_p, cases_t):
    quality = np.absolute(cases_p - cases_t) / cases_t * 100
    return quality

def quality_all(cases_p, cases_t):
    quality = 0
    for key in cases_p:
        quality += quality_country(cases_p[key], cases_t[key])
    quality /= len(cases_p)
    return quality