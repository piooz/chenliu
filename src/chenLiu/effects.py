import numpy as np
from . import arma2ma


def io_effect(n, ind, ar, ma, w=1.0):
    try:
        arr = arma2ma.arma2ma(ar, ma, n - ind - 1)
    except ValueError:
        arr = np.zeros(n)
        arr[-1] = 1
        return arr
    arr = np.concatenate([np.zeros(ind), [1], arr])
    return arr * w


def ao_effect(n, ind, w=1.0):
    array = np.zeros(n)
    array[ind] = w
    return array


def tc_effect(n, ind, w=1.0, delta=0.7):
    result = np.zeros(n)
    for i in range(0, n - ind):
        result[i + ind] = (delta**i) * w
    return result


def ls_effect(n, ind, w=1.0):
    result = np.zeros(n)
    for i in range(ind, n):
        result[i] = 1
    return result * w
