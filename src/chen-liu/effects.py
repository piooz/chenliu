import numpy as np

from statsmodels.tsa.arima_process import arma2ma


def io_effect(n, ind, ar, ma, w=1.0):
    arr = arma2ma(ar, ma, n - ind - 1)
    arr = np.concatenate([np.zeros(ind), [1], arr])
    return arr * w


def ao_effect(n, ind, w=1.0):
    array = np.zeros(n)
    array[ind] = w
    return array


def tc_effect(n, ind, w=1.0, delta=0.7):
    # logging.debug(f'w: {w}')
    result = np.zeros(n)
    for i in range(0, n - ind):
        result[i + ind] = (delta**i) * w
    return result


def ls_effect(n, ind, w=1.0):
    result = np.zeros(n)
    for i in range(ind, n):
        result[i] = 1
    return result * w
