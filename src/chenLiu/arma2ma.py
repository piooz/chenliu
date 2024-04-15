import numpy as np


def arma2ma(ar, ma, lag_max):
    ar = ar.tolist() if isinstance(ar, np.ndarray) else ar
    ma = ma.tolist() if isinstance(ma, np.ndarray) else ma

    if not isinstance(ar, list):
        ar = [ar]
    if not isinstance(ma, list):
        ma = [ma]

    if len(ar) == 0:
        ar = [1]
    if len(ma) == 0:
        ma = [1]

    p = len(ar)
    q = len(ma)
    m = int(lag_max)

    if m <= 0:
        raise ValueError('Invalid value of lag_max')

    psi = np.zeros(m)
    for i in range(m):
        tmp = ma[i] if i < q else 0.0
        for j in range(min(i + 1, p)):
            tmp += ar[j] * (psi[i - j - 1] if i - j - 1 >= 0 else 1.0)
        psi[i] = tmp

    return psi
