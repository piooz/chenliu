import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import statsmodels.tsa.arima.model as tsa

import matplotlib.pyplot as plt
import statsmodels.api as sm
import logging as log

import effects as eff
from statsmodels.tsa.arima_process import arma2ma


def calcx4(poly, delta) -> np.ndarray:
    n = len(poly)
    arr = np.zeros(n)

    for k in range(1, n):
        dd = delta**k
        sum = 0
        for j in range(k - 1):
            sum += delta ** (k - j) * poly[j]   # - poly[k]
        arr[k] = dd - sum - poly[k]

    # # chui wie które dobre
    # powers = np.power(delta, np.arange(1, n + 1))
    # print(powers)
    # cum0 = np.cumsum(poly)
    # cum1 = np.concatenate([[0], np.cumsum(poly)])[0:100]
    # lsd = (cum1 * np.flip(powers)) - cum0
    # lsd = powers - lsd
    # print(arr)

    return arr


def calc_stats(delta: float, fit: tsa.ARIMAResults, n):
    poly = arma2ma(fit.polynomial_ar, fit.polynomial_ma, n)
    poly = -poly
    poly[0] = 1
    sigma = 1.483 * fit.mae

    log.debug(poly)

    # calc x stats

    x2 = -poly
    x2[0] = 1
    x3 = -np.cumsum(poly) + 1
    x3[0] = 1
    x4 = calcx4(poly, delta)
    x4[0] = 1

    # calc omegas
    omio = fit.resid
    omao = np.cumsum(fit.resid * x2) / np.cumsum(np.square(x2))
    omls = np.cumsum(fit.resid * x3) / np.cumsum(np.square(x3))
    omtc = np.cumsum(fit.resid * x4) / np.cumsum(np.square(x4))
    log.debug(omio)
    log.debug(omao)
    log.debug(omls)
    log.debug(omtc)
    df_om = pd.DataFrame(
        {
            'omega_io': omio,
            'omega_ao': omao,
            'omega_ls': omls,
            'omega_tc': omtc,
        }
    )

    # calc tau
    tauio = omio / sigma
    tauao = (omao / sigma) * np.sqrt(np.cumsum(np.square(x2)))
    tauls = (omls / sigma) * np.sqrt(np.cumsum(np.square(x3)))
    tautc = (omtc / sigma) * np.sqrt(np.cumsum(np.square(x4)))
    log.debug(tauio)
    log.debug(tauao)
    log.debug(tauls)
    log.debug(tautc)

    df = pd.DataFrame(
        {
            'tau_io': tauio,
            'tau_ao': tauao,
            'tau_ls': tauls,
            'tau_tc': tautc,
        }
    )
    return df, df_om


def effects_matrix(fit: tsa.ARIMAResults, df: DataFrame, delta: float, n: int):
    array = []
    for i, row in df.iterrows():
        s = []
        w = float(row['omega'])
        match row['type']:
            case 'ls':
                s = eff.ls_effect(n, i, w)
            case 'io':
                s = eff.io_effect(n, i, fit.arparams, fit.maparams, w)
            case 'ao':
                s = eff.ao_effect(n, i, w)
            case 'tc':
                s = eff.tc_effect(n, i, w, delta)
            case _:
                log.warning(f'cannot recognize type {row["type"]}. Ignoring')
        array.append(s)
    return array


def calculate_effect(
    fit: tsa.ARIMAResults, df: DataFrame, delta: float, n: int
):
    out = np.zeros(n)
    for i, row in df.iterrows():
        s = []
        w = float(row['omega'])
        match row['type']:
            case 'ls':
                s = eff.ls_effect(n, i, w)
            case 'io':
                s = eff.io_effect(n, i, fit.arparams, fit.maparams, w)
            case 'ao':
                s = eff.ao_effect(n, i, w)
            case 'tc':
                s = eff.tc_effect(n, i, w, delta)
            case _:
                log.warning(f'cannot recognize type {row["type"]}. Ignoring')
        out += s
    return out


def remove_effects(
    y: Series, fit: tsa.ARIMAResults, df: DataFrame, delta: float
):
    newseries = np.copy(y)
    effect = calculate_effect(fit, df, delta, len(y))
    return Series(newseries - effect)


def determine_type(row):
    max_tau = row[['tau_io', 'tau_ao', 'tau_ls', 'tau_tc']].abs().idxmax()
    if max_tau == 'tau_io':
        return 'io'
    elif max_tau == 'tau_ao':
        return 'ao'
    elif max_tau == 'tau_ls':
        return 'ls'
    elif max_tau == 'tau_tc':
        return 'tc'


def transform_dataframe(df: DataFrame, cval):
    out = pd.DataFrame(
        columns=[
            'type',
            'omega',
            'tau',
        ]
    )
    for i, row in df.iterrows():
        type = determine_type(row)
        omega = row[f'omega_{type}']
        tau = row[f'tau_{type}']
        if abs(tau) > cval:
            out.loc[i] = [type, omega, tau]
    return out


# ARIMA model
def stage1(
    y: Series,
    order: tuple,
    cval: float = 2.0,
    delta: float = 0.7,
    iter: int = 2,
):

    n = len(y)
    series = y.copy()
    out = pd.DataFrame(
        columns=[
            'type',
            'omega',
            'tau',
        ]
    )

    for _ in range(iter):
        log.warning(series)
        fit: tsa.ARIMAResults = tsa.ARIMA(series, order=order).fit()
        df_tau, df_om = calc_stats(delta, fit, n)
        tau_om = pd.concat([df_tau, df_om], axis=1)

        raport = transform_dataframe(tau_om, cval)
        raport = raport.drop(index=out.index.tolist(), errors='ignore')
        if out.empty:
            out = raport
        else:
            out = pd.concat([out, raport])

        if out.empty:
            log.error('Not found any potential outlier...')
            return out

        series = remove_effects(series, fit, raport, delta)

    return out


def stage2(
    y: Series, stage1_report: DataFrame, cval: float, order, delta: float
):

    report = stage1_report
    fit: tsa.ARIMAResults = tsa.ARIMA(y, order=order).fit()

    while True:
        matrix = effects_matrix(fit, report, delta, len(y))

        effects_df = DataFrame.from_records(matrix).transpose()
        effects_df.columns = report.index.values

        fit_prawilny: tsa.ARIMAResults = tsa.ARIMA(
            y, order=order, exog=effects_df
        ).fit()

        data = {}
        for i in iter(effects_df.columns):
            data[i] = fit_prawilny.params[i]

        omega_std = np.array([data[i] for i in data]).std()

        for k, v in data.items():
            data[k] = v / omega_std

        series = Series(data)

        log.warning(series)
        if series[series.abs() < cval].any():
            report = report.drop(series.abs().idxmin())

        if len(report) <= 1 or (series.abs() > cval).all():
            return report, remove_effects(y, fit_prawilny, report, delta)


def stage3(y: Series, order, cval: float, delta: float = 0.7):
    out1 = stage1(y, order, cval, delta, 1)
    if out1.empty:
        return None, None
    out2, final_series = stage2(y, out1, cval, order, delta)

    return out2, final_series


def chen_liu(y: Series, arima_order=(2, 0, 2), cval=2):
    delta = 0.7

    stage1_output = stage1(y, arima_order, cval)
    if stage1_output.empty:
        return 1
    stage2_output, series = stage2(y, stage1_output, cval, arima_order, delta)
    fin_raport, fin_series = stage3(Series(y), arima_order, cval, delta)

    if fin_raport is None and fin_series is None:
        fin_raport = stage2_output
        fin_series = series

    return fin_raport, fin_series


if __name__ == '__main__':

    def main():
        y = sm.datasets.nile.data.load_pandas().data['volume']
        cval = 1.9
        order = (2, 0, 2)
        delta = 0.7

        stage1_output = stage1(y, order, cval)

        if stage1_output.empty:
            return 1
        stage2_output, series = stage2(y, stage1_output, cval, order, delta)
        raport, out = stage3(Series(y), order, cval, delta)

        if raport is None and out is None:
            raport = stage2_output
            out = series

        print(raport)

        plt.plot(y)
        plt.plot(out)
        plt.show()

        return 0

    main()
