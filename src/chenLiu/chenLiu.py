import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import statsmodels.tsa.arima.model as tsa

import matplotlib.pyplot as plt
import statsmodels.api as sm
import logging as log

from . import effects as eff
from . import arma2ma

# from statsmodels.tsa.arima_process import arma2ma


def calcx4(poly, delta) -> np.ndarray:
    n = len(poly)
    arr = np.zeros(n)

    for k in range(1, n):
        dd = delta**k
        sum = 0
        for j in range(k - 1):
            sum += delta ** (k - j) * poly[j]   # - poly[k]
        arr[k] = dd - sum - poly[k]

    # powers = np.power(delta, np.arange(1, n + 1))
    # print(powers)
    # cum0 = np.cumsum(poly)
    # cum1 = np.concatenate([[0], np.cumsum(poly)])[0:100]
    # lsd = (cum1 * np.flip(powers)) - cum0
    # lsd = powers - lsd
    # print(arr)

    return arr


def calc_stats(delta: float, fit: tsa.ARIMAResults, n):
    poly = arma2ma.arma2ma(fit.polynomial_ar, fit.polynomial_ma, n)
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
    data = {}
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
        data[i] = s
    return DataFrame(data)


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
    effect = calculate_effect(fit, df, delta, y.size)
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
    fixed_fit=False,
):

    n = y.size
    series = y.copy()
    out = pd.DataFrame(
        columns=[
            'type',
            'omega',
            'tau',
        ]
    )

    fit: tsa.ARIMAResults = tsa.ARIMA(series, order=order).fit()
    for _ in range(iter):
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
            return out, None

        series = remove_effects(series, fit, raport, delta)
        if not fixed_fit:
            fit: tsa.ARIMAResults = tsa.ARIMA(series, order=order).fit()

    return out, fit


def stage2(
    y: Series,
    stage1_report: DataFrame,
    cval: float,
    order,
    delta: float,
    fit: tsa.ARIMAResults,
):
    eff = effects_matrix(fit, stage1_report, delta, len(y))
    stage2_outliers = stage1_report
    while True:
        resid = fit.resid
        result = sm.OLS(resid, eff).fit()
        tau = result.params / np.std(result.params)
        if tau[tau.abs() < cval].any():
            eff = eff.drop(columns=tau.abs().idxmin())
            stage2_outliers = stage2_outliers.drop(tau.abs().idxmin())
        else:
            corrected_series = remove_effects(y, fit, stage2_outliers, delta)
            fit = tsa.ARIMA(y, order=order).fit()

            return (
                stage2_outliers,
                corrected_series,
                fit,
            )


def stage3(
    y: Series, order, cval: float, fit: tsa.ARIMAResults, delta: float = 0.7
):
    out1, f1 = stage1(y, order, cval, delta, 1, True)
    if out1.empty:
        return None, None
    out2, final_series, fit = stage2(y, out1, cval, order, delta, f1)

    return out2, final_series, fit


def chen_liu(y: Series | list, arima_order=(2, 0, 2), cval=2):
    if isinstance(y, list):
        y = Series(y)
    y = y.reset_index(drop=True)
    delta = 0.7

    stage1_output, fit = stage1(y, arima_order, cval)
    if stage1_output.empty:
        log.warning('After stage1: Did not found any outlier')
        return 1
    (stage2_outliers, corrected_series, fit) = stage2(
        y, stage1_output, cval, arima_order, delta, fit
    )
    if stage2_outliers.empty:
        log.warning('After stage2: Did not found any outlier')
        return 2

    stage3_outliers, out, fin_fit = stage3(
        Series(y), arima_order, cval, fit, delta
    )

    fin_effects = calculate_effect(fin_fit, stage3_outliers, delta, len(y))

    return stage3_outliers, out, fin_effects, fin_fit


# TODO: Maybe use async to make it usable for bigger sets
# def chen_liu_chunked(y: Series, arima_order=(2, 0, 2), cval=2, chunks=2):
#     assert chunks >= 2
#
#     eff_out = np.array([])
#     series_out = np.array([])
#     raport_out = DataFrame()
#     fit = None
#     for chunk in np.split(y, chunks):
#         chunk = chunk.reset_index(drop=True)
#         print(len(chunk))
#         print(chunk)
#         out, ser, eff, fit = chen_liu(chunk, arima_order, cval)
#
#         raport_out = raport_out._append(out)
#         series_out = series_out.concatenate(ser, ignore_index=True)
#         eff_out = series_out.append(eff, ignore_index=True)
#
#     return raport_out, series_out, eff_out, fit
# print(results)


# if __name__ == '__main__':
#
#     def main():
#         y = sm.datasets.nile.data.load_pandas().data['volume']
#         cval = 2.0
#         order = (1, 0, 1)
#         delta = 0.7
#
#         stage1_output, fit = stage1(y, order, cval)
#         if stage1_output.empty:
#             log.warning('After stage1: Did not found any outlier')
#             return 1
#         (stage2_outliers, corrected_series, fit) = stage2(
#             y, stage1_output, cval, order, delta, fit
#         )
#         if stage2_outliers.empty:
#             log.warning('After stage2: Did not found any outlier')
#             return 2
#
#         stage2_outliers, out, fin_fit = stage3(
#             Series(y), order, cval, fit, delta
#         )
#
#         print(stage2_outliers)
#
#         plt.plot(y)
#         plt.plot(out)
#         plt.show()
#
#         return 0
#
#     main()
