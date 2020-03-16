import numpy as np
import pandas as pd
from datetime import timedelta
from datetime import datetime
from io import StringIO
from urllib import request as url_request
import os
import sys

do_imgs = False

if len(sys.argv) > 1:
    do_imgs = True
    import matplotlib
    matplotlib.use('Agg')
    os.makedirs('imgs/', exist_ok=True)
    

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scipy.optimize import curve_fit

# This stuff because pandas or matplot lib complained...
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

days_past = -2 # days beyond the start of the data to plot
days_future = 50 # days after the end of the data to predict and plot

myFmt = mdates.DateFormatter('%d/%m') # date formatter for matplotlib
show_every = 3 # int value that defines how often to show a date in the x axis. (used not to clutter the axis)

coeff_std = 1.1 # coefficient that defines how many standard deviations to use
coeff_std_d = 0.4

def logistic(x, L, k, x0, y0):
    """
    General Logistic function.

    Args:
        x    float or array-like, it represents the time
        L    float, the curve's maximum value
        k    float, the logistic growth rate or steepness of the curve.
        x0   float, the x-value of the sigmoid's midpoint
        y0   float, curve's shift in the y axis
    """
    y = L / (1 + np.exp(-k*(x-x0))) + y0
    return y

def logistic_derivative(x, L, k, x0):
    """
    General Gaussian like function (derivative of the logistic).

    Args:
        x    float or array-like, it represents the time
        L    float, the curve's integral (area under the curve)
        k    float, the logistic growth rate or steepness of the curve.
        x0   float, the x-value of the max value
    """
    y = k * L * (np.exp(-k*(x-x0))) / np.power(1 + np.exp(-k*(x-x0)), 2)
    return y

def fit_curve(curve, ydata, title, ylabel, last_date, coeff_std):
    xdata = np.array(list(range(-len(ydata), 0))) + 1

    if curve.__name__ == 'logistic':
        p0=[20000, 0.5, 1, 0]
        bounds=([0, 0, -100, 0], [200000, 10, 100, 1])
        params_names = ['L', 'k', 'x0', 'y0']
    elif curve.__name__ == 'logistic_derivative':
        p0=[20000, 0.5, 1]
        bounds=([0, 0, -100], [200000, 10, 100])
        params_names = ['L', 'k', 'x0']
    else:
        print('this curve is unknown')
        return -1

    popt, pcov = curve_fit(curve, xdata, ydata, p0=p0, bounds=bounds)

    print(title)
    descr = '    fit: '
    for i, param in enumerate(params_names):
        descr = descr + "{}={:.3f}".format(param, popt[i])
        if i < len(params_names) - 1:
            descr = descr + ', '
    print(descr)

    perr = np.sqrt(np.diag(pcov))
    print(perr)

    pworst = popt + coeff_std*perr
    pbest = popt - coeff_std*perr

    fig, ax = plt.subplots(figsize=(15,8))

    ax.xaxis.set_major_formatter(myFmt)
    fig.autofmt_xdate()

    total_xaxis = np.array(list(range(-len(ydata) + days_past, days_future))) + 1

    date_xdata = [last_date + timedelta(days=int(i)) for i in xdata]
    date_total_xaxis = [last_date + timedelta(days=int(i)) for i in total_xaxis]

    ax.plot(date_total_xaxis, curve(total_xaxis, *popt), 'g-', label='prediction')
    ax.plot(date_xdata, ydata, 'b-', label='real data')

    # popt, pcov = curve_fit(logistic, xdata[:-4], ydata[:-4], p0=[20000, 0.5, 1, 0], bounds=([0, 0, -100, 0], [200000, 10, 100, 1]))
    # ax.plot(date_total_xaxis, logistic(total_xaxis, *popt), 'r-', label='old prediction')

    future_axis = total_xaxis[len(ydata) - days_past:]
    date_future_axis = [last_date + timedelta(days=int(i)) for i in future_axis]
    ax.fill_between(date_future_axis, curve(future_axis, *pbest), curve(future_axis, *pworst), 
        facecolor='red', alpha=0.2, label='std')

    start = (len(ydata) - days_past - 1) % show_every
    ax.set_xticks(date_total_xaxis[start::show_every])
    ax.set_xlabel('Giorni - date')
    ax.set_ylabel(ylabel)
    ax.set_title(title + ' - ' + str(last_date.strftime("%d-%m-%Y")))
    ax.legend(loc='upper left')
    ax.grid(True)

    if do_imgs:
        plt.savefig('imgs/' + title + '.png', dpi=200)
        plt.clf()
    else:
        plt.show()

    return popt, perr

if __name__ == '__main__':
    if os.path.exists('data.csv'):
        os.remove('data.csv')

    url = 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv'
    webFile = url_request.urlopen(url).read()
    webFile = webFile.decode('utf-8')  

    data = pd.read_csv(StringIO(webFile))

    date_string = data.iloc[-1:]['data'].values[0]
    last_date = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
    print("Ultimo aggiornamento: {}".format(last_date))

    totale_casi = data.iloc[-1:]['totale_casi'].values[0]
    print('Tot contagiati: {}'.format(totale_casi))

    totale_dimessi_guariti = data.iloc[-1:]['dimessi_guariti'].values[0]
    print('Tot dimessi guariti: {}'.format(totale_dimessi_guariti))

    totale_deceduti = data.iloc[-1:]['deceduti'].values[0]
    print('Tot deceduti: {}'.format(totale_deceduti))

    tot_tamponi = data.iloc[-1:]['tamponi'].values[0]
    print('Tot tamponi: {}'.format(tot_tamponi))

    nuovi = np.array(data['nuovi_attualmente_positivi'].tolist())

    nuovi_oggi = nuovi[-1]
    print('Tot Nuovi casi oggi: {}'.format(nuovi_oggi))

    gf_list = nuovi[1:] / nuovi[:-1]

    growth_factor = gf_list[-1]
    print('Fattore di crescita: {:.3f}'.format(growth_factor))

    avg_growth_factor = np.mean(gf_list[-3:])
    print('Fattore di crescita mediato: {:.3f}'.format(avg_growth_factor))

    print(gf_list)

    ydata = data['totale_casi'].tolist()
    p_cont, err_cont = fit_curve(logistic, ydata, 'Contagi', 'totale contagiati', last_date, coeff_std)

    ydata = data['deceduti'].tolist()
    p_dead, err_dead = fit_curve(logistic, ydata, 'Deceduti', 'totale deceduti', last_date, coeff_std)

    ydata = data['ricoverati_con_sintomi'].tolist()
    fit_curve(logistic, ydata, 'Ricoverati', 'totale ricoverati', last_date, coeff_std)

    ydata = data['terapia_intensiva'].tolist()
    fit_curve(logistic, ydata, 'Terapia Intensiva', 'totale in terapia', last_date, coeff_std)

    ydata = data['dimessi_guariti'].tolist()
    p_healed, err_healed = fit_curve(logistic, ydata, 'Dimessi Guariti', 'totale dimessi guariti', last_date, coeff_std_d)

    ydata = data['nuovi_attualmente_positivi'].tolist()
    fit_curve(logistic_derivative, ydata, 'Nuovi Contagiati', 'nuovi contagiati', last_date, coeff_std_d)

