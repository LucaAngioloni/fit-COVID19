import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import numpy as np
import pandas as pd
import datetime
from datetime import timedelta
import wget
import os

from scipy.optimize import curve_fit

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

days_past = -2 # days beyond the start of the data to plot
days_future = 50 # days after the end of the data to predict and plot

myFmt = mdates.DateFormatter('%d/%m') # date formatter for matplotlib
show_every = 3 # int value that defines how often to show a date in the x axis. (used not to clutter the axis)

coeff_std = 1.1 # coefficient that defines how many standard deviations to use
coeff_std_d = 0.2

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

def fit_logistic(ydata, title, ylabel):
    xdata = np.array(list(range(-len(ydata), 0))) + 1

    popt, pcov = curve_fit(logistic, xdata, ydata, p0=[20000, 0.5, 1, 0], bounds=([0, 0, -100, 0], [200000, 10, 100, 1]))

    print(title)
    print('    fit: L=%5.3f, k=%5.3f, x0=%5.3f, y0=%5.3f' % tuple(popt))

    perr = np.sqrt(np.diag(pcov))
    print(perr)

    pworst = popt + coeff_std*perr
    pbest = popt - coeff_std*perr

    fig, ax = plt.subplots(figsize=(15,8))

    ax.xaxis.set_major_formatter(myFmt)
    fig.autofmt_xdate()

    total_xaxis = np.array(list(range(-len(ydata) + days_past, days_future))) + 1

    date_xdata = [datetime.date.today() + timedelta(days=int(i)) for i in xdata]
    date_total_xaxis = [datetime.date.today() + timedelta(days=int(i)) for i in total_xaxis]

    ax.plot(date_total_xaxis, logistic(total_xaxis, *popt), 'g-', label='prediction')
    ax.plot(date_xdata, ydata, 'b-', label='real data')

    # popt, pcov = curve_fit(logistic, xdata[:-4], ydata[:-4], p0=[20000, 0.5, 1, 0], bounds=([0, 0, -100, 0], [200000, 10, 100, 1]))
    # ax.plot(date_total_xaxis, logistic(total_xaxis, *popt), 'r-', label='old prediction')

    future_axis = total_xaxis[len(ydata) - days_past:]
    date_future_axis = [datetime.date.today() + timedelta(days=int(i)) for i in future_axis]
    ax.fill_between(date_future_axis, logistic(future_axis, *pbest), logistic(future_axis, *pworst), 
        facecolor='red', alpha=0.2, label='std')

    start = (len(ydata) - days_past - 1) % show_every
    ax.set_xticks(date_total_xaxis[start::show_every])
    ax.set_xlabel('Giorni - date')
    ax.set_ylabel(ylabel)
    ax.set_title(title + ' - ' + str(datetime.date.today().strftime("%d-%m-%Y")))
    ax.legend(loc='upper left')
    ax.grid(True)

    plt.show()


def fit_logistic_derivative(ydata, title, ylabel):
    xdata = np.array(list(range(-len(ydata), 0))) + 1

    popt, pcov = curve_fit(logistic_derivative, xdata, ydata, p0=[20000, 0.5, 1], bounds=([0, 0, -100], [200000, 10, 100]))

    print(title)
    print('    fit: L=%5.3f, k=%5.3f, x0=%5.3f' % tuple(popt))

    perr = np.sqrt(np.diag(pcov))
    print(perr)

    pworst = popt + coeff_std_d*perr
    pbest = popt - coeff_std_d*perr

    fig, ax = plt.subplots(figsize=(15,8))

    ax.xaxis.set_major_formatter(myFmt)
    fig.autofmt_xdate()

    total_xaxis = np.array(list(range(-len(ydata) + days_past, days_future))) + 1

    date_xdata = [datetime.date.today() + timedelta(days=int(i)) for i in xdata]
    date_total_xaxis = [datetime.date.today() + timedelta(days=int(i)) for i in total_xaxis]

    ax.plot(date_total_xaxis, logistic_derivative(total_xaxis, *popt), 'g-', label='prediction')
    ax.plot(date_xdata, ydata, 'b-', label='real data')

    # popt, pcov = curve_fit(logistic_derivative, xdata[:-4], ydata[:-4], p0=[20000, 0.5, 1], bounds=([0, 0, -100], [200000, 10, 100]))
    # ax.plot(date_total_xaxis, logistic_derivative(total_xaxis, *popt), 'r-', label='old prediction')

    future_axis = total_xaxis[len(ydata) - days_past:]
    date_future_axis = [datetime.date.today() + timedelta(days=int(i)) for i in future_axis]
    ax.fill_between(date_future_axis, logistic_derivative(future_axis, *pbest), logistic_derivative(future_axis, *pworst), 
        facecolor='red', alpha=0.2, label='std')

    start = (len(ydata) - days_past - 1) % show_every
    ax.set_xticks(date_total_xaxis[start::show_every])

    ax.set_xlabel('Giorni - date')
    ax.set_ylabel(ylabel)
    ax.set_title(title + ' - ' + str(datetime.date.today().strftime("%d-%m-%Y")))
    ax.legend(loc='upper left')
    ax.grid(True)
    
    plt.show()

if os.path.exists('data.csv'):
    os.remove('data.csv')

url = 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv'
try:
    file = wget.download(url, out='data.csv')
    print("\n")
except:
    print("BAD", url)

data = pd.read_csv('data.csv')

ydata = data['totale_casi'].tolist()
fit_logistic(ydata, 'Contagi', 'totale contagiati')

ydata = data['deceduti'].tolist()
fit_logistic(ydata, 'Deceduti', 'totale deceduti')

ydata = data['ricoverati_con_sintomi'].tolist()
fit_logistic(ydata, 'Ricoverati', 'totale ricoverati')

ydata = data['terapia_intensiva'].tolist()
fit_logistic(ydata, 'Terapia Intensiva', 'totale in terapia')

ydata = data['nuovi_attualmente_positivi'].tolist()
fit_logistic_derivative(ydata, 'Nuovi contagiati', 'nuovi contagiati')

