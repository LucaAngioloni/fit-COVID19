import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import wget
import os

from scipy.optimize import curve_fit

days_past = -5
days_future = 60

def logistic(x, L, k, x0, y0):
    y = L / (1 + np.exp(-k*(x-x0))) + y0
    return y

def logistic_derivative(x, L, k, x0):
    y = k * L * (np.exp(-k*(x-x0))) / np.power(1 + np.exp(-k*(x-x0)), 2)
    return y

def fit_logistic(ydata, title, ylabel):
    xdata = np.array(list(range(len(ydata)))) - len(ydata)

    popt, pcov = curve_fit(logistic, xdata, ydata, bounds=([0, 0, -100, 0], [100000, 1, 100, 10]))

    print(title)
    print('    fit: L=%5.3f, k=%5.3f, x0=%5.3f, y0=%5.3f' % tuple(popt))

    total_xaxis = np.array(list(range(days_past, days_future))) - len(ydata)
    plt.plot(total_xaxis, logistic(total_xaxis, *popt), 'g-', label='prediction')
    plt.plot(xdata, ydata, 'b-', label='real data')
    plt.xlabel('giorni (0 = oggi)')
    plt.ylabel(ylabel)
    plt.title(title + ' - ' + str(datetime.date.today().strftime("%d-%m-%Y")))
    plt.legend()
    plt.show()


def fit_logistic_derivative(ydata, title, ylabel):
    xdata = np.array(list(range(len(ydata)))) - len(ydata)

    popt, pcov = curve_fit(logistic_derivative, xdata, ydata, bounds=([0, 0, -100], [100000, 1, 100]))

    print(title)
    print('    fit: L=%5.3f, k=%5.3f, x0=%5.3f' % tuple(popt))

    total_xaxis = np.array(list(range(days_past, days_future))) - len(ydata)
    plt.plot(total_xaxis, logistic_derivative(total_xaxis, *popt), 'g-', label='prediction')
    plt.plot(xdata, ydata, 'b-', label='real data')
    plt.xlabel('giorni (0 = oggi)')
    plt.ylabel(ylabel)
    plt.title(title + ' - ' + str(datetime.date.today().strftime("%d-%m-%Y")))
    plt.legend()
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
fit_logistic(ydata, 'Contagi', 'contagiati')

ydata = data['deceduti'].tolist()
fit_logistic(ydata, 'Deceduti', 'deceduti')

ydata = data['ricoverati_con_sintomi'].tolist()
fit_logistic(ydata, 'Ricoverati', 'ricoverati')

ydata = data['terapia_intensiva'].tolist()
fit_logistic(ydata, 'Terapia Intensiva', 'in terapia')

ydata = data['nuovi_attualmente_positivi'].tolist()
fit_logistic_derivative(ydata, 'Nuovi positivi', 'nuovi positivi')

