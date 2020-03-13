import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime

from scipy.optimize import curve_fit

def sigmoid(x, L, k, x0, y0):
    y = L / (1 + np.exp(-k*(x-x0))) + y0
    return y

def sigmoid_derivative(x, L, k, x0):
    y = k * L * (np.exp(-k*(x-x0))) / np.power(1 + np.exp(-k*(x-x0)), 2)
    return y

def fit_sigmoid(ydata, title, ylabel):
	xdata = np.array(list(range(len(ydata)))) - len(ydata)

	popt, pcov = curve_fit(sigmoid, xdata, ydata, bounds=([0, 0, -100, 0], [100000, 1, 100, 10]))

	print(title)
	print('    fit: L=%5.3f, k=%5.3f, x0=%5.3f, y0=%5.3f' % tuple(popt))

	# plt.plot(xdata, ydata, 'b-', label='data')
	# plt.plot(xdata, sigmoid(xdata, *popt), 'r-', label='fit: L=%5.3f, k=%5.3f, x0=%5.3f, y0=%5.3f' % tuple(popt))
	# plt.xlabel('giorni dopo 19 febbraio')
	# plt.ylabel(ylabel)
	# plt.legend()
	# plt.show()

	# plt.figure(2)

	total_xaxis = np.array(list(range(-5, 60))) - len(ydata)
	plt.plot(total_xaxis, sigmoid(total_xaxis, *popt), 'g-', label='fit: L=%5.3f, k=%5.3f, x0=%5.3f, y0=%5.3f' % tuple(popt))
	# y = np.zeros((50))
	# y[:len(ydata)] = ydata
	# plt.plot(list(range(50)), y, 'b-', label='data')
	plt.plot(xdata, ydata, 'b-', label='data')
	plt.xlabel('giorni (0 = oggi)')
	plt.ylabel(ylabel)
	plt.title(title + ' - ' + str(datetime.date.today().strftime("%d-%m-%Y")))
	#plt.legend()
	plt.show()


def fit_sigmoid_derivative(ydata, title, ylabel):
    xdata = np.array(list(range(len(ydata)))) - len(ydata)

    popt, pcov = curve_fit(sigmoid_derivative, xdata, ydata, bounds=([0, 0, -100], [100000, 1, 100]))

    print(title)
    print('    fit: L=%5.3f, k=%5.3f, x0=%5.3f' % tuple(popt))

    # plt.plot(xdata, ydata, 'b-', label='data')
    # plt.plot(xdata, sigmoid(xdata, *popt), 'r-', label='fit: L=%5.3f, k=%5.3f, x0=%5.3f, y0=%5.3f' % tuple(popt))
    # plt.xlabel('giorni dopo 19 febbraio')
    # plt.ylabel(ylabel)
    # plt.legend()
    # plt.show()

    # plt.figure(2)

    total_xaxis = np.array(list(range(-5, 60))) - len(ydata)
    plt.plot(total_xaxis, sigmoid_derivative(total_xaxis, *popt), 'g-',
             label='fit: L=%5.3f, k=%5.3f, x0=%5.3f' % tuple(popt))
    # y = np.zeros((50))
    # y[:len(ydata)] = ydata
    # plt.plot(list(range(50)), y, 'b-', label='data')
    plt.plot(xdata, ydata, 'b-', label='data')
    plt.xlabel('giorni (0 = oggi)')
    plt.ylabel(ylabel)
    plt.title(title + ' - ' + str(datetime.date.today().strftime("%d-%m-%Y")))
    # plt.legend()
    plt.show()

data = pd.read_csv('data.csv')

ydata = data['totale_casi'].tolist()
fit_sigmoid(ydata, 'Contagi', 'contagiati')

ydata = data['deceduti'].tolist()
fit_sigmoid(ydata, 'Deceduti', 'deceduti')

ydata = data['ricoverati_con_sintomi'].tolist()
fit_sigmoid(ydata, 'Ricoverati', 'ricoverati')

ydata = data['terapia_intensiva'].tolist()
fit_sigmoid(ydata, 'Terapia Intensiva', 'in terapia')

ydata = data['nuovi_attualmente_positivi'].tolist()
fit_sigmoid_derivative(ydata, 'Nuovi positivi', 'nuovi positivi')

