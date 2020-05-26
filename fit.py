import numpy as np
import pandas as pd
from datetime import timedelta
from datetime import datetime
from io import StringIO
from urllib import request as url_request
import os
import sys

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

coeff_std = 4 # coefficient that defines how many standard deviations to use
coeff_std_d = 2

old_pred_days = 7

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

def logistic_2_ord_derivative(x, L, k, x0):
    """
    General Gaussian like derivative function (derivative of the logistic derivative).
    """
    y = (k**2 * L * np.exp(k*(x+x0)) * (np.exp(k*x0) - np.exp(k*x))) / np.power(np.exp(k*x) + np.exp(k*x0), 3)
    return y


def moving_average(x, w):
    """
    Moving average function that takes an imput x and computes the average with the last w elements.

    Args:
        x    array-like, the input time series
        w    int, the window size for the avg
    """
    return np.convolve(x, np.ones(w), 'valid') / w

def check_style(style):
    if style == 'cyberpunk':
        try:
            global mplcyberpunk
            import mplcyberpunk
            plt.style.use("cyberpunk")
        except:
            style = 'normal'
            print("You need to have mplcyberpunk installed to use the cyberpunk style:")
            print("    pip install mplcyberpunk")
    return style

def fit_curve(curve, ydata, title, ylabel, last_date, coeff_std, avg=0, do_imgs=False, style='normal', old_pred=False):
    style = check_style(style)

    xdata = np.array(list(range(-len(ydata), 0))) + 1

    if curve.__name__ == 'logistic':
        max_val = max(ydata)
        p0=[max_val, 0.5, 1, 0]
        bounds=(-np.inf, np.inf) #([10, 0.1, -100, 0], [1000000, 10, 100, 1])
        params_names = ['L', 'k', 'x0', 'y0']
    elif curve.__name__ == 'logistic_derivative':
        p0=None #[100000, 0.5, 1]
        bounds=(-np.inf, np.inf) #([10, 0.1, -100], [1000000, 10, 100])
        params_names = ['L', 'k', 'x0']
    elif curve.__name__ == 'logistic_2_ord_derivative':
        max_val = sum(ydata) / 2 # empirical value...
        p0=[max_val, 0.2, 0]
        bounds=([10, 0.01, -100], [1000000, 0.5, 100])
        params_names = ['L', 'k', 'x0']
    else:
        print('this curve is unknown')
        return -1

    popt, pcov = curve_fit(curve, xdata, ydata, p0=p0, bounds=bounds, maxfev=7000)

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

    if avg > 1 and len(ydata) > avg + 1:
        real_data = moving_average(ydata, avg)
        real_label = 'real data avg'
        date_xdata = date_xdata[-len(real_data):]
    else:
        real_data = ydata
        real_label = 'real data'

    if style == 'cyberpunk': # leave default colors for cyberpunk
        ax.plot(date_total_xaxis, curve(total_xaxis, *popt), label='prediction')
        ax.plot(date_xdata, real_data, label=real_label)
    else:
        ax.plot(date_total_xaxis, curve(total_xaxis, *popt), 'g-', label='prediction')
        ax.plot(date_xdata, real_data, 'b-', label=real_label)

    if old_pred and len(ydata) > old_pred_days + 1:
        popt, pcov = curve_fit(curve, xdata[:-old_pred_days], ydata[:-old_pred_days], p0=p0, bounds=bounds, maxfev=7000)
        ax.plot(date_total_xaxis, curve(total_xaxis, *popt), label='old prediction')

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

    if style == 'cyberpunk':
        mplcyberpunk.add_glow_effects()

    if do_imgs:
        plt.savefig('imgs/' + title + '.png', dpi=200)
        plt.clf()
    else:
        plt.show()

    return popt, perr

def plot_data(ydata, ylabel, title, last_date, avg=0, do_imgs=False, style='normal'):
    style = check_style(style)

    fig, ax = plt.subplots(figsize=(15,8))
    ax.xaxis.set_major_formatter(myFmt)
    fig.autofmt_xdate()

    if avg > 1:
        ydata = moving_average(ydata, avg)

    xdata = np.array(list(range(-len(ydata), 0))) + 1
    date_xdata = [last_date + timedelta(days=int(i)) for i in xdata]
    if style == 'cyberpunk': # leave default colors for cyberpunk
        ax.plot(date_xdata, ydata, label='real data')
    else:
        ax.plot(date_xdata, ydata, 'b-', label='real data')

    ax.set_xlabel('Giorni - date')
    ax.set_ylabel(ylabel)
    ax.set_title(title + ' - ' + str(last_date.strftime("%d-%m-%Y")))
    ax.legend(loc='upper left')
    ax.grid(True)
    
    if style == 'cyberpunk':
        mplcyberpunk.add_glow_effects()

    if do_imgs:
        plt.savefig('imgs/' + title + '.png', dpi=200)
        plt.clf()
    else:
        plt.show()

if __name__ == '__main__':
    import argparse
    from str2bool import str2bool
    parser = argparse.ArgumentParser(
        description='Modello COVID-19 in Italia.')
    parser.add_argument(
        '--img',
        type=str,
        default="n",
        help='y, save imgs - n do not save imgs')
    parser.add_argument(
        '--avg',
        type=int,
        default=0,
        help='if > 1 draw plot of avg last --avg days.')
    parser.add_argument(
        '--style',
        type=str,
        default="normal",
        help='[normal, cyberpunk] : normal, standard mpl - cyberpunk, cyberpunk style')
    parser.add_argument(
        '--old_pred',
        type=str2bool,
        default=False,
        help='if True plot also the prediction curve from a week ago.')

    args = parser.parse_args()

    do_imgs = False

    if args.img == "y":
        do_imgs = True
        import matplotlib
        matplotlib.use('Agg')
        os.makedirs('imgs/', exist_ok=True)
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

    # Download and read data -----------------------------------

    url = 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv'
    webFile = url_request.urlopen(url).read()
    webFile = webFile.decode('utf-8')  

    data = pd.read_csv(StringIO(webFile))

    # Parse data and compute time series -----------------------

    totale_casi = data['totale_casi'].tolist()
    deceduti = data['deceduti'].tolist()
    ricoverati_con_sintomi = data['ricoverati_con_sintomi'].tolist()
    terapia_intensiva = data['terapia_intensiva'].tolist()
    dimessi_guariti = data['dimessi_guariti'].tolist()
    # nuovi_positivi = data['nuovi_positivi'].tolist()
    tamponi_totali = np.array(data['tamponi'].tolist())

    totale_casi = np.array(totale_casi)
    nuovi = totale_casi[1:] - totale_casi[:-1]

    deceduti = np.array(deceduti)
    nuovi_deceduti = deceduti[1:] - deceduti[:-1]

    dimessi_guariti = np.array(dimessi_guariti)
    nuovi_guariti = dimessi_guariti[1:] - dimessi_guariti[:-1]

    gf_list = nuovi[1:] / nuovi[:-1]

    ricoverati_con_sintomi = np.array(ricoverati_con_sintomi)
    nuovi_ricoverati = ricoverati_con_sintomi[1:] - ricoverati_con_sintomi[:-1]

    terapia_intensiva = np.array(terapia_intensiva)
    nuovi_terapia_intensiva = terapia_intensiva[1:] - terapia_intensiva[:-1]

    totale_attualmente_positivi = totale_casi - deceduti - dimessi_guariti

    nuovi_tamponi = tamponi_totali[1:] - tamponi_totali[:-1]

    tasso_mortalita = nuovi_deceduti / totale_attualmente_positivi[:-1]

    # Print stats ---------------------------------------------

    date_string = data.iloc[-1:]['data'].values[0]
    # date_format = "%Y-%m-%d %H:%M:%S" # Old date format (changed 25/03/2020)
    date_format = "%Y-%m-%dT%H:%M:%S"
    last_date = datetime.strptime(date_string, date_format)
    print("Ultimo aggiornamento: {}".format(last_date))

    totale_casi_oggi = totale_casi[-1]
    print('Tot contagiati: {}'.format(totale_casi_oggi))

    totale_guariti = dimessi_guariti[-1]
    print('Tot dimessi guariti: {}'.format(totale_guariti))

    totale_deceduti = deceduti[-1]
    print('Tot deceduti: {}'.format(totale_deceduti))

    totale_positivi = totale_attualmente_positivi[-1]
    print('Tot attualmente positivi: {}'.format(totale_positivi))

    nuovi_oggi = nuovi[-1]
    print('Tot Nuovi casi oggi: {}'.format(nuovi_oggi))

    decessi_oggi = nuovi_deceduti[-1]
    print('Tot Nuovi decessi oggi: {}'.format(decessi_oggi))

    guariti_oggi = nuovi_guariti[-1]
    print('Tot Nuovi guariti oggi: {}'.format(guariti_oggi))

    tot_tamponi = data.iloc[-1:]['tamponi'].values[0]
    print('Tot tamponi: {}'.format(tot_tamponi))

    growth_factor = gf_list[-1]
    print('Fattore di crescita: {:.3f}'.format(growth_factor))

    avg_growth_factor = np.mean(gf_list[-4:])
    print('Fattore di crescita mediato: {:.3f}'.format(avg_growth_factor))

    print(gf_list)

    # Fit curves and generate plots ---------------------------------

    p_cont, err_cont = fit_curve(logistic, totale_casi, 'Contagi', 'totale contagiati', last_date, coeff_std, args.avg, do_imgs, args.style, old_pred=args.old_pred)

    fit_curve(logistic_derivative, nuovi, 'Nuovi Contagiati', 'nuovi contagiati', last_date, coeff_std_d, args.avg, do_imgs, args.style, old_pred=args.old_pred)


    p_dead, err_dead = fit_curve(logistic, deceduti, 'Deceduti', 'totale deceduti', last_date, coeff_std, args.avg, do_imgs, args.style, old_pred=args.old_pred)

    fit_curve(logistic_derivative, nuovi_deceduti, 'Nuovi Deceduti', 'nuovi deceduti', last_date, coeff_std_d, args.avg, do_imgs, args.style, old_pred=args.old_pred)


    p_hosp, err_hosp = fit_curve(logistic_derivative, ricoverati_con_sintomi, 'Ricoverati', 'totale ricoverati', last_date, coeff_std, args.avg, do_imgs, args.style, old_pred=args.old_pred)

    fit_curve(logistic_2_ord_derivative, nuovi_ricoverati, 'Nuovi Ricoverati', 'nuovi ricoverati', last_date, coeff_std_d, args.avg, do_imgs, args.style, old_pred=args.old_pred)


    p_intens, err_intens = fit_curve(logistic_derivative, terapia_intensiva, 'Terapia Intensiva', 'totale in terapia', last_date, coeff_std, args.avg, do_imgs, args.style, old_pred=args.old_pred)

    fit_curve(logistic_2_ord_derivative, nuovi_terapia_intensiva, 'Nuovi in Terapia Intensiva', 'nuovi in terapia', last_date, coeff_std_d, args.avg, do_imgs, args.style, old_pred=args.old_pred)


    p_healed, err_healed = fit_curve(logistic, dimessi_guariti, 'Dimessi Guariti', 'totale dimessi guariti', last_date, coeff_std_d, args.avg, do_imgs, args.style, old_pred=args.old_pred)
    
    fit_curve(logistic_derivative, nuovi_guariti, 'Nuovi Guariti', 'nuovi guariti', last_date, coeff_std_d, args.avg, do_imgs, args.style, old_pred=args.old_pred)


    fit_curve(logistic_derivative, totale_attualmente_positivi, 'Attualmente Positivi', 'positivi', last_date, coeff_std_d, args.avg, do_imgs, args.style, old_pred=args.old_pred)


    # Plot number of tests and % of positives --------------------------

    plot_data(nuovi_tamponi, 'tamponi al giorno', 'Tamponi Giornalieri', last_date, args.avg, do_imgs, args.style)

    plot_data(nuovi/nuovi_tamponi, '% nuovi', 'Nuovi positivi %', last_date, args.avg, do_imgs, args.style)

    plot_data(tasso_mortalita, '% mortalità', 'Tasso di mortalità %', last_date, args.avg, do_imgs, args.style)
    

