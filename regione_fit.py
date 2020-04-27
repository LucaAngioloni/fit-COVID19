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
days_future = 40 # days after the end of the data to predict and plot

myFmt = mdates.DateFormatter('%d/%m') # date formatter for matplotlib
show_every = 3 # int value that defines how often to show a date in the x axis. (used not to clutter the axis)

coeff_std = 3.5 # coefficient that defines how many standard deviations to use
coeff_std_d = 1.5

from fit import logistic, logistic_derivative, logistic_2_ord_derivative, fit_curve, plot_data

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Modello COVID-19 per regione.')
    parser.add_argument(
        '--regione',
        type=str,
        help='Nome regione su cui effettuare le predizioni.',
        required=True)
    parser.add_argument(
        '--img',
        type=str,
        default="n",
        help='y, save imgs - n do not save imgs')
    parser.add_argument(
        '--avg',
        type=int,
        default=0,
        help='if > 0 draw plot of avg last --avg days.')
    parser.add_argument(
        '--style',
        type=str,
        default="normal",
        help='[normal, cyberpunk] : normal, standard mpl - cyberpunk, cyberpunk style')

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

    url = 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv'
    webFile = url_request.urlopen(url).read()
    webFile = webFile.decode('utf-8')  

    data = pd.read_csv(StringIO(webFile))

    regioni_ammesse = str(
        set(data['denominazione_regione'])
        ).replace("'", "")

    data = data[data['denominazione_regione'] == args.regione]

    if len(data) == 0:
        print("Regione non esistente o scritta in modo sbagliato!")
        print("Regioni ammesse:")
        print(regioni_ammesse)
        sys.exit(-1)

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

    avg_growth_factor = np.mean(gf_list[-3:])
    print('Fattore di crescita mediato: {:.3f}'.format(avg_growth_factor))

    print(gf_list)

    # Fit curves and generate plots ---------------------------------

    p_cont, err_cont = fit_curve(logistic, totale_casi, 'Contagi', 'totale contagiati', last_date, coeff_std, args.avg, do_imgs, args.style)

    fit_curve(logistic_derivative, nuovi, 'Nuovi Contagiati', 'nuovi contagiati', last_date, coeff_std_d, args.avg, do_imgs, args.style)


    p_dead, err_dead = fit_curve(logistic, deceduti, 'Deceduti', 'totale deceduti', last_date, coeff_std, args.avg, do_imgs, args.style)

    fit_curve(logistic_derivative, nuovi_deceduti, 'Nuovi Deceduti', 'nuovi deceduti', last_date, coeff_std_d, args.avg, do_imgs, args.style)


    p_hosp, err_hosp = fit_curve(logistic_derivative, ricoverati_con_sintomi, 'Ricoverati', 'totale ricoverati', last_date, coeff_std, args.avg, do_imgs, args.style)

    fit_curve(logistic_2_ord_derivative, nuovi_ricoverati, 'Nuovi Ricoverati', 'nuovi ricoverati', last_date, coeff_std_d, args.avg, do_imgs, args.style)


    p_intens, err_intens = fit_curve(logistic_derivative, terapia_intensiva, 'Terapia Intensiva', 'totale in terapia', last_date, coeff_std, args.avg, do_imgs, args.style)

    fit_curve(logistic_2_ord_derivative, nuovi_terapia_intensiva, 'Nuovi in Terapia Intensiva', 'nuovi in terapia', last_date, coeff_std_d, args.avg, do_imgs, args.style)


    p_healed, err_healed = fit_curve(logistic, dimessi_guariti, 'Dimessi Guariti', 'totale dimessi guariti', last_date, coeff_std_d, args.avg, do_imgs, args.style)
    
    fit_curve(logistic_derivative, nuovi_guariti, 'Nuovi Guariti', 'nuovi guariti', last_date, coeff_std_d, args.avg, do_imgs, args.style)


    # fit_curve(logistic_derivative, totale_attualmente_positivi, 'Attualmente Positivi', 'positivi', last_date, coeff_std_d, do_imgs, args.style)


    # Plot number of tests and % of positives --------------------------

    plot_data(nuovi_tamponi, 'tamponi al giorno', 'Tamponi Giornalieri', last_date, args.avg, do_imgs, args.style)

    plot_data(nuovi/nuovi_tamponi, '% nuovi', 'Nuovi positivi %', last_date, args.avg, do_imgs, args.style)


