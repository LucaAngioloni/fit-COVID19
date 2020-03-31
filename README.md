# fit-COVID19
Easy model to fit logistic curve to COVID19 data from Italy.

Data is taken from [this official repo](https://github.com/pcm-dpc/COVID-19)

Live demo: https://fit-covid19.herokuapp.com
(It could be slow because it is a free heroku app)

Predictions for Italy:
```bash
usage: fit.py [-h] [--img IMG]

Modello COVID-19 in Italia.

optional arguments:
  -h, --help  show this help message and exit
  --img IMG   y, save imgs - n do not save imgs
 ```

Predictions for a region:
```bash
usage: regione_fit.py [-h] --regione REGIONE [--img IMG]

Modello COVID-19 per regione.

optional arguments:
  -h, --help         show this help message and exit
  --regione REGIONE  Nome regione su cui effettuare le predizioni.
  --img IMG          y, save imgs - n do not save imgs
```

![Totale contagi](https://fit-covid19.herokuapp.com/imgs/Contagi.png?rel=true "Totale contagi")
![Contagi giornalieri](https://fit-covid19.herokuapp.com/imgs/Nuovi%20Contagiati.png?rel=true "Contagi giornalieri")

If you know this stuff and you think you can contribute please just let me know: fork this repo, pull request, star this repo, send me an email.

### Requirements
- Python >=3
- Pandas
- Numpy
- ScyPy
