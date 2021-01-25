# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 19:49:59 2021

@author: DANIEL
"""
# importanto la api de statsmodels
import pandas as pd
import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
# import seaborn
import seaborn as sns
from pandas_datareader import data
import matplotlib 
# importanto la api de statsmodels
import statsmodels.formula.api as smf
import statsmodels.api as sm
# Asistente estadistico 
from scipy import stats

#Nombramos las variables 

tickers = ["CVX","DJD","FENY","IGE","IXC","IYE","JHME","NANR","VDE","XLE"]
start_date = '2018-01-06'
end_date = '2021-01-06'

#Las descargamos de Yhoo Finance 
panel_data = data.DataReader(tickers, 'yahoo', start_date, end_date)
panel_data.head(9)

# Getting all weekdays between 2018-01-06 and 2021-01-06
all_weekdays = pd.date_range(start=start_date, end=end_date, freq='B')


#Llamamos los precios de cierre 
close = panel_data['Close']
close = close.reindex(all_weekdays)
close = close.fillna(method='ffill')
close = close.iloc[1:,]
close.head(10)

#Descrición de los precios de cierre por ETF´s 
close.describe()

# Obtenga la serie temporal de CVX. Esto ahora devuelve un objeto de la serie Pandas indexado por fecha.
cvx = close.loc[:, 'CVX']

# Calcule las medias móviles de 20 y 100 días de los precios de cierre (Promedio )
short_rolling_cvx = cvx.rolling(window=20).mean()
long_rolling_cvx = cvx.rolling(window=100).mean()

# Plot everything by leveraging the very powerful matplotlib package
fig, ax = plt.subplots(figsize=(16,9))

ax.plot(cvx.index, cvx, label='CVX')
ax.plot(short_rolling_cvx.index, short_rolling_cvx, label='20 days rolling')
ax.plot(long_rolling_cvx.index, long_rolling_cvx, label='100 days rolling')

ax.set_xlabel('Date')
ax.set_ylabel('Adjusted closing price ($)')
ax.legend()

#Vamos a sacar los retornos de  CVX con sus ETF´s
returns=close.pct_change()
returns=returns.iloc[1:,]
returns.head(5)

#Realizamos una matriz de correlación 
CorrDJD=np.corrcoef(close.loc[:, 'CVX'],close.loc[:, 'DJD'])
CorrFENY=np.corrcoef(close.loc[:, 'CVX'],close.loc[:, 'FENY'])
CorrIGE=np.corrcoef(close.loc[:, 'CVX'],close.loc[:, 'IGE'])
CorrIXC=np.corrcoef(close.loc[:, 'CVX'],close.loc[:, 'IXC'])
CorrIYE=np.corrcoef(close.loc[:, 'CVX'],close.loc[:, 'IYE'])
CorrJHME=np.corrcoef(close.loc[:, 'CVX'],close.loc[:, 'JHME'])
CorrNANR=np.corrcoef(close.loc[:, 'CVX'],close.loc[:, 'NANR'])
CorrVDE=np.corrcoef(close.loc[:, 'CVX'],close.loc[:, 'VDE'])
CorrXLE=np.corrcoef(close.loc[:, 'CVX'],close.loc[:, 'XLE'])

Correlación = np.array([CorrDJD[-1][0],CorrFENY[-1][0],CorrIGE[-1][0],CorrIXC[-1][0],CorrIYE[-1][0],CorrJHME[-1][0],CorrNANR[-1][0],CorrVDE[-1][0],CorrXLE[-1][0]])

#DJD = 0.355049  NO
#FENY = 0.928119
#IGE = 0.913764
#IXC = 0.951957
#IYE = 0.934711
#JHME = 0.912108
#NANR = 0.710339  NO
#VDE = 0.92851
#XLE = 0.945447

est = smf.ols(formula='CVX ~ DJD+FENY+IGE+IXC+IYE+JHME+NANR+VDE+XLE', data=returns).fit()
est.summary()
