# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 14:45:08 2021

@author: DANIEL ESTEBAN CASTIBLANCO MALDONADO
"""
### def ETFsArbitrage (stock,ETFs,startdate,enddate):

##Importamos todas las librerias 
import pandas as pd
import numpy as np
from pandas_datareader import data
import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima_model import ARIMA
import math
import statistics
from pylab import *
import seaborn as sns

##Descargamos los datos
tickers = ["FENY","IGE","IXC","IYE","JHME","VDE","XLE"]
start_date = '2020-05-09'
end_date = '2020-07-08'

start_datef = '2020-05-08'
end_datef = '2020-07-07'

etfs = data.DataReader(tickers, 'yahoo', start_datef, end_datef)
stck = data.DataReader('cvx', 'yahoo', start_date, end_date)
etfs.head(9)
stck.head(9)


##Lo convertimos en una serie de tiempo para que coja los datos diarios de cada semana.
all_weekdaysf = pd.date_range(start=start_datef, end=end_datef, freq='B')
all_weekdays = pd.date_range(start=start_date, end=end_date, freq='B')

##Sacamos los precios de cierre para los ETFs 
closetfs = etfs['Close']
closetfs = closetfs.reindex(all_weekdaysf) 
closetfs = closetfs.fillna(method='ffill') #Encontrar NAN error 
closetfs = closetfs.iloc[1:,] #Resolver error
closetfs.describe() # Describe los precios de cierre por activo. 

##Sacamos los precios de cierre de nuestro stock
closestck = stck['Close']
closestck = closestck.reindex(all_weekdays) 
closestck = closestck.fillna(method='ffill') #Encontrar NAN error 
closestck = closestck.iloc[1:,] #Resolver error
closestck.describe() # Describe los precios de cierre por activo. 


##Graficarla 
# Calcule las medias móviles de 20 y 100 días de los precios de cierre (Promedio )
short_rolling_closestck = closestck.rolling(window=10).mean()
long_rolling_closestck = closestck.rolling(window=20).mean()
# Plot everything by leveraging the very powerful matplotlib package
fig, ax = plt.subplots(figsize=(16,9)) #Tamaño de los subtitulo 

ax.plot(closestck.index, closestck, label='CVX')
ax.plot(short_rolling_closestck.index, short_rolling_closestck, label='10 days rolling')
ax.plot(long_rolling_closestck.index, long_rolling_closestck, label='20 days rolling')

ax.set_xlabel('Date')
ax.set_ylabel('Adjusted closing price ($)')
ax.legend()


##Sacamos los retornos de nuestras ETFs
returnsETF=closetfs.pct_change()
returnsETF=returnsETF.iloc[1:,]
returnsETF.head(5)

##Sacamos los retornos de nuestro Stock
returnstck=closestck.pct_change()
returnstck=returnstck.iloc[1:,]
returnstck.head(5)
plt.plot(returnstck)


###Betas ETFs
##Calculamos nuestra regresión para sacr los betas βˆ=(XTX)^−1 XTY 
#Creamos una matriz de 1
beta0=np.ones(len(returnsETF['FENY']))
beta0 
# Creamos el vector Y  
Y=returnsETF.iloc[:,0] #le estamos diciendo coja el primera columnas de los retornos 
Y.head()
#Creamos una matriz con os retornos y un vector de 1 que equivale al intercepto. 
X=returnsETF.iloc[:,0:10]
X["Beta0"]=beta0
X.head()
# Sacamos un Vector correspondicente a cada beta 
beta_hat = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
beta_hat


##ARIMA
model = ARIMA(returnstck, order=(1,0,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
model_fit.params
Ar1 = model_fit.params[1]
Int1 = model_fit.params[0]


##  PARAMETROS
#K 
kappa=-math.log(abs(Ar1))/252 # Parametro K 
kappa
#m
m=Int1/(1-Ar1 ) #Parametro m 
m
#s
sdchevron = np.std(returnstck)
csdchevron = pow (sdchevron,2)
varc = ((csdchevron /(2*kappa))*(1-(math.exp(-2*kappa*252))))

c= np.random.normal(0,math.sqrt(varc),252)
SigmaAr= math.sqrt((varc*2*kappa)/(1-pow(Ar1, 2))) #Parametro s 


### Variación del Spread 
xn = list(range(60)) 
vSpread = (xn-m)/(SigmaAr/math.sqrt(2*kappa))
print (vSpread)


#### Planteamiento del modelo Ornstein-Uhlembeck 
nsteps = 100 #N
nsims = 1000 #i 
SigmaAr #Volatility in market
t = 1
deltat = t/nsteps
p = range(0,nsteps,1)

#creamos un Valor inicial Rentabilidad mini del activo 
xt0= kappa*m*math.exp(kappa*t)*t + (SigmaAr*math.exp(kappa*t))*math.sqrt(t)*np.random.normal()
#Creamos una matriz de 0
xt = np.zeros([nsims,nsteps])

#Plantiamos la ecuación del modelo 
for y in range(0,nsims-1):
    xt[y,0]=xt0
    for x in range(0,nsteps-1):
        xt[y,x+1] = xt[y,x]*(np.exp((1-(SigmaAr**2)/2)*deltat + SigmaAr*deltat*np.random.normal(0,1)))
    plt.plot(p,xt[y])

plt.title('nsims %d nsteps %d SigmaAr  %.2f xt0 %.2f' % (nsims, nsteps, SigmaAr, xt0))
plt.xlabel('Steps')
plt.ylabel('Ornstein Uhlenbeck')
plt.show()

##Sacamos la diferencia de xt 
dxt = diff(xt)
plt.plot (dxt)

##Sacamos el valor esperado de dxt
a= dxt[:,-1]
Edxt= mean(a)

##Planteamos la ecuación de la regresión 
Rt = beta_hat[-1]+ (beta_hat[0]*(returnsETF['FENY'][-1])) + (beta_hat[1]*(returnsETF['IGE'][-1])) + (beta_hat[2]*(returnsETF['IXC'][-1])) + (beta_hat[3]*(returnsETF['IYE'][-1])) + (beta_hat[4]*(returnsETF['JHME'][-1])) + (beta_hat[5]*(returnsETF['VDE'][-1])) + (beta_hat[6]*(returnsETF['XLE'][-1])) + Edxt
print (Rt)



