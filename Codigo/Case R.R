library(quantmod)
library(forecast)
library(tseries)
library(lmtest)

asset.names = c("XOP","CVX")
getSymbols(asset.names, src="yahoo", from="2015-09-07",to="2016-09-07")
CVX=(CVX$CVX.Close)
cvx=Delt(CVX$CVX.Close)[-1]
View(CVX)


tsdatacvx<-ts(CVX,start=c(2015,9,8),frequency=365) # Serie de tiempo diaria
View(tsdatacvx)

#Sacomos el Arima para establecer a y b 
ARIMACVX <- arima(tsdatacvx, order=c(1,0,0))
coeftest(ARIMACVX) # Para mirar si los coheficientes son significaticvos o no 
ARIMACVX$coef
Ar1=ARIMACVX$coef[1]
In1=ARIMACVX$coef[2]

K=-log(Ar1)/252 
m=In1/(1-Ar1 ) 

#Para calcular sigma primero hacemos los errores
sdcvx=(sd(cvx))^2
varc= ((sdcvx/(2*K))*(1-(exp(-2*K*252)))) 
c=rnorm(252,0,varc)

SigmaAr=sqrt((varc*2*K)/(1-(0.9876^2)))

# Calculamos la variaci?n del Spread 
Xn=seq(1,60) #Vector
Xn
s=(Xn-m)/SigmaAr/sqrt(2*K)
View(s)


##Simulaciones residuo dXt
set.seed(123)# Número Pseudo aleatorios
caminatasResiduales <- function(Kapa=K, miu=m, sigma=SigmaAr, nsims, periods) 
{
  # K
  # m
  # sigma Volatilidad de los rendimientos
  # nsim número de simulaciones
  # periods Es un vector
  
  
  nsteps = length(periods)
  dt =diff(periods)[1] #diferencia 
  
  
   
      (Kapa*miu*exp(Kapa*dt))*dt + (sigma*exp(Kapa+dt))*sqrt(dt)*rnorm(nsims)
  
      temp = matrix((Kapa*miu*exp(Kapa*dt))*dt + (sigma*exp(Kapa+dt))*sqrt(dt)*rnorm(nsteps * nsims),nc=nsims) 

    }
  
  
dXt=caminatasResiduales(K,m,SigmaAr,10000,0:60)
tiempo=0:60
matplot(tiempo,dXt[,1:10000], type='l', xlab='días',ylab='Residuales',main='Variaci?n del Spread')









(Kapa*miu*exp(Kapa*dt))*dt + (sigma*exp(Kapa+dt))*sqrt(dt)*rnorm(nsims)

dt = c(periods[1], diff(periods))

periods=10


nsteps=60
nsims=10


dXt=matrix(data=0,nrow=nsteps,ncol=nsims)
t=1
dt=1
dXt0=K*m*exp(K*t)*dt + (SigmaAr*exp(K*t))*sqrt(dt)*rnorm(1)
dXt[1,]=dXt0
Xt=matrix(data=0,nrow=nsteps,ncol=nsims)
Xt[1,]=dXt0
dW=matrix(data=sqrt(dt)*rnorm(nsims*nsteps),nrow=nsteps,ncol=nsims)

K
View(Xt)

for (i in 2:nsteps){

    
  Xt[i,]=exp(-K*dt)*Xt[i-1]+m*(1-exp(-K*dt))+SigmaAr*exp(-K*dt)*dW[i,]
  
  Xt=as.data.frame(Xt)
   
}

b=((Xt)[-1,])
a=((Xt))[-60,]
c= (b-a)
View(c)
View(Xt)
