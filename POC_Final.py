import yfinance as yf
import numpy as np
import sklearn.model_selection as ms

from varred import *

#Inputs
k=10 #Number of training folds
testnum=500 #Number of total observations in test set
pred=50 #Number of observations to be predicted
rep=10000 #Number of paths to create for each fold. Applicable to every technique


#Load data (Google's Closing Data)
GOOG=yf.Ticker('GOOG').history(start='2010-01-01').Close

#Calculate Returns
GRet=np.diff(np.log(GOOG))

#Create training and test 
train=GRet[0:len(GRet)-testnum]
test=GRet[len(GRet)-testnum:]

#Create folds
trainf=np.array_split(train, 10)

#Save weights
ws=[]

for j in range(k):

    #Control variates
    cv=convar(trainf[j], rep)
    
    #Antithetic Variables
    av=antvar(trainf[j], rep)
    
    #Stratified sampling
    ss=strsam(trainf[j], rep)
    
    #LCG
    lg=lcg(trainf[j], rep)
    
    #Calculate weights
    ws_alg=tuple(1-x/np.sqrt(1+x) for x in (cv[1], av[1], ss[1], lg[1]))
    ws.append(tuple(x/sum(ws_alg) for x in ws_alg))
    
#Extract weights
w=np.mean(ws, axis=0)  

#Predict paths
pths=[]
pths_test=(convar(test[0:testnum-pred], rep)[0], antvar(test[0:testnum-pred], rep)[0], strsam(test[0:testnum-pred], rep)[0], lcg(test[0:testnum-pred], rep)[0])
    
#Weighted average
for i in range(4):
  pths.append(w[i]*pths_test[i])

#Prediction
sum(pths)[0:pred]

#Real values
test[testnum-pred:]  
  
    
