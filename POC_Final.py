import yfinance as yf
import numpy as np
import sklearn.model_selection as ms

from varred import *

#Inputs
k=10 #Number of training folds
testnum=500 #Might be too much?
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

#Control variates
cv=convar(trainf[1], rep)
av=antvar(trainf[1], rep)
ss=strsam(trainf[1], rep)
lg=lcg(trainf[1], rep)
