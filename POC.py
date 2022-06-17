# -*- coding: utf-8 -*-
"""
# An implementation of https://medium.com/analytics-vidhya/monte-carlo-simulations-for-predicting-stock-prices-python-a64f53585662

This is a Python Translation of a code originally written in R.
"""

import yfinance as yf
import numpy
import random

#Inputs
fwd=50
rep=100000
k=10
train=500

#Load data (Google's Closing Data)
GOOG=yf.Ticker('GOOG').history(start='2010-01-01').Close

#Calculate Returns
GRet=numpy.diff(numpy.log(GOOG))

#Select 10 dates at random - these will serve as our test cases
#Let's give ourselves at least 500 training observations
random.seed(1)
samp=random.choices(range(train, len(GRet)-fwd), k=k)
df=[GRet[1:x] for x in samp]

pred = numpy.empty(shape=(k,fwd),dtype='object')

for y in range(1,k):

    #Calculate drift
    mu=numpy.mean(df[y])
    s=numpy.std(df[y])
    d=mu-0.5*(s**2)
    
    #Generate volatilities
    Z=s*numpy.random.normal(0, 1, size=(rep, fwd))
    
    #Evaluate the rates of return
    r=Z+d
    
    #Simulate paths
    S = numpy.empty(shape=(rep,fwd),dtype='object')
    S[:,0]=GOOG[len(df[y])]
    for i in range(1,fwd):
        S[:,i]=S[:,i-1]*numpy.exp(r[:,i])
    
    pred[y,:]=numpy.mean(S, axis=0)
    
#Find mean absolute relative error difference
trueval=[GOOG[(x):(x+fwd)] for x in samp]
pred_diff=[numpy.abs(pred[x]-trueval[x])/trueval[x] for x in range(1,k)]
print(numpy.mean(pred_diff, axis=0)) #Again, about a 5% MARE - fine performance
