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

#Load data (Google's Closing Data)
GOOG=yf.Ticker('GOOG').history(start='2010-01-01').Close

#Calculate Returns
GRet=numpy.diff(numpy.log(GOOG))

#Select 10 dates at random - these will serve as our test cases
#Let's give ourselves at least 500 training observations
random.seed(1)
samp=random.choices(range(500, len(GRet)-fwd), k=10)
df=[GRet[1:x] for x in samp]

#FIRST CASE OF LOOP
y=df[1]
print(y)

#Calculate drift
mu=numpy.mean(y)
s=numpy.std(y)
d=mu-0.5*(s**2)
print(d)