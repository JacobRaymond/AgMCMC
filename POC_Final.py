# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 18:59:55 2022

@author: jacob
"""

import yfinance as yf
import numpy as np
import sklearn.model_selection as ms

#Function for control variate #NEED TO CLEAN
#SEPARATE FILE? NEED PREDICTION AND VARIANCE REDUCTION
def convar (df, rep):
    
    fwd=round(len(df)*0.2)
    
    #Calculate drift
    mu=np.mean(df)
    s=np.std(df)
    d=mu-0.5*(s**2)
    
    #Generate volatilities
    Z=s*np.random.normal(0, 1, size=(rep, fwd))
    
    #Evaluate the rates of return
    r=Z+d
    
    #Simulate paths
    S = np.empty(shape=(rep,fwd),dtype='object')
    S[:,0]=df[len(df)-1]
    for i in range(1,fwd):
        S[:,i]=S[:,i-1]*np.exp(r[:,i])
    
    #Control variate
    Z1=Z/s
    S1 = np.empty(shape=(rep,fwd),dtype='object')        
    b=np.cov(m=S.astype(float).flatten(), y=Z1.flatten())[0][1]
    S1=S-b*Z1
    
    #Output variance reduction
    pred=np.mean(S1, axis=0)
    vrr=np.var(S1[:, 1:], axis=0)/np.var(S[:, 1:], axis=0)
    
    return pred, np.mean(vrr)

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
blah=convar(trainf[1], rep)
