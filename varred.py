import numpy as numpy
from scipy.stats import norm
import random
import math


def StMc (df, rep):
    fwd=round(len(df)*0.2)
    
    #Calculate drift
    mu=numpy.mean(df)
    s=numpy.std(df)
    d=mu-0.5*(s**2)
    
    #Generate volatilities
    Z=s*numpy.random.normal(0, 1, size=(rep, fwd))
    
    #Evaluate the rates of return
    r=Z+d
    
    #Simulate paths
    S = numpy.empty(shape=(rep,fwd),dtype='object')
    S[:,0]=df[len(df)-1]
    for i in range(1,fwd):
        S[:,i]=S[:,i-1]*numpy.exp(r[:,i])
        
    #Output prediction
    pred=numpy.mean(S, axis=0)
    vr=numpy.var(S[:, 1:], axis=0)
    
    return pred, numpy.mean(vr)

def convar (df, rep):
    
    fwd=round(len(df)*0.2)
    
    #Calculate drift
    mu=numpy.mean(df)
    s=numpy.std(df)
    d=mu-0.5*(s**2)
    
    #Generate volatilities
    Z=s*numpy.random.normal(0, 1, size=(rep, fwd))
    
    #Evaluate the rates of return
    r=Z+d
    
    #Simulate paths
    S = numpy.empty(shape=(rep,fwd),dtype='object')
    S[:,0]=df[len(df)-1]
    for i in range(1,fwd):
        S[:,i]=S[:,i-1]*numpy.exp(r[:,i])
    
    #Control variate
    Z1=Z/s
    S1 = numpy.empty(shape=(rep,fwd),dtype='object')        
    b=numpy.cov(m=S.astype(float).flatten(), y=Z1.flatten())[0][1]
    S1=S-b*Z1
    
    #Output variance reduction
    pred=numpy.mean(S1, axis=0)
    vrr=numpy.var(S1[:, 1:], axis=0)/numpy.var(S[:, 1:], axis=0)
    
    return pred, numpy.mean(vrr)

def antvar (df, rep):
    
    fwd=round(len(df)*0.2)
    
    #Calculate drift
    mu=numpy.mean(df)
    s=numpy.std(df)
    d=mu-0.5*(s**2)
    
    #Generate volatilities
    Z=s*numpy.random.normal(0, 1, size=(rep, fwd))
    
   #Evaluate the rates of return
    r=Z+d
    
    #Simulate paths
    S = numpy.empty(shape=(rep,fwd),dtype='object')
    S[:,0]=df[len(df)-1]
    for i in range(1,fwd):
        S[:,i]=S[:,i-1]*numpy.exp(r[:,i])
    
    #Antithetic Variables
    r1=d-Z
    S1 = numpy.empty(shape=(rep,fwd),dtype='object')
    S1[:,0]=df[len(df)-1]
    for i in range(1,fwd):
        S1[:,i]=S1[:,i-1]*numpy.exp(r1[:,i])
    S2=(S+S1)/2
    
    #Output variance reduction
    pred=numpy.mean(S2, axis=0)
    vrr=numpy.var(S2[:, 1:], axis=0)/numpy.var(S[:, 1:], axis=0)
    
    return pred, numpy.mean(vrr)

def strsam (df, rep):
    
    fwd=round(len(df)*0.2)
    
    #Calculate drift
    mu=numpy.mean(df)
    s=numpy.std(df)
    d=mu-0.5*(s**2)
    
    #Generate volatilities
    Z=s*numpy.random.normal(0, 1, size=(rep, fwd))
    
   #Evaluate the rates of return
    r=Z+d
    
    #Simulate paths
    S = numpy.empty(shape=(rep,fwd),dtype='object')
    S[:,0]=df[len(df)-1]
    for i in range(1,fwd):
        S[:,i]=S[:,i-1]*numpy.exp(r[:,i])
        
    #Terminal stratified sampling of the Brownian motion, see Glasserman p.223
    W=numpy.empty(shape=(rep, fwd),dtype='object')
    W[:,0]=Z[:,0]/s
    for i in range(0,rep): #Final observation
        W[i,fwd-1]=norm.ppf((i+random.random())/rep)
    for j in range(2,fwd): #Remaining observations
        W[:,j-1]=((fwd-j)/(fwd-(j-1)))*W[:,j-2]+(1/(fwd-(j-1)))*W[:,(fwd-1)]+math.sqrt((1-j/fwd)/(fwd-(j-1)))*numpy.random.normal(0, 1, rep)
    Z1=numpy.c_[W[:,0], W[:, 1:fwd]-W[:, 0:(fwd-1)]]
    
    #Simulate paths using the stratified samples
    r1=numpy.array((s*Z1)+d, dtype=numpy.float64)
    S1 = numpy.empty(shape=(rep,fwd),dtype='object')
    S1[:,0]=df[len(df)-1]
    for i in range(1,fwd):
        S1[:,i]=S1[:,i-1]*numpy.exp(r1[:,i])
        
    #Output variance reduction
    pred=numpy.mean(S1, axis=0)
    vrr=numpy.var(S1[:, 1:], axis=0)/numpy.var(S[:, 1:], axis=0)
        
    return pred, numpy.mean(vrr)

def lcg (df, rep):
    
    fwd=round(len(df)*0.2)
    
    #Calculate drift
    mu=numpy.mean(df)
    s=numpy.std(df)
    d=mu-0.5*(s**2)
    
    #Generate volatilities
    Z=s*numpy.random.normal(0, 1, size=(rep, fwd))
    
   #Evaluate the rates of return
    r=Z+d
    
    #Simulate paths
    S = numpy.empty(shape=(rep,fwd),dtype='object')
    S[:,0]=df[len(df)-1]
    for i in range(1,fwd):
        S[:,i]=S[:,i-1]*numpy.exp(r[:,i])
        
    #LCG Paths
    a=17364
    n=fwd*rep
    Z1=numpy.zeros(n)
    Z1[1]=1/n
    for i in range(3, n):
        Z1[i-1]=((a*n*Z1[i-2])%n)/n
       
    #Cranley-Patterson rotation
    u=random.random()
    Z1=(Z1+u)%1
   
    #Extract the generated paths
    Z1=s*norm.ppf(Z1)
   
    #Evaluate the rates of return
    r1=Z1+d
   
    #Simulate LCG Paths
    S1 = numpy.empty(shape=(rep,fwd),dtype='object')
    S1[:,0]=df[len(df)-1]
    for i in range(1,fwd):
        S1[:,i]=S1[:,i-1]*numpy.exp(r1[i])
        
    #Output variance reduction
    pred=numpy.mean(S1, axis=0)
    vrr=numpy.var(S1[:, 1:], axis=0)/numpy.var(S[:, 1:], axis=0)
        
    return pred, numpy.mean(vrr)
