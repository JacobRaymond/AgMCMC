import yfinance as yf
import numpy
import random
from scipy.stats import norm
import math

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

#Save results
cms1 = numpy.empty(shape=(k,fwd),dtype='object')
cms2 = numpy.empty(shape=(k,fwd),dtype='object')
vrr = numpy.empty(shape=(k,fwd-1),dtype='object')

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
    
    #Simulate paths
    S1 = numpy.empty(shape=(rep,fwd),dtype='object')
    S1[:,0]=GOOG[len(df[y])]
    for i in range(1,fwd):
        S1[:,i]=S1[:,i-1]*numpy.exp(r1[i]) 

    
    cms1[y,:]=numpy.mean(S, axis=0)
    cms2[y,:]=numpy.mean(S1, axis=0)
    vrr[y, :]=numpy.var(S1[:, 1:], axis=0)/numpy.var(S[:, 1:], axis=0)
    
#Find mean absolute relative error difference
trueval=[GOOG[(x):(x+fwd)] for x in samp]

#Normal MC Simulation
pred_diff1=[numpy.abs(cms1[x]-trueval[x])/trueval[x] for x in range(1,k)]
print(numpy.mean(pred_diff1, axis=1)) #About a 5% MARE - fine performance

#Strat sampling MC Simulation
pred_diff2=[numpy.abs(cms2[x]-trueval[x])/trueval[x] for x in range(1,k)]
print(numpy.mean(pred_diff2, axis=1)) #Slightly higher error, though nothing too drammatic

#Variance reduction
print(numpy.mean(vrr[1:, :], axis=1)) #Massive reduction in variance!
