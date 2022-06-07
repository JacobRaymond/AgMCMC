#An implementation of https://medium.com/analytics-vidhya/monte-carlo-simulations-for-predicting-stock-prices-python-a64f53585662

library(quantmod)
library(magrittr)

#Inputs
fwd=50
rep=10000

#Load data
getSymbols("GOOG")
GOOG=GOOG$GOOG.Close %>% as.vector()

#Calculate returns
G.Ret=GOOG %>% log() %>% diff() 

#Select 10 dates at random - these will serve as our test cases
#Let's give ourselves at least 500 training observations
set.seed(1)
samp=sample(500:(length(G.Ret)-fwd), size=10)
df=sapply(samp, function(x) G.Ret[seq.int(1,x)]) %>% as.list()

#Results

preds=lapply(df, function(x){
  
  #Calculate drift
  mu=mean(x)
  s=sd(x)
  d=mu-0.5*s^2
  
  #Generate the volatilities
  Z=s*rnorm(n = fwd*rep)
  dim(Z)<-c(rep, fwd)
  
  #Evaluate the rates of return
  r=Z+d
  
  #Simulate paths (unfortunately, iteratively)
  S=matrix(nrow=rep, ncol=fwd)
  S[,1]=GOOG[length(x)]
  for(i in 2:ncol(S)){
    S[,i]=S[,i-1]*exp(r[,i])
  }
  
  colMeans(S)
})

#Find mean absolute relative error difference
trueval=sapply(1:10, function(x) GOOG[(samp[x]+1): (samp[x]+fwd)]) %>% as.data.frame()
pred_diff=abs(as.data.frame(preds)-trueval)/trueval
colMeans(pred_diff) #About a 5% MARE - fine performance

        