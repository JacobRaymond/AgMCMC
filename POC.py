{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "aa64db07",
   "metadata": {},
   "source": [
    "# #An implementation of https://medium.com/analytics-vidhya/monte-carlo-simulations-for-predicting-stock-prices-python-a64f53585662\n",
    "\n",
    "This is a Python Translation of a code originally written in R.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "78806f34",
   "metadata": {},
   "outputs": [],
   "source": [
    "import yfinance as yf\n",
    "import numpy as np\n",
    "import random"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "fc8f99bd",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Inputs\n",
    "fwd=50\n",
    "rep=100000"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "9779f3aa",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Load data (Google's Closing Data)\n",
    "GOOG=yf.Ticker('GOOG').history(start='2010-01-01').Close"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "61146b06",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Calculate Returns\n",
    "GRet=np.diff(np.log(GOOG))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "7c4afc89",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([-0.0044134 , -0.02553193, -0.02355476, ..., -0.01998718,\n",
       "       -0.0308447 , -0.0417002 ])"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "GRet"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "befaf04c",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Select 10 dates at random - these will serve as our test cases\n",
    "#Let's give ourselves at least 500 training observations\n",
    "random.seed(1)\n",
    "samp=random.choices(range(500, len(GRet)-fwd), k=10)\n",
    "df=[GRet[1:x] for x in samp]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "d2099714",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([-0.02553193, -0.02355476,  0.01324304, ..., -0.05130917,\n",
       "       -0.03142963, -0.0375593 ])"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#FIRST CASE OF LOOP\n",
    "y=df[1]\n",
    "y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "dfb0f312",
   "metadata": {},
   "outputs": [],
   "source": [
    "  #Calculate drift\n",
    "mu=np.mean(y)\n",
    "s=np.std(y)\n",
    "d=mu-0.5*(s**2)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ed3819fd",
   "metadata": {},
   "source": [
    "Rest of code to translate:\n",
    "#Results\n",
    "\n",
    "preds=lapply(df, function(x){\n",
    "  \n",
    "  #Calculate drift\n",
    "  mu=mean(x)\n",
    "  s=sd(x)\n",
    "  d=mu-0.5*s^2\n",
    "  \n",
    "  #Generate the volatilities\n",
    "  Z=s*rnorm(n = fwd*rep)\n",
    "  dim(Z)<-c(rep, fwd)\n",
    "  \n",
    "  #Evaluate the rates of return\n",
    "  r=Z+d\n",
    "  \n",
    "  #Simulate paths (unfortunately, iteratively)\n",
    "  S=matrix(nrow=rep, ncol=fwd)\n",
    "  S[,1]=GOOG[length(x)]\n",
    "  for(i in 2:ncol(S)){\n",
    "    S[,i]=S[,i-1]*exp(r[,i])\n",
    "  }\n",
    "  \n",
    "  colMeans(S)\n",
    "})\n",
    "\n",
    "#Find mean absolute relative error difference\n",
    "trueval=sapply(1:10, function(x) GOOG[(samp[x]+1): (samp[x]+fwd)]) %>% as.data.frame()\n",
    "pred_diff=abs(as.data.frame(preds)-trueval)/trueval\n",
    "colMeans(pred_diff) #About a 5% MARE - fine performance\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5db0b5fb",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
