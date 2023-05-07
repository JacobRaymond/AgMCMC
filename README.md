# AgMCMC
This package contains my solution. a simpler version of the problem posed here: https://warwick.ac.uk/fac/sci/statistics/postgrad/research/statmed-compstat/compstat/estmc. The goal is to aggregate the results of multiple Monte Carlo variance reduction methods into a dynamic learner.

A major reference is "Monte Carlo Methods in Financial Engineering" by Paul Glasserman, 2003, Springer.

POC_Final is the final code, with varred containing dependency functions.

## Description of the method

Monte Carlo methods are a useful class of algorithms to numerically evaluate values that might otherwise be difficult to capture. By drawing a large sample from a given distribution, we are able to estimate the values of given variables. However, by definition, Monte Carlo methods are probabilistic. Therefore, any time a simulation is run, the results might be slightly different. 

Given the importance of precision for certain cases that call for Monte Carlo methods, a class of so-called "variance reduction" methods have been derived over the past centure. The aim of these methods is to modify or restrict the draws, thus reducing the distance between draws and ensuring more consistent estimates. While there are some general rules of thumb, no one method is the most appropriate for any kind of process.

Based on the machine learning theory of ensemble learning, I wanted to determine whether it was possible to create an "ensemble variance reduction algorithm". This paradigm, like its machine learning analogs, would be able to achieve a better performance than any single method might on its own.The question of performance, here, is tricky, for we want to reduce variance without sacrificing the accuracy of the algorithms. Still, the question is worth exploring. If a single algorithm can mitigate the needs to test out different methods on a data set, and variance reduction is ultimately the final goal, then a small amount of bias could be acceptable.

My ensemble learner is comprised of four methods:
- Control variates: we correct the error of our target variable by substracting from the draws the errors of a variable with known values and a strong correlation to the target variable.
- Antithetic variables: we introduce negative correlation between pairs of draws and take their mean. Thus, while the estimates will still be valid draws, their variance will be diminished.
- Stratified sampling: the sample space is divided into a set of equal-sized strata. We sample a number of observations from each stratum proportional to its draw probability. 
- Linear Congruential Generator (LCG): inspired by "quasi-Monte Carlo", sampling using an LCG allows for a more thorough exploration of the sample space. Thus, repeated draws, while remaining probabilistic, tend to cover a comparable area of the sample space, as opposed to the more chaotic exploration done by random draws. 

The weight applied to each method is proportional to its individual variance reduction potential.

## File description
POC.R : A crude implementation of the Monte Carlo approach https://medium.com/analytics-vidhya/monte-carlo-simulations-for-predicting-stock-prices-python-a64f53585662. Coded in R and with no real regards to optimization: the goal is to obtain a baseline idea of prediction in finance using a simple GBM stock model. We attempt to predict the share price of Alphabet (GOOG) 50 days in the future, using 50 starting points. 

POC.py : A basic Python implementation of POC.R. 

Control Variates.py: A version of POC.py with a control variate implementation. The control variates are simply the standard normal values that were generated for the simulation. Refer to Glasserman (2003) section 4.1.

Antithetic Variables.py: A version of POC.py with an antithetic variables implementation. The control variates are simply the negative of the standard normal values that were generated for the simulation. Refer to Glasserman (2003) section 4.2.

Strat Sampling.py: A version of POC.py with a stratified sampling implementation. The stratification employed here concerns the terminal value of the Brownian motion. Refer to Glasserman (2003) section 4.3.

LCG.py: A version of POC.py where the paths were generated using a linear congruential generator (a=17364) with Cranley-Patterson Rotation. Source: "Monte Carlo and Quasi-Monte Carlo Sampling" by Christiane Lemieux, 2009, Springer.
