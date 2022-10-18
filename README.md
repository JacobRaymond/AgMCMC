# AgMCMC
My solution to a simpler version of the problem posed here: https://warwick.ac.uk/fac/sci/statistics/postgrad/research/statmed-compstat/compstat/estmc. The goal is to aggregate the results of multiple Monte Carlo variance reduction methods into a dynamic learner.

A major reference is "Monte Carlo Methods in Financial Engineering" by Paul Glasserman, 2003, Springer.

## File description
POC.R : A crude implementation of the Monte Carlo approach https://medium.com/analytics-vidhya/monte-carlo-simulations-for-predicting-stock-prices-python-a64f53585662. Coded in R and with no real regards to optimization: the goal is to obtain a baseline idea of prediction in finance using a simple GBM stock model. We attempt to predict the share price of Alphabet (GOOG) 50 days in the future, using 50 starting points. 

POC.py : A basic Python implementation of POC.R. 

Control Variates.py: A version of POC.py with a control variate implementation. The control variates are simply the standard normal values that were generated for the simulation. Refer to Glasserman (2003) section 4.1.

Antithetic Variables.py: A version of POC.py with an antithetic variables implementation. The control variates are simply the negative of the standard normal values that were generated for the simulation. Refer to Glasserman (2003) section 4.2.

Strat Sampling.py: A version of POC.py with a stratified sampling implementation. The stratification employed here concerns the terminal value of the Brownian motion. Refer to Glasserman (2003) section 4.3.

LCG.py: A version of POC.py where the paths were generated using a linear congruential generator (a=17364) with Cranley-Patterson Rotation. Source: "Monte Carlo and Quasi-Monte Carlo Sampling" by Christiane Lemieux, 2009, Springer.
