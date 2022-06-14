# AgMCMC
My solution to a simpler version of the problem posed here: https://warwick.ac.uk/fac/sci/statistics/postgrad/research/statmed-compstat/compstat/estmc. The goal is to aggregate the results of multiple Monte Carlo variance reduction methods into a dynamic learner.

## File description
POC.R : A crude implementation of the Monte Carlo approach https://medium.com/analytics-vidhya/monte-carlo-simulations-for-predicting-stock-prices-python-a64f53585662. Coded in R and with no real regards to optimization: the goal is to obtain a baseline idea of prediction in finance using a simple GBM stock model. We attempt to predict the share price of Alphabet (GOOG) 50 days in the future, using 50 starting points. 

POC.py : A Python implementation of POC.R. Work in progress.
