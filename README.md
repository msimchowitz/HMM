HMM
===

An (Almost-Correct) Implementation of an Auto-Regressive HMM

This repository contains a java implemention of an Auto-Regressive HMM: that is, x(t) ~ Normal(ax(t-1)+b,sigma^2), where the parameters a,b,sigma vary depending on the latent markov state. Though the likelihood decreases monotonically, numerical issues abound. Additionally, the model learns approximately the same transition probability i -> j for each state i.

The file is called logHmm.java. Updates are implemented in log space to avoid numerical issues. The code allows for multiple separate time series taking place along different paths in a common Markov chain. The code reads in a csv file, and uses this as input for the algorithm.

The intention of the code is to model fluctuations in economic data. 
