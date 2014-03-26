HMM
===

An (Almost-Correct) Implementation of an Auto-Regressive HMM

This repository contains a java implemention of an Auto-Regressive HMM: that is, x(t) ~ Normal(ax(t-1)+b,sigma^2), where the parameters a,b,sigma vary depending on the latent markov state. Though the likelihood decreases monotonically, numerical issues abound. Additionally, the model learns approximately the same transition probability i -> j for each state i. Any ways, here is the code...
