#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 12:44:14 2020

Forward Algorithm - HMM Fwd-Bwd Algo 

@author: alejandrogonzales
"""

"""
Foward algorithm calculates the alpha probs. that the machine will be
at hidden state S at time t after emitting t visible symbols

@params:
    A - transition matrix
    B - emission matrix
    init_dist - initial state probabilities
    V - visible state vector
"""
import numpy as np

def fwd_algo(A, B, init_dist, V):
    
    # Initialize the alpha probability of the hidden state occuring at
    # time t after emitting t visible symbols
    alpha = np.zeros((V.shape[0], A.shape[0]))
    
    
    # Initialize array sizes for time and hidden state space
    T = V.shape[0];
    M = A.shape[0];
#    print("Time T = " + str(T))
#    print("H-states = " + str(M));
    
    # Set the first value of the FWD algorithm
    t = 0;
    k = V[t];
    alpha[t, :] = init_dist * B[:, k];
    print("alpha[t=0, :] = " + str(alpha[t, :]));
    
    # Loop through the number of data points in V^T 
    for t in range(1,T):
        k = V[t];   # visible state at time t
        # Loop through the number hidden states in A (FWD Algo)
        for j in range(M):
            alpha[t, j] = B[j, k] * alpha[t - 1].dot(A[:, j])
            
    return alpha
            

