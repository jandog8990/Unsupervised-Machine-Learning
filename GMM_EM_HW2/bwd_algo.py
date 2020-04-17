#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 13:59:39 2020

@author: alejandrogonzales
"""

"""
Backward algorithm calculates the beta probs. that the machine will be
in hidden state S at time step t and will generate remaining
V^T visible states in the sequence

@params:
    A - transition matrix
    B - emission matrix
    beta - reverse hidde state system prbabilities
    V - visible state vector
"""
import numpy as np

def bwd_algo(A, B, V):
    
    
    # Intiilize beta probabilities for future visible states and prob.
    # of the current state Si occurring
    beta = np.zeros((V.shape[0], A.shape[0]))
    
    # Initialize array sizes for time and hidden state space
    T = V.shape[0];
    M = A.shape[0];
    print("Time T = " + str(T))
    print("H-states = " + str(M));
    
    # Set the first beta probabilities 
    # NOTE: Due to the conditional probability p(xn, zn | zn-1)
    # the initial probabilitie will all be 1 for all states
    t = T-1;
    beta[t, :] = np.ones((M));
    
    # Run reverse loop through the visible states V^T
    # starting from T-1 with 0 index => T-2 to 0
    count = 0;
    for t in range(T-2, -1, -1):
        k = V[t];
        # Loop through the number hidden states in A (Bwd Algo)
        for j in range(M):
            beta[t, j] = (beta[t + 1, :]*B[:, k]).dot(A[j, :])
            
    return beta