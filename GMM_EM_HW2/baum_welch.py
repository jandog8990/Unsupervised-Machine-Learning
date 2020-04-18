#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 22:15:29 2020

Baum Welch Algorithm - FWD-BWD Algorithm

@author: alejandrogonzales
"""

"""
Combines forward and backward algorithm to compute the 
latent variables [epsilon, gamma] for the EM Algorithm
"""
from fwd_algo import fwd_algo
from bwd_algo import bwd_algo
import numpy as np

def baum_welch(V, A, B, init_dist, n_iter=100):
    T = V.shape[0]
    M = A.shape[0]
#    print("T = " + str(T));
#    print("M = " + str(M));
    
    for n in range(n_iter):
        alpha = fwd_algo(A, B, init_dist, V);
        beta = bwd_algo(A, B, V);
        
        xi = np.zeros((M, M, T-1));
        for t in range(T-1):

            denominator = np.dot(np.dot(alpha[t, :].T, A) * B[:, V[t+1]].T, beta[t + 1, :]);
            for i in range(M):
                numerator = alpha[t, i] * A[i, :] * B[:, V[t+1]].T * beta[t + 1, :].T
                xi[i, :, t] = numerator / denominator
                
        # Re-calculate the A state matrix
        gamma = np.sum(xi, axis=1)
        A = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1));
        
        # Additional T-th element in gamma
        gamma = np.hstack((gamma, np.sum(xi[:, :, T-2], axis=0).reshape((-1, 1))))
        
        # Re-calculate the B visible station matrix
        K = B.shape[1]  # number of visible states to capture
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            B[:, l] = np.sum(gamma[:, V == l], axis=1);
            
        B = np.divide(B, denominator.reshape((-1, 1)))
        
    return {"A": A, "B": B}
