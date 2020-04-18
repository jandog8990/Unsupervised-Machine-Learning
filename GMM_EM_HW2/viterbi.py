#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 13:10:45 2020

Viterbi Algorith for decoding hidden states
Uses backtracking to find the most probable states

@author: alejandrogonzales
"""

import numpy as np

def viterbi(V, A, B, init_dist):
    T = V.shape[0];
    M = A.shape[0];
    
    # highest state probability along single path for first t observations
    # initialize the visible state at t = 0
    t = 0;
    k = V[t];
    omega = np.zeros((T, M));
    omega[0, :] = np.log(init_dist * B[:, k]);
    prev = np.zeros((T-1, M));  # all previous states
    
    # Akin to the forward algorithm in time T and states M
    for t in range(1, T):
        k = V[t];
        for j in range(M):
            # same as the forward prob. but summing probs
            prob = omega[t-1] + np.log(A[:, j]) + np.log(B[j, k])
            
            # index of most probable state given previous state at time t (1)
            prev[t-1, j] = np.argmax(prob);
            
            # probability of most probable state (2)
            omega[t, j] = np.max(prob); # at each time step take most probable
            
            # print info for sanity
#            print("State probabilities:"); print(prob);
#            print("Most probable state index = " + str(prev[t-1, j]));
#            print("Probability of most prob. state = " + str(omega[t, j]));
#            print("\n");
            
    # Print info
    print("Previous state indices:"); print(prev[200:210, :]); print("\n");
    print("Most probable states:"); print(omega[200:210, :]); print("\n");
            
    # State path array
    S = np.zeros(T);
    
    # Most probable last hidden state
    last_state = np.argmax(omega[T-1, :]);
    
    # set the initial state path value
    S[0] = last_state;
    
    # backtracking for accruing the probable states
    backtrack_index = 1;
    for i in range(T-2, -1, -1):
        S[backtrack_index] = prev[i, int(last_state)];
        last_state = prev[i, int(last_state)];
        backtrack_index += 1;
        
    # flip the path array since we're backtracking
    S = np.flip(S, axis=0)
    print("State path:");print(S[200:210]); print("\n");
    
    # Convert numeric states to result
    result = [];
    for s in S:
        if s == 0:
            result.append("A");
        else:
            result.append("B");
        
    return result;
    