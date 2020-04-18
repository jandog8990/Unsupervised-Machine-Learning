#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 00:03:36 2020

Forward Algorithm for the FWD-BWD HMM

@author: alejandrogonzales
"""

# We have 2 Hidden States (A,B) and 3 visible states (0,1,2)
# assume that we already know our probability transitions (a,b)
import pandas as pd
import numpy as np
from fwd_algo import fwd_algo
from bwd_algo import bwd_algo
from baum_welch import baum_welch
from viterbi import viterbi

# Create the A and B matrices
A = np.array([[0.54, 0.46], [0.49, 0.51]]);
B = np.array([[0.16, 0.26, 0.58], [0.25, 0.28, 0.47]]);
print("A:");print(A);
print("B:");print(B);

# Read the CSV and store in a Pandas Table
data = pd.read_csv('data_python.csv');
print("Data:");
print(data);
print("\n");

# Create the visible values in array
V = data['Visible'].values

# Initialize distribution (pi dist) for the A and B states
init_dist = np.array([0.5, 0.5])

# ------------------------
# Forward Algorithm
# ------------------------
alpha = fwd_algo(A, B, init_dist, V);
print("Final HMM FWD State Probabilities:");
print("alpha.shape = " + str(alpha.shape));
print("\n");
#print(alpha);

# ------------------------
# Backward Algorithm
# ------------------------
beta = bwd_algo(A, B, V);
print("Final HMM BWD State Probabilities:");
print("beta.shape = " + str(beta.shape));
print("\n");

# --------------------------------------------------
# Baum-Welch Algorithm - Fwd/Bwd Latent Algorithm
# --------------------------------------------------
# Transition Probabilities
A = np.ones((2, 2))
A = A / np.sum(A, axis=1)
 
# Emission Probabilities
B = np.array(((1, 3, 5), (2, 4, 6)))
B = B / np.sum(B, axis=1).reshape((-1, 1))

myAB = baum_welch(V, A, B, init_dist, n_iter=100);
print("Baum Welch Algo:");
print("myAB:");print(myAB);

# ----------------------------------------------------
# Viterbi Backtrack Algorithm - most probable states
# ----------------------------------------------------
predictedStates = viterbi(V, A, B, init_dist);
print("Final predicted states:");
print(predictedStates[200:210]);
print("\n");