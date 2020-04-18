#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 21:12:28 2020

@author: alejandrogonzales
"""
import numpy as np

def markovprocess(P, sigma, mu, N):
    
    # Create random variable and adjust the vector
    a = np.random.uniform(0,1)
    p = np.cumsum(P, 1)
    zact = int(np.ceil(a*(mu.size-1)))
    z = []
    print("a random = ", a)
    print("p cumsum:");
    print(p);
    print("zact:");
    print(zact)
    print("\n");
    
    for i in range(N):
        b = np.random.uniform(0,1)
        print("b:");print(b);
        zact = np.amin(np.argwhere(p[zact,:] > b))
        print("p[zact,:]");
        print(p[zact,:]);
        print("argwhere p:");
        print(np.argwhere(p[zact,:] > b));
        print("zact:");print(zact);
        z.append(zact)
        #z = [z zact]
        print(z)
        print("\n");
        
    zarr = np.array(z);
    x = np.random.normal(size=zarr.size)*sigma[z]+mu[z]
    return [x, z]


a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(a)
print(a.shape)

# Transition matrix between nodes and connections
P = np.array([[0.8, 0.1, 0.1], [0.2, 0.5, 0.3], [0.3, 0.1, 0.6]])
print(P)
print(P.shape)

# Mean and std deviation vectors 
mu = np.array([1, 2, 3])
sigma = 0.3*np.ones(3)
print("Mean and std dev:")
print(mu)
print(sigma)

# Number of samples for the Markov chain
N = 100

[x, z] = markovprocess(P, sigma, mu, N)
print("Markove Done:");
print("x:");
print(x)
print("z:");
print(z);

