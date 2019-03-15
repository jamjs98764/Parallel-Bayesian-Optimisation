#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 13:45:16 2017

@author: robin
"""

import numpy as np
import math
import scipy as sp
from scipy import stats
from scipy.stats import norm
import scipy.integrate as integrate

import sobol_seq

'''Utility Functions'''

def gamma_fromEV(x, E, V):
    """
    Creates an instance of a Gamma Prior  by specifying the Expected value(s)
    and Variance(s) of the distribution.  by # Copyright (c) 2012, GPy authors (see AUTHORS.txt).
    Licensed under the BSD 3-clause license (see LICENSE.txt)

    :param E: expected value
    :param V: variance
    """
    alpha = np.square(E) / V
    beta = E / V
    s = 1.0 / beta
    return stats.gamma.pdf(x, a=alpha, scale=s)

def Gaussianprior(lntheta, mean, variance):
    # lntheta = 1 x N , mean = 1 x N ,  variance = N x N
    return sp.stats.multivariate_normal.pdf(lntheta, mean, variance)


def GMMpdf(x, Means, Variances,W):

    gaussianpdf = norm.pdf(x,Means,np.sqrt(Variances))
    p = np.sum(W*gaussianpdf)
    return p

def computeEntropyGMM1Dquad(means, covs, weights, N):

    '''Compute entropy of a 1D Gaussian mixture by numerical integration
    means - array of means of Gaussians in the mixture
    covs - array of covariance matrices of Gaussians in the mixture
    weights - vector of weights of Gaussians in the mixture
    N - (optional) number of st.dev. to include in the integration region (large values can make integration unstable)
    '''
    # set parameters
    if N is None:
        N = 3     # number of st.dev. to include in the integration region

    # set integration boundaries
    std = np.sqrt(covs)
    minX   = np.min(means - N * std)
    maxX   = np.max(means + N * std)

    # compute entropy
    integrant = lambda x: - GMMpdf(x, means, covs, weights) * np.log(GMMpdf(x, means, covs, weights))
    H = integrate.quad(integrant, minX, maxX, epsabs=1e-4)[0]
    # H = integrate.quad(integrant,minX,maxX)

    return H


''' Initial Sequence Generation '''

def sobol_normalized(num_dim, num_init):
    ''' 
    Generate ndarray Sobol sequence 
    Assumes domain: 0 to 1
    Assumes all numerical data

    Used for egg, branin, hartmann

    Documentation: https://github.com/naught101/sobol_seq/tree/6ac1799818a1b9a359a5fc86517173584fe34613
    '''
    return sobol_seq.i4_sobol_generate(num_dim, num_init) # rows = sequence, columns = dimensions/features

def sobol_mixed_unnormalized(num_continuous_dim, num_discrete_dim, num_categorical_dim, 
                            continuous_bounds, discrete_bounds, categorical_choice, 
                            num_init, seed):
    ''' 
    Generates Sobol sequence for continuous input domain, random selection for categorical domain
    Discrete domain uses continuous generation, but rounds to nearest integer

    Assumes domain: any range, given by continuous_bounds

    Returns (continuous_sequence, discrete_sequence, categorical_sequence)
    '''
    np.random.seed(seed)

    ####
    # Continuous sequence
    ####
    continuous_seq = sobol_seq.i4_sobol_generate(num_continuous_dim, num_init)
    continuous_seq = np.transpose(continuous_seq) # we use different convention
    
    # Scaling to bounds
    for i in range(num_continuous_dim):
        continuous_seq[:,i] = continuous_seq[:,i]*(continuous_bounds[i][1] - continuous_bounds[i][0])

    ####
    # Discrete sequence
    ####
    if num_discrete_dim > 0:
        discrete_seq = sobol_seq.i4_sobol_generate(num_discrete_dim, num_init)
        discrete_seq = np.transpose(discrete_seq)

        # Scaling to bounds then rounding
        for i in range(num_discrete_dim):
            discrete_seq[:,i] = np.ceil(discrete_seq[:,i]*(discrete_bounds[i][1] - discrete_bounds[i][0])).astype(int)
    else: 
        discrete_seq = 0

    ####
    # Categorical sequence
    ####
    if num_categorical_dim > 0:
        categorical_seq = np.ones((num_init, num_categorical_dim))

        for j in range(num_categorical_dim):
            for i in range(num_init):
                categorical_seq[i, j] = np.random.randint(categorical_choice[j][0],categorical_choice[j][1])
                # get random integer between lower and upper bound

    else:
        categorical_seq = 0

    return (continuous_seq, discrete_seq, categorical_seq) 

def random_mixed_unnormalized(num_continuous_dim, num_discrete_dim, num_categorical_dim, 
                            continuous_bounds, discrete_bounds, categorical_choice, 
                            num_init, seed):
    ''' 
    Generates Sobol sequence for continuous input domain, random selection for categorical domain
    Discrete domain uses continuous generation, but rounds to nearest integer

    Assumes domain: any range, given by continuous_bounds

    Returns (continuous_sequence, discrete_sequence, categorical_sequence)
    '''
    np.random.seed(seed)

    ####
    # Continuous sequence
    ####
    continuous_seq = np.random.random((num_init, num_continuous_dim))

    # Scaling to bounds
    for i in range(num_continuous_dim):
        continuous_seq[:,i] = continuous_seq[:,i]*(continuous_bounds[i][1] - continuous_bounds[i][0])

    ####
    # Discrete sequence
    ####
    discrete_seq = np.zeros((num_init, num_discrete_dim))

    if num_discrete_dim > 0:
        for i in range(num_discrete_dim):
            dim_seq = np.random.randint(discrete_bounds[i][0], discrete_bounds[i][1], (num_init, 1))
            discrete_seq[:, i] = dim_seq.flatten()

    else: 
        discrete_seq = 0

    ####
    # Categorical sequence
    ####
    if num_categorical_dim > 0:
        categorical_seq = np.ones((num_init, num_categorical_dim))

        for j in range(num_categorical_dim):
            for i in range(num_init):
                categorical_seq[i, j] = np.random.randint(categorical_choice[j][0],categorical_choice[j][1])
                # get random integer between lower and upper bound

    else:
        categorical_seq = 0

    return (continuous_seq, discrete_seq, categorical_seq) 






