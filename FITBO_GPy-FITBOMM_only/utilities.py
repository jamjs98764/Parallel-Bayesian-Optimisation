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
