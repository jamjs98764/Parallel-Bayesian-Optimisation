#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:51:37 2019

@author: jian
"""

import GPy
import numpy as np
X = np.random.uniform(-3.,3.,(20,1))
Y = np.sin(X) + np.random.randn(20,1)*0.05
kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
m = GPy.models.GPRegression(X,Y,kernel)

Xgrid = np.linspace(-3,3,num=1000)
Xgrid = np.atleast_2d(Xgrid)
Xgrid = np.transpose(Xgrid)
# https://gpy.readthedocs.io/en/deploy/GPy.models.html?highlight=posterior_samples_f

f = m.posterior_samples_f(Xgrid, size = 50)

func1 = f[:,0,0]
func2 = f[:,0,1]

import matplotlib.pyplot as plt
plt.plot(Xgrid, func2)

from scipy.optimize import minimize

def _gloabl_minimser(func):
    ntest = 2000;
    Xgrid = np.random.uniform(-3., 3.0, (ntest, 1))
    Func_value_test = func(Xgrid)
    idx_test = np.argmin(Func_value_test)
    X_start = Xgrid[idx_test, :]
    # bnds = ((0.0, 1.0), (0.0, 1.0))
    bnds = tuple(-3.,3.)

    res = minimize(func, X_start, method='L-BFGS-B', jac=False, bounds=bnds)
    x_opt = res.x[None, :]
    return x_opt

def pos_samples_wrapper(X, model = m):

