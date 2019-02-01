# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 21:21:33 2019

@author: jianhong
"""

import GPy
import GPyOpt
import numpy as np
from sklearn import svm
from numpy.random import seed
seed(12345)

nfold = 3
def fit_svr_val(x):
    x = np.atleast_2d(np.exp(x))
    fs = np.zeros((x.shape[0],1))
    for i in range(x.shape[0]):
        fs[i] = 0
        for n in range(nfold):
            idx = np.array(range(X_train.shape[0]))
            idx_valid = np.logical_and(idx>=X_train.shape[0]/nfold*n, idx<X_train.shape[0]/nfold*(n+1))
            idx_train = np.logical_not(idx_valid)
            svr = svm.SVR(C=x[i,0], epsilon=x[i,1],gamma=x[i,2])
            svr.fit(X_train[idx_train],Y_train[idx_train])
            fs[i] += np.sqrt(np.square(svr.predict(X_train[idx_valid])-Y_train[idx_valid]).mean())
        fs[i] *= 1./nfold
    return fs

domain       =[{'name': 'C',      'type': 'continuous', 'domain': (0.,7.)},
               {'name': 'epsilon','type': 'continuous', 'domain': (-12.,-2.)},
               {'name': 'gamma',  'type': 'continuous', 'domain': (-12.,-2.)}]

opt = GPyOpt.methods.BayesianOptimization(f = fit_svr_val,            # function to optimize       
                                          domain = domain,         # box-constraints of the problem
                                          acquisition_type ='EI',       # LCB acquisition
                                          acquisition_weight = 0.1)   # Exploration exploitation
                                          
# it may take a few seconds
opt.run_optimization(max_iter=20, verbosity=True)
opt.plot_convergence()