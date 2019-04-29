# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 16:33:45 2018

@author: jianhong
"""
import numpy as np
import GPy

def _fit_GP(self):
    '''collect GPs defined using observed g values and all hyperparameter samples'''
    self.GP = [] # Contains a collection of GPs based on hyperparameter sampling
    self.kernel = [] # Contains a collection of kernels based on hyperparameter sampling
    lscale_param = self.params[:, 0:self.X_dim]
    var_param = self.params[:, self.X_dim]
    noise_param = self.params[:, self.X_dim + 1]
    eta = np.min(self.Y) - self.params[:, self.X_dim + 2] # There are 

    for i in range(len(self.params)): # iterate across each MC sample of hyperparaters (Nsamples = 50)
        diff = self.Y - eta[i]
        diff.clip(min=0) # Change negative values to zero
        self.G = np.sqrt(2.0 * diff)
        self.kernel.append(GPy.kern.RBF(input_dim=self.X_dim, ARD=True, variance=var_param[i], lengthscale=lscale_param[i, :]))
        self.GP.append(GPy.models.GPRegression(self.X, self.G, self.kernel[i], noise_var=(noise_param[i])))


def _fit_GP_normal(self):
    '''collect GPs defined using observed y values and all hyperparameter samples'''
    self.GP_normal = []
    noise_param = self.params[:, self.X_dim + 1]
    for i in range(len(self.params)):
        self.GP_normal.append(GPy.models.GPRegression(self.X, self.Y, self.kernel[i], noise_var=noise_param[i]))
