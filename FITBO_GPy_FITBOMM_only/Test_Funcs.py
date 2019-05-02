#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 13:45:16 2017

@author: robin
"""

import numpy as np
import math

'''
Benchmark Test Functions

Input: vector, x
Output: Function value (noise-less)


Note: difference between normal-func and gpyopt-func
- for normal-func, X is size (num_iter, input_dim)
- for gpyopt-func, X is size (1, input_dim)
'''


def func1D(x):
    '''1D gaussian mixture'''
    var_1 = 0.01
    var_2 = 0.03
    var_3 = 0.05
    var_4 = 0.03

    mean_1 = 0.3
    mean_2 = 0.6
    mean_3 = 0
    mean_4 = 1

    f = 1.5 - (((1 / np.sqrt(2 * np.pi * var_1)) * np.exp(-pow(x - mean_1, 2) / var_1)) \
               + ((1 / np.sqrt(2 * np.pi * var_2)) * np.exp(-pow(x - mean_2, 2) / var_2)) \
               + ((1 / np.sqrt(2 * np.pi * var_3)) * np.exp(-pow(x - mean_3, 2) / var_2))
               + ((1 / np.sqrt(2 * np.pi * var_4)) * np.exp(-pow(x - mean_4, 2) / var_2)))
    return f[:, None]

def branin(X):
    '''2D branin
    f_min =-14.96021125
    x_min = [0.1239, 0.8183]; [0.5428, 0.1517]; [0.9617, 0.1650]
    '''
    x1 = X[:, 0] * 15 - 5
    x2 = X[:, 1] * 15
    y_unscaled = (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 * x1 / np.pi - 6) ** 2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    y = y_unscaled / 10 - 15
    return y[:, None]

def branin_gpyopt(X):
    '''2D branin in gpyopt format
    f_min =-14.96021125
    x_min = [0.1239, 0.8183]; [0.5428, 0.1517]; [0.9617, 0.1650]'''
    X = np.atleast_2d(X)
    x1 = X[0,0]
    x2 = X[0,1]
    x1 = x1 * 15 - 5
    x2 = x2 * 15
    y_unscaled = (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 * x1 / np.pi - 6) ** 2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    y = y_unscaled / 10 - 15
    return y

def branin_gpyopt_v2(X):
    '''2D branin in gpyopt format
    f_min =-14.96021125
    x_min = [0.1239, 0.8183]; [0.5428, 0.1517]; [0.9617, 0.1650]'''
    X = np.atleast_2d(X)
    x1 = X[0,0]
    x2 = X[0,1]
    x1 = x1 * 15 - 5
    x2 = x2 * 15
    y_unscaled = (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 * x1 / np.pi - 6) ** 2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    y = y_unscaled / 10 - 15
    return -y

def egg(x):
    '''2D eggholder
    f_min = -9.596407
    x_min = [1.0, 0.7895]'''
    x0 = x[:, 0] * 512
    x1 = x[:, 1] * 512
    term1 = -(x1 + 47) * np.sin(np.sqrt(np.abs(x1 + x0 / 2 + 47)))
    term2 = -x0 * np.sin(np.sqrt(np.abs(x0 - (x1 + 47))))
    y = (term1 + term2) / 100
    return y[:, None]

def egg_gpyopt(x):
    '''2D eggholder
    f_min = -9.596407
    x_min = [1.0, 0.7895]'''
    X = np.atleast_2d(x)
    x0 = X[0,0]
    x1 = X[0,1]

    x0 = x0 * 512
    x1 = x1 * 512
    term1 = -(x1 + 47) * np.sin(np.sqrt(np.abs(x1 + x0 / 2 + 47)))
    term2 = -x0 * np.sin(np.sqrt(np.abs(x0 - (x1 + 47))))
    y = (term1 + term2) / 100
    return y


def hartmann(x):
    '''6D hartmann
    f_min = -18.22368011
    x_min = [0.20169,0.150011,0.476874,0.275332,0.311652,0.6573]
    '''
    a = np.array([[10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
                  [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
                  [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
                  [17.0, 8.0, 0.05, 10.0, 0.1, 14.0]])
    c = np.array([1.0, 1.2, 3.0, 3.2])
    p = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                  [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                  [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
                  [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])
    y = 0.0
    for i in range(4):
        sum = 0.0
        for j in range(6):
            sum = sum - a[i][j]*(x[:,j]-p[i][j])**2
        y = y - c[i]*np.exp(sum)
    y_biased = 10*(y+1.5)
    return y_biased[:, None]

def hartmann_gpyopt(x):
    '''6D hartmann
    f_min = -18.22368011
    x_min = [0.20169,0.150011,0.476874,0.275332,0.311652,0.6573]
    '''
    x = x.flatten()
    a = np.array([[10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
                  [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
                  [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
                  [17.0, 8.0, 0.05, 10.0, 0.1, 14.0]])
    c = np.array([1.0, 1.2, 3.0, 3.2])
    p = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                  [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                  [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
                  [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])
    y = 0.0
    for i in range(4):
        sum = 0.0
        for j in range(6):
            sum = sum - a[i][j]*(x[j]-p[i][j])**2
        y = y - c[i]*np.exp(sum)
    y_biased = 10*(y+1.5)
    return y_biased

def branin_pso(X):
    '''2D branin
    f_min =-14.96021125
    x_min = [0.1239, 0.8183]; [0.5428, 0.1517]; [0.9617, 0.1650]
    '''
    x1 = X[:, 0] * 15 - 5
    x2 = X[:, 1] * 15
    y_unscaled = (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 * x1 / np.pi - 6) ** 2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    y = y_unscaled / 10 - 15
    return y[:, None].flatten()

def egg_pso(x):
    '''2D eggholder
    f_min = -9.596407
    x_min = [1.0, 0.7895]'''
    x0 = x[:, 0] * 512
    x1 = x[:, 1] * 512
    term1 = -(x1 + 47) * np.sin(np.sqrt(np.abs(x1 + x0 / 2 + 47)))
    term2 = -x0 * np.sin(np.sqrt(np.abs(x0 - (x1 + 47))))
    y = (term1 + term2) / 100
    return y[:, None].flatten()

def hartmann_pso(x):
    '''6D hartmann
    f_min = -18.22368011
    x_min = [0.20169,0.150011,0.476874,0.275332,0.311652,0.6573]
    '''
    a = np.array([[10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
                  [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
                  [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
                  [17.0, 8.0, 0.05, 10.0, 0.1, 14.0]])
    c = np.array([1.0, 1.2, 3.0, 3.2])
    p = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                  [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                  [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
                  [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])
    y = 0.0
    for i in range(4):
        sum = 0.0
        for j in range(6):
            sum = sum - a[i][j]*(x[:,j]-p[i][j])**2
        y = y - c[i]*np.exp(sum)
    y_biased = 10*(y+1.5)
    return y_biased[:, None].flatten()


michalewicz_m = 10  # orig 10: ^20 => underflow

def michalewicz_fitbo(X):  # mich.m
    Y = np.zeros((len(X),1))
    for i in range(len(X)):
        x = X[i]
        x = np.asarray_chkfinite(x)
        n = len(x)
        j = np.arange( 1., n+1 )
        y = - sum( np.sin(x) * np.sin( j * x**2 / np.pi ) ** (2 * michalewicz_m) )
        Y[i] = y
    return Y

def michalewicz_gpyopt(x):  # mich.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange( 1., n+1 )
    y = - sum( np.sin(x) * np.sin( j * x**2 / np.pi ) ** (2 * michalewicz_m) )
    return y

