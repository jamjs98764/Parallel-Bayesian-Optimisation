#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 13:58:04 2019

@author: jian
"""

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import numpy as np
from tensorflow import set_random_seed
import cifar_utils
import pickle
import GPyOpt

from class_FITBOMM import Bayes_opt
from class_FITBOMM import Bayes_opt_batch

np.set_printoptions(suppress=True)

a = np.load("allseed_normalised_initx.npy")
b = np.load("allseed_normalised_inity.npy")
print(a,b)
x_init_dict = {}
y_init_dict = {}
x_init_dict[0] = a[0:3]
x_init_dict[1] = a[3:6]
x_init_dict[2] = a[6:9]
y_init_dict[0] = b[0:3]
y_init_dict[1] = b[3:6]
y_init_dict[2] = b[6:9]

np.random.seed(10)
set_random_seed(10)

import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

seed_size = 3
batch_size = 2
if batch_size == 1:
  batch = False
else: batch = True
acq_func = "EI"
eval_type = "local_penalization"
iterations = 40


# Values for marginalisation of GP hyperparameters
n_samples = 150
n_burning = 100
gp_model = "GP"
num_cores = 1
initialsamplesize = 3

X_record = {}
y_opt_record = {}
X_hist_record = {}

domain = [{'name': 'x1', 'type': 'continuous', 'domain': (0., 1.)},
 {'name': 'x2', 'type': 'continuous', 'domain': (0., 1.)},
 {'name': 'x3', 'type': 'continuous', 'domain': (0., 1.)},
 {'name': 'x4', 'type': 'continuous', 'domain': (0., 1.)},
 {'name': 'x5', 'type': 'continuous', 'domain': (0., 1.)},
 {'name': 'x6', 'type': 'continuous', 'domain': (0., 1.)}]

for seed_i in range(seed_size):
    print("Currently on seed: ", seed_i)
    np.random.seed(seed_i)
    seed_i += 1

    # Objective
    obj_func = cifar_utils.cifar_cnn_gpyopt
    obj_func_noise = cifar_utils.cifar_cnn_gpyopt

    # Generating initialising points
    x_ob = x_init_dict[seed_i]
    y_ob = y_init_dict[seed_i]

    # Iter= 0 value
    arg_opt = np.argmin(y_ob)
    x_opt_init = x_ob[arg_opt]
    y_opt_init = y_ob[arg_opt]

    if batch == True:
        # batch
        BO = GPyOpt.methods.BayesianOptimization(f = obj_func_noise,
                                                domain = domain,
                                                acquisition_type = acq_func,
                                                evaluator_type = eval_type,
                                                model_type=gp_model,
                                                normalize_Y = True,
                                                # NEW CHANGES HERE
                                                X = x_ob,
                                                Y = y_ob,
                                                # END
                                                batch_size = batch_size,
                                                num_cores = num_cores,
                                                acquisition_jitter = 0,
                                                n_burning = n_burning,
                                                n_samples = n_samples,
                                                verbosity = False,
                                                eps = 1e-10)
        BO.run_optimization(max_iter = int(iterations / batch_size))
    else:
        # sequential
        BO = GPyOpt.methods.BayesianOptimization(f = obj_func_noise,
                                                domain = domain,
                                                acquisition_type = acq_func,
                                                model_type=gp_model,
                                                normalize_Y = True,
                                                X = x_ob,
                                                Y = y_ob,
                                                batch_size = batch_size,
                                                num_cores = num_cores,
                                                acquisition_jitter = 0,
                                                n_burning = n_burning,
                                                n_samples = n_samples,
                                                verbosity = False,
                                                eps = 1e-10)

        BO.run_optimization(max_iter = int(iterations))

    # Per seed
    eval_record = BO.get_evaluations()[0]
    X_opt = BO.return_minimiser() # (rows = iterations, columns = X_dimensions)
    num_iter = X_opt.shape[0]
    Y_opt = np.zeros((num_iter, 1)) # cols = output dimension
    for i in range(num_iter):
        Y_opt[i] = obj_func(X_opt[i])

    X_record[seed_i] = np.vstack((x_opt_init,X_opt[initialsamplesize:])) # Initial samples dont count (keep zero as first point)
    y_opt_record[seed_i] = np.vstack((y_opt_init,Y_opt[initialsamplesize:]))
    X_hist_record[seed_i] = eval_record[initialsamplesize:]
