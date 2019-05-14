#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 15:35:20 2019

@author: jian
"""

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
import sys
sys.path.insert(0,'..')
import GPyOpt_mod

from class_FITBOMM import Bayes_opt
from class_FITBOMM import Bayes_opt_batch

np.set_printoptions(suppress=True)

initial_num = 3
a = np.load("allseed_normalised_initx.npy")
b = np.load("allseed_normalised_inity.npy")
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

seed_size = 1

total_evals = 80

# For FITBO
num_continuous_dim = 6
num_discrete_dim = 0
num_categorical_dim = 0

input_dim = num_continuous_dim + num_discrete_dim + num_categorical_dim
input_type = [False, False, False, False, False, False] # True if domain is discrete

"""
batch_size = x[0]
fc_units = x[1]
l1_dropout = x[2]
l2_dropout = x[3]
l3_dropout = x[4]
rms_l_rate = x[5]
"""

discrete_bounds = [(16,256), (64, 1024)]
fitbo_lb = []
fitbo_ub = []
continuous_bounds = [(0.0,1.0),
                     (0.0,1.0),
                     (0.0,1.0),
                     (0.0,1.0),
                     (0.0,1.0),
                     (0.0, 1.0),]
for i in continuous_bounds:
    fitbo_lb.append(i[0])
    fitbo_ub.append(i[1])

categorical_choice = []

params_simple = {"batch_size": 32,
                 "fc_units": 512,
          "l1_dropout": 0.25,
          "l2_dropout": 0.25,
          "l3_dropout": 0.5,
          "rms_l_rate": 0.0001,
          }

batch_list = [4]
acq_func_list = ["LCB"]
eval_type = "local_penalization"
seed_size = 3
iterations = 80
n_samples = 150
n_burning = 100
gp_model = "GP"
num_cores = 1
initialsamplesize = 3
domain = [{'name': 'x1', 'type': 'continuous', 'domain': (0., 1.)},
 {'name': 'x2', 'type': 'continuous', 'domain': (0., 1.)},
 {'name': 'x3', 'type': 'continuous', 'domain': (0., 1.)},
 {'name': 'x4', 'type': 'continuous', 'domain': (0., 1.)},
 {'name': 'x5', 'type': 'continuous', 'domain': (0., 1.)},
 {'name': 'x6', 'type': 'continuous', 'domain': (0., 1.)}]

for acq_func in acq_func_list:
    for batch_size in batch_list:
        print("\n Currently running exp: ", acq_func)
        print("\n Currently running batch_size: ", str(batch_size))
        if batch_size == 1:
          batch = False
        else: batch = True

        X_record = {}
        y_opt_record = {}
        X_hist_record = {}

        for seed_i in range(seed_size):
            print("\n Currently on seed: ", seed_i)
            np.random.seed(seed_i)
            set_random_seed(seed_i)

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
                BO = GPyOpt_mod.methods.BayesianOptimization(f = obj_func_noise,
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
                BO = GPyOpt_mod.methods.BayesianOptimization(f = obj_func_noise,
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

        file_name = str(batch_size) + "_batch," + acq_func + ",local_penalization"

        with open(file_name + ",X_record.pickle", 'wb') as f:
            pickle.dump(X_record, f)

        with open(file_name + ",y_opt_record.pickle", 'wb') as f:
            pickle.dump(y_opt_record, f)

        with open(file_name + ",X_hist_record.pickle", 'wb') as f:
            pickle.dump(X_hist_record, f)

