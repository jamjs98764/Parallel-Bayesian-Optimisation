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

seed_size = 3
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

batch_size_list = [8]
heuristic_list = ['cl-min', 'kb']

BO_method = 'FITBOMM'
burnin = 100
sample_size = 50
resample_interval = 1

save_dir = 'fyp_bo_jian/FITBO_GPy_FITBOMM_only/Exp_Data/cifar10'
dir_name = save_dir + "/FITBO/80_iter," + str(batch_size) + "_batch"

for batch_size in batch_size_list:
    for heuristic in heuristic_list:
        if batch_size == 1: # Sequential
            heuristic = "sequential"
            results_X_hist = np.zeros(shape=(seed_size, total_evals + initial_num, input_dim))
            results_X_optimum = np.zeros(shape=(seed_size, total_evals + 1, input_dim))
            results_Y_hist = np.zeros(shape=(seed_size, total_evals + initial_num))
            results_Y_optimum = np.zeros(shape=(seed_size, total_evals + 1))

            for seed in range(seed_size):
                np.random.seed(seed)
                set_random_seed(seed)
                print("Currently on seed: ", seed)
                x_ob = x_init_dict[seed]
                y_ob = y_init_dict[seed]
                
                bayes_opt = Bayes_opt(cifar_utils.cifar_cnn_fitbo, fitbo_lb, fitbo_ub, var_noise = 0, input_type = input_type)
                bayes_opt.initialise(x_ob, y_ob)
                X_optimum, Y_optimum = bayes_opt.iteration_step(iterations=total_evals, mc_burn=burnin, \
                                                                mc_samples=sample_size, bo_method=BO_method, \
                                                                seed=seed, resample_interval= resample_interval, \
                                                                dir_name = dir_name)
                results_X_hist[seed, :] = bayes_opt.X
                results_X_optimum[seed, :] = X_optimum
                results_Y_hist[seed, :] = bayes_opt.Y.flatten()
                results_Y_optimum[seed, :] = Y_optimum.flatten()

                X_file_name = dir_name + "batch_" + str(batch_size) + ",seed_" + str(seed_size) + "," + str(heuristic) + ",X_optimum"
                Y_file_name = dir_name + "batch_" + str(batch_size) + ",seed_" + str(seed_size) + "," + str(heuristic) + ",Y_optimum"
                X_hist_file_name = dir_name + "batch_" + str(batch_size) + ",seed_" + str(seed_size) + "," + str(heuristic) + ",X_hist"
                Y_hist_file_name = dir_name + "batch_" + str(batch_size) + ",seed_" + str(seed_size) + "," + str(heuristic) + ",Y_hist"

                np.save(X_file_name, X_optimum) # results_IR/L2 is np array of shape (num_iterations + 1, seed_size)
                np.save(Y_file_name, Y_optimum)
                np.save(X_hist_file_name, bayes_opt.X)
                np.save(Y_hist_file_name, bayes_opt.Y)

                np.save(X_file_name, results_X_optimum)
                np.save(Y_file_name, results_Y_optimum)
                np.save(X_hist_file_name, results_X_hist)
                np.save(Y_hist_file_name, results_Y_hist)

        else: # Batch
            num_batches = int(total_evals / batch_size)
            results_X_hist = np.zeros(shape=(seed_size, total_evals + initial_num, input_dim))
            results_X_optimum = np.zeros(shape=(seed_size, num_batches + 1, input_dim))
            results_Y_hist = np.zeros(shape=(seed_size, total_evals + initial_num))
            results_Y_optimum = np.zeros(shape=(seed_size, num_batches + 1))

            for seed in range(seed_size):
                print("Currently on seed: ", seed)
                np.random.seed(seed)
                set_random_seed(seed)
                x_ob = x_init_dict[seed]
                y_ob = y_init_dict[seed]

                bayes_opt = Bayes_opt_batch(cifar_utils.cifar_cnn_fitbo, fitbo_lb, fitbo_ub, var_noise = 0, input_type = input_type)
                bayes_opt.initialise(x_ob, y_ob)
                X_optimum, Y_optimum = bayes_opt.iteration_step_batch(num_batches=num_batches, mc_burn=burnin, \
                                                                      mc_samples=sample_size, bo_method=BO_method, seed=seed, \
                                                                      resample_interval= resample_interval, batch_size = batch_size, \
                                                                      heuristic = heuristic, dir_name = dir_name)

                results_X_hist[seed, :] = bayes_opt.X
                results_X_optimum[seed, :] = X_optimum
                results_Y_hist[seed, :] = bayes_opt.Y.flatten()
                results_Y_optimum[seed, :] = Y_optimum.flatten()

            X_file_name = dir_name + "norm_batch_" + str(batch_size) + ",seed_" + str(seed_size) + "," + str(heuristic) + ",X_optimum"
            Y_file_name = dir_name + "norm_batch_" + str(batch_size) + ",seed_" + str(seed_size) + "," + str(heuristic) + ",Y_optimum"
            X_hist_file_name = dir_name + "norm_batch_" + str(batch_size) + ",seed_" + str(seed_size) + "," + str(heuristic) + ",X_hist"
            Y_hist_file_name = dir_name + "norm_batch_" + str(batch_size) + ",seed_" + str(seed_size) + "," + str(heuristic) + ",Y_hist"

            np.save(X_file_name, results_X_optimum)
            np.save(Y_file_name, results_Y_optimum)
            np.save(X_hist_file_name, results_X_hist)
            np.save(Y_hist_file_name, results_Y_hist)
