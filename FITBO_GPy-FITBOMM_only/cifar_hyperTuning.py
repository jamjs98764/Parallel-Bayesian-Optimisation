#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:36:29 2019

@author: jian
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:06:52 2019

@author: jian
"""


'''
#Train a simple deep CNN on the CIFAR10 small images dataset.
It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

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

from class_FITBOMM import Bayes_opt
from class_FITBOMM import Bayes_opt_batch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
"""
seed
_BATCH_SIZE (16 to 128)
dropout_rate (0 to 1)
fully_connected_units
"""
#####
# Experiment parameters
#####

# REMINDER!!!!!!!!
# Set epoch parameters too

seed_size = 1
total_evals = 4

# For FITBO
num_continuous_dim = 4
num_discrete_dim = 2
num_categorical_dim = 0

input_dim = num_continuous_dim + num_discrete_dim + num_categorical_dim
input_type = [True, True, False, False, False, False] # True if domain is discrete

"""
batch_size = x[0]
fc_units = x[1]
l1_dropout = x[2]
l2_dropout = x[3]
l3_dropout = x[4]
rms_l_rate = x[5]
"""

discrete_bounds = [(4,256), (32, 1024)]
fitbo_lb = [discrete_bounds[0][0], discrete_bounds[1][0]]
fitbo_ub = [discrete_bounds[0][1], discrete_bounds[1][0]]
continuous_bounds = [(0.0,1.0),
                     (0.0,1.0),
                     (0.0,1.0),
                     (0.00001, 0.1),]
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

#####
# Initial Points
#####
np.random.seed(1)
set_random_seed(1)
initial_num = 3
x_ob1 = np.array([32, 256, 0.2, 0.2, 0.7, 0.0001])
x_ob2 = np.array([64, 128,  0.7, 0.3, 0.5, 0.01])
x_ob3 = np.array([32, 512, 0.1, 0.1, 0.2, 0.001])

x_ob = np.vstack((x_ob1, x_ob2, x_ob3))

y_ob = cifar_utils.cifar_cnn_fitbo(x_ob)
np.save("cifar-y_ob.npy", y_ob)
"""
y_ob = np.load("cifar-y_ob.npy")
"""

def cifar_fitbo_wrapper(batch_size, heuristic = "cl-min"):

    BO_method = 'FITBOMM'
    burnin = 100
    sample_size = 50
    resample_interval = 1

    save_dir = os.path.join(os.getcwd(), 'Exp_Data/cifar10')
    dir_name = save_dir + "/FITBO/" + str(batch_size) + "_batch"

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

        X_file_name = dir_name + "batch_" + str(batch_size) + ",seed_" + str(seed_size) + "," + str(heuristic) + ",X_optimum"
        Y_file_name = dir_name + "batch_" + str(batch_size) + ",seed_" + str(seed_size) + "," + str(heuristic) + ",Y_optimum"
        X_hist_file_name = dir_name + "batch_" + str(batch_size) + ",seed_" + str(seed_size) + "," + str(heuristic) + ",X_hist"
        Y_hist_file_name = dir_name + "batch_" + str(batch_size) + ",seed_" + str(seed_size) + "," + str(heuristic) + ",Y_hist"

        np.save(X_file_name, results_X_optimum)
        np.save(Y_file_name, results_Y_optimum)
        np.save(X_hist_file_name, results_X_hist)
        np.save(Y_hist_file_name, results_Y_hist)

    return None

batch_list = [2]
heuristic_list = ['cl-min']

#cifar_fitbo_wrapper(batch_size = 1, heuristic = "kb")

for batch in batch_list:
    for heur in heuristic_list:
        cifar_fitbo_wrapper(batch_size = batch, heuristic = heur)

