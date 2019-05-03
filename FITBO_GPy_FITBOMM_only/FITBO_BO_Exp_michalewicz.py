#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 12:55:41 2019

@author: jian
"""


"""
Created on Fri Nov 10 13:45:16 2017

@author: robin
"""

import numpy as np
import os
from Test_Funcs import michalewicz_fitbo
from class_FITBOMM import Bayes_opt
from class_FITBOMM import Bayes_opt_batch

##### Initializing experiment parameters

seed_size = 50
num_iters = 80
iter_80 = True

v2_seed_start = 50
v2_seed_size = 50

def BO_test(test_func, BO_method = 'FITBOMM', burnin = 100, sample_size = 50, resample_interval = 1, \
            seed_size = seed_size, num_iterations = num_iters, batch = False, batch_size = 2, heuristic = "kb",
            MLE = False):

    # BO_method is either FITBOMM (moment matching) or FITBO (quadrature)
    # Sample size = MC sample size
    # Seed size = run BO experiment for x times to get spread of results
    # num_iterations = number of sample points to take
    # batch = Boolean for whether sequential or batch mode
    # batch_size = ditto. For "batch" mode, num_iterations = (num_batch / batch_size) for equal comparison

    var_noise = 1.0e-3

    # speficy test func
    if test_func == 'branin':
        obj_func = branin
        d = 2
        initialsamplesize = 3
        true_min = np.array([-14.96021125])
        true_location = np.array([[0.1239, 0.8183],[0.5428, 0.1517],[0.9617, 0.1650]])

    elif test_func == 'egg':
        obj_func = egg
        d = 2
        initialsamplesize = 3
        true_min = np.array([-9.596407])
        true_location = np.array([[1.0, 0.7895]])
    elif test_func == 'hartmann':
        obj_func = hartmann
        d = 6
        initialsamplesize = 9
        true_min = np.array([-18.22368011])
        true_location = np.array([[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]])

    elif test_func == 'mich':
        obj_func = michalewicz_fitbo
        d = 10
        initialsamplesize = 25
        true_min = np.array([-9.6601517])

    else:
        print("Function does not exist in repository")
        return 0

    sigma0 = np.sqrt(var_noise)

    results_IR = np.zeros(shape=(seed_size, num_iterations + 1)) # Immediate regret

    # Creating directory to save
    if batch == False:
        dir_name = 'Exp_Data/' + test_func + ',' + str(seed_size) + '_seed,sequential/'
    else:
        dir_name = 'Exp_Data/' + test_func + ',' + str(seed_size) + '_seed,' + str(batch_size) + '_batch_size/'
    try:
        os.mkdir(dir_name)
    except FileExistsError:
        pass

    if batch == False: # Sequential
        results_IR = np.zeros(shape=(seed_size, num_iterations + 1)) # Immediate regret
        results_L2 = np.zeros(shape=(seed_size, num_iterations + 1)) # L2 norm of x
        X_opt_record = np.zeros(shape=(seed_size, num_iterations + 1, d))
        Y_opt_record = np.zeros(shape=(seed_size, num_iterations + 1))

        for j in range(seed_size):
            seed = j
            np.random.seed(seed)
            x_ob = np.random.uniform(0., 1., (initialsamplesize, d))
            y_ob = obj_func(x_ob) + sigma0 * np.random.randn(initialsamplesize, 1)

            if MLE == False:
                bayes_opt = Bayes_opt(obj_func, np.zeros(d), np.ones(d)*np.pi, var_noise) # Changed upper bound to pi

            bayes_opt.initialise(x_ob, y_ob)
            X_optimum, Y_optimum = bayes_opt.iteration_step(iterations=num_iterations, mc_burn=burnin, \
                                                            mc_samples=sample_size, bo_method=BO_method, \
                                                            seed=seed, resample_interval= resample_interval, \
                                                            dir_name = dir_name)
            results_IR[j, :] = np.abs(Y_optimum - true_min).ravel()

            IR_file_name = dir_name + 'A_results_IR,sequential'

            X_opt_file_name = dir_name + 'A_results_X-opt,sequential'
            Y_opt_file_name = dir_name + 'A_results_Y-opt,sequential'

            X_opt_record[j] = X_optimum
            Y_opt_record[j] = Y_optimum.flatten()

            np.save(IR_file_name, results_IR)
            np.save(X_opt_file_name, X_opt_record)
            np.save(Y_opt_file_name, Y_opt_record)

    if batch == True:
        num_batches = int(num_iterations / batch_size)
        results_IR = np.zeros(shape=(seed_size, num_batches + 1)) # Immediate regret
        results_L2 = np.zeros(shape=(seed_size, num_batches + 1)) # L2 norm of x

        for j in range(seed_size):
            seed = j
            np.random.seed(seed)
            x_ob = np.random.uniform(0., 1., (initialsamplesize, d))
            y_ob = obj_func(x_ob) + sigma0 * np.random.randn(initialsamplesize, 1)

            if MLE == False:
                bayes_opt = Bayes_opt_batch(obj_func, np.zeros(d), np.ones(d)*np.pi, var_noise)  # Changed upper bound to pi

            bayes_opt.initialise(x_ob, y_ob)
            X_optimum, Y_optimum = bayes_opt.iteration_step_batch(num_batches=num_batches, mc_burn=burnin, mc_samples=sample_size, \
                                                                              bo_method=BO_method, seed=seed, resample_interval= resample_interval, \
                                                                              batch_size = batch_size, heuristic = heuristic,
                                                                              dir_name = dir_name)

            results_IR[j, :] = np.abs(Y_optimum - true_min).ravel()

            IR_file_name = dir_name + 'A_results_IR,' + heuristic + '_heuristic'

            X_opt_file_name = dir_name + 'A_results_X-opt,' + heuristic + '_heuristic'
            Y_opt_file_name = dir_name + 'A_results_Y-opt,' + heuristic + '_heuristic'

            np.save(IR_file_name, results_IR)
            np.save(X_opt_file_name, X_optimum)
            np.save(Y_opt_file_name, Y_optimum)


#####
# Running tests
#####
def test_sequential(test_func):
    ## Single test sequential
    BO_test(test_func = test_func)
    return None

def test_all(test_func, current_batch_size):
    ## Single test batch

    BO_test(test_func = test_func, BO_method = 'FITBOMM', batch = True, batch_size = current_batch_size, heuristic = 'kb')
    BO_test(test_func = test_func, BO_method = 'FITBOMM', batch = True, batch_size = current_batch_size, heuristic = 'cl-min')
    """
    BO_test(test_func = test_func, BO_method = 'FITBOMM', batch = True, batch_size = current_batch_size, heuristic = 'cl-mean')
    BO_test(test_func = test_func, BO_method = 'FITBOMM', batch = True, batch_size = current_batch_size, heuristic = 'cl-max')
    BO_test(test_func = test_func, BO_method = 'FITBOMM', batch = True, batch_size = current_batch_size, heuristic = 'random')
    """
    return None

# Sequential

"""
test_funcs = ["mich"]

for func in test_funcs:
    test_sequential(func)
"""
batch_sizes = [8]

test_funcs = ["mich"]
for batch_size in batch_sizes:
    for test_func in test_funcs:
        test_all(test_func, batch_size)

