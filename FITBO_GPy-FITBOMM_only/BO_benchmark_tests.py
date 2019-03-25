# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 15:18:49 2019

@author: jianhong

Implementing GPyOpt to serve as benchmark

"""
import GPyOpt
import numpy as np
import scipy as sp
import Test_Funcs
from functools import partial
from numpy.random import seed
import pickle
import os
from plotting_utilities import *

var_noise = 1.0e-3 # y_next = self.func(x_next) + np.random.normal(0, self.var_noise, len(x_next))
seed_size = 50

def wrapper_GPyOpt(test_func, acq_func = "EI", eval_type = "random", \
    seed_size = seed_size, iterations = 40, batch_size = 2):
    """
    Wrapper function which implements GPyOpt BO
    Returns all query points

    X: 2d numpy array for initial inputs, one per row
    acq_func: "EI" / "EI_MCMC" / "MPI_MCMC" /  "LCB" / "LCB_MCMC"
    evaluator_type: sequential / random  (1st random in Jian's deifnition) / local_penalization / thompson_sampling

    Documentation:
    https://gpyopt.readthedocs.io/en/latest/GPyOpt.core.html#GPyOpt.core.bo.BO
    https://gpyopt.readthedocs.io/en/latest/GPyOpt.methods.html#
    """
    # Noise
    var_noise = 1.0e-3  # Same noise setting as FITBO tests
    sigma0 = np.sqrt(var_noise)

    # Values for marginalisation of GP hyperparameters
    n_samples = 150
    n_burning = 100

    # Specifying GP model type
    # if MCMC acq func used, require GP_MCMC
    if acq_func[-4:] == "MCMC":
        gp_model = "GP_MCMC"
    else:
        gp_model = "GP"


    num_cores = 1
    if batch_size > 1:
        batch = True
    else:
        batch = False

    if test_func == 'branin':
        obj_func = Test_Funcs.branin_gpyopt
        d = 2
        domain = [{'name': 'x1', 'type': 'continuous', 'domain': (0., 1.)},
                   {'name': 'x2', 'type': 'continuous', 'domain': (0., 1.)}]
        initialsamplesize = 3

    elif test_func == 'egg':
        obj_func = Test_Funcs.egg_gpyopt
        d = 2
        domain = [{'name': 'x1', 'type': 'continuous', 'domain': (0., 1.)},
           {'name': 'x2', 'type': 'continuous', 'domain': (0., 1.)}]
        initialsamplesize = 3

    elif test_func == 'hartmann':
        obj_func = Test_Funcs.hartmann_gpyopt
        d = 6
        domain = [{'name': 'x1', 'type': 'continuous', 'domain': (0., 1.)},
           {'name': 'x2', 'type': 'continuous', 'domain': (0., 1.)},
           {'name': 'x3', 'type': 'continuous', 'domain': (0., 1.)},
           {'name': 'x4', 'type': 'continuous', 'domain': (0., 1.)},
           {'name': 'x5', 'type': 'continuous', 'domain': (0., 1.)},
           {'name': 'x6', 'type': 'continuous', 'domain': (0., 1.)}]
        initialsamplesize = 9

    else:
        print("Function does not exist in repository")
        return 0

    X_record = {}
    y_opt_record = {}
    X_hist_record = {}

    for seed_i in range(seed_size):
        print("Currently on seed: ", seed_i)
        np.random.seed(seed_i)

        # Noisy function
        def obj_func_noise(x):
            noisy_eval = obj_func(x) + sigma0 * np.random.randn()
            return noisy_eval

        # Generating initialising points
        x_ob = np.random.uniform(0., 1., (initialsamplesize, d))

        # NOTE: gpyopt function is applied row by row

        y_ob = np.zeros((initialsamplesize, 1))
        for i in range(initialsamplesize):
            y_ob[i] = obj_func_noise(x_ob[i,:]) # initial samples have noise too

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
                                                    normalize_Y = False,
                                                    # NEW CHANGES HERE
                                                    initial_design_numdata = initialsamplesize,
                                                    initial_design_type = 'random',
                                                    # END
                                                    batch_size = batch_size,
                                                    num_cores = num_cores,
                                                    acquisition_jitter = 0,
                                                    n_burning = n_burning,
                                                    n_samples = n_samples)
            BO.run_optimization(max_iter = int(iterations / batch_size))
        else:
            # sequential
            BO = GPyOpt.methods.BayesianOptimization(f = obj_func_noise,
                                                    domain = domain,
                                                    acquisition_type = acq_func,
                                                    model_type=gp_model,
                                                    normalize_Y = False,
                                                    initial_design_numdata = initialsamplesize,
                                                    initial_design_type = 'random',
                                                    batch_size = batch_size,
                                                    num_cores = num_cores,
                                                    acquisition_jitter = 0,
                                                    n_burning = n_burning,
                                                    n_samples = n_samples)
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

        """
        # Changes to save acq func grid
        dir_name = 'Exp_Data/gpyopt/' + test_func + ',' + str(seed_size) + '_seed,' + str(batch_size) + '_batch/'
        file_name = dir_name + str(acq_func) + ',' + str(eval_type) + ',acq_func.png'
        BO.plot_acquisition()
        """
    return X_record, y_opt_record, X_hist_record

def min_y_hist(y_hist):
    """
    Takes y_hist and returns min_y_hist (same size), which is the minimum y_hist at each iteration
    """
    import copy

    y_copy = copy.deepcopy(y_hist)
    seed_size, iters = y_copy.shape

    for i in range(seed_size):
        current_min = y_copy[i,0]

        for j in range(iters):
            if y_copy[i,j] < current_min:
                current_min = y_copy[i,j]
            y_copy[i,j] = current_min
    return y_copy

def saving_data(X_record, min_y_record, X_hist_record):
    """
    For saving data

    X_record = x_opt
    min_y_record = y_opt
    """
    dir_name = 'Exp_Data/gpyopt/' + test_func + ',' + str(seed_size) + '_seed,' + str(batch_size) + '_batch/'
    file_name = dir_name + str(acq_func) + ',' + str(eval_type) + ',results_vars.pickle'

    try: # creates new folder
        os.mkdir(dir_name)
    except FileExistsError:
        pass

    pickle_dict = {
        "X": X_record,
        "min_y": min_y_record,
        "eval_record": X_hist_record
        }

    with open(file_name, 'wb') as f:
        pickle.dump(pickle_dict, f)

    return 0

#acq_funcs =  ["EI", "EI_MCMC", "MPI_MCMC",  "LCB", "LCB_MCMC"]
#evaluator_types = ["sequential", "random", "local_penalization", "thompson_sampling"]

batch_sizes = [2, 4, 8]
test_funcs = ["branin", "egg", "hartmann"]
#test_funcs = ["hartmann"]
acq_funcs =  ["EI"]
evaluator_types = ["local_penalization"] # does not matter for batch size = 1

for test_func in test_funcs:
    for batch_size in batch_sizes:
        for acq_func in acq_funcs:
            for eval_type in evaluator_types:
                print(test_func, batch_size, acq_func, eval_type)
                X_record, min_y_record, eval_record = wrapper_GPyOpt(test_func, acq_func = acq_func, eval_type = eval_type, \
                                                  batch_size = batch_size)
                saving_data(X_record, min_y_record, eval_record)
