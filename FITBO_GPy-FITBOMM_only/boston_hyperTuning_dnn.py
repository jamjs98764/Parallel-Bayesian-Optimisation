#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 19:38:33 2019

@author: jian
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 16:08:52 2018

@author: jianhong

Performs one-layer MLP Regressor
Tries 3 different packages: scikit, gpyopt, fitbo

Records cross-validation error at each iteration

Tunes:
    i. learning rate
    ii. alpha - l2 penalizer
    iii. number of neurons
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os

from sklearn.datasets import load_boston
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import utilities

total_evals = 40 # on top of initial points 
initial_num = 4
seed_size = 30

n_folds = 5

init_type = "random"
#init_type = "sobol"
#init_type = "latin"

#####
# Experiment parameters
#####
boston = load_boston()
X, y = boston.data, boston.target
n_features = X.shape[1]

space  = [Real(10**-5, 10**0, name='learning_rate_init'),
          Real(10**-5, 10**0, name='alpha'),
          Integer(1, 100, name='hidden_layer_sizes')]

space_gpyopt = [{"name": "learning_rate_init", "type": "continuous", "domain": (10**-5,10**0)},
                {"name": "alpha", "type": "continuous", "domain": (10**-5,10**0)},
                {"name": "hidden_layer_sizes", "type": "discrete", "domain": np.arange(1,101)},]

# For FITBO

num_continuous_dim = 2
num_discrete_dim = 1
num_categorical_dim = 0

input_dim = num_continuous_dim + num_discrete_dim + num_categorical_dim

input_type = [False, False, True] # True if domain is discrete

continuous_bounds = [(10**-5,10**0), 
                     (10**-5,10**0)]

discrete_bounds = [(1,101)] # upper bound exclusive

fitbo_lb = []
fitbo_ub = []

for i in continuous_bounds:
    fitbo_lb.append(i[0])
    fitbo_ub.append(i[1])
    
for i in discrete_bounds:
    fitbo_lb.append(i[0])
    fitbo_ub.append(i[1])
    
categorical_choice = []

# gradient boosted trees tend to do well on problems like this
reg = MLPRegressor(random_state=0, learning_rate = "constant")

# this decorator allows your objective function to receive a the parameters as
# keyword arguments. This is particularly convenient when you want to set scikit-learn
# estimator parameters
@use_named_args(space)
def objective(**params):
    params["hidden_layer_sizes"] = (params["hidden_layer_sizes"],)
    reg.set_params(**params)
    return -np.mean(cross_val_score(reg, X, y, cv= n_folds, n_jobs=1,
                                    scoring="neg_mean_absolute_error"))

def gpyopt_objective(x):
    x = np.ravel(x)
    
    # Wrapper around "objective" to suit gpyopt notation
    params = {
        "learning_rate_init": x[0],
        "alpha": x[1],
        "hidden_layer_sizes": (int(x[2]),),
    }
    reg.set_params(**params)
    
    return -np.mean(cross_val_score(reg, X, y, cv= n_folds, n_jobs=1,
        scoring="neg_mean_absolute_error"))

def fitbo_objective(X):
    # same as gpyopt_objective, except X is size (num_iter, input_dim)
    y = np.zeros((X.shape[0], 1))

    for i in range(X.shape[0]):
        y[i] = gpyopt_objective(X[i])
        
    return y

####
# Scikit learn wrapper
####

def sklearn_wrapper(acq_func = 'gp_hedge', batch = 1):
    """
    Documentation with scikit-optimize

    https://scikit-optimize.github.io/

    acq_func = "LCB", "EI", "PI", "gp_hedge"
    """
    from skopt import gp_minimize
    dir_name = "Exp_Data/boston_dnn/sklearn/"
    
    for seed in range(seed_size):
        res_gp = gp_minimize(objective, space_gpyopt, n_calls = total_evals, 
            random_state = seed, n_random_starts = initial_num, acq_func = acq_func)
        file_name = dir_name + "batch_" + str(batch) + "," + acq_func + ",seed_" + str(seed)

        with open(file_name, 'wb') as f:
            pickle.dump(res_gp, f)

        print("""Best parameters:
        - max_depth=%f
        - alpha=%f
        - hidden_layer_sizes=%d
        - min_samples_split=%d
        - min_samples_leaf=%d""" % (res_gp.x[0], res_gp.x[1], 
                                    res_gp.x[2]))

# sklearn_wrapper()

####
# GPyOpt learn wrapper
####
            
def saving_data(X_record, min_y_record, x_hist_dict,y_hist_dict, batch_size, acq_func, eval_type):
    """
    For saving data
    """
    dir_name = 'Exp_Data/boston_dnn/gpyopt/' + str(batch_size) + '_batch/'
    file_name = dir_name + str(acq_func) + ',' + str(eval_type) + ',results_vars.pickle'
    
    try: # creates new folder
        os.mkdir(dir_name)
    except FileExistsError:
        pass

    pickle_dict = {
        "X": X_record, 
        "min_y": min_y_record, 
        "x_hist_dict": x_hist_dict,
        "y_hist_dict": y_hist_dict,
        }
    
    with open(file_name, 'wb') as f:
        pickle.dump(pickle_dict, f)
        
    return 0  

def gpyopt_wrapper(acq_func = 'EI', batch_size = 1, eval_type = 'local_penalization'):
    import GPyOpt
    X_record = {}
    min_y_record = {}
    x_hist_dict = {}
    y_hist_dict = {}
        
    for seed in range(seed_size):
        np.random.seed(seed)
        print("Currently on seed: ", seed)
        x_ob = generate_initial_points_x(init_type, seed)
        y_ob = generate_initial_points_y(x_ob)
        
        # Iter= 0 value
        arg_opt = np.argmin(y_ob)
        x_opt_init = x_ob[arg_opt]
        y_opt_init = y_ob[arg_opt]
        
        BO = GPyOpt.methods.BayesianOptimization(f = gpyopt_objective,
                                                domain = space_gpyopt,
                                                acquisition_type = acq_func,
                                                evaluator_type = eval_type,
                                                model_type="GP",
                                                initial_design_numdata = initial_num,
                                                initial_design_type = init_type,
                                                batch_size = batch_size,
                                                n_burning = 100,
                                                n_samples = 150)
        
        BO.run_optimization(max_iter = int(total_evals / batch_size))
        
        # For saving

        x_hist, y_hist = BO.get_evaluations()
        X_opt = BO.return_minimiser() # (rows = iterations, columns = X_dimensions)
        num_iter = X_opt.shape[0]
        min_y = np.zeros((num_iter, 1)) # cols = output dimension
        for i in range(num_iter):
            min_y[i] = gpyopt_objective(X_opt[i])
            
        X_record[seed] = np.vstack((x_opt_init,X_opt[initial_num:])) # Initial samples dont count (keep zero as first point)
        min_y_record[seed] = np.vstack((y_opt_init,min_y[initial_num:]))
        
        x_hist_dict[seed] = x_hist
        y_hist_dict[seed] = y_hist
        
    saving_data(X_record, min_y_record, x_hist_dict, y_hist_dict, batch_size, acq_func, eval_type)

####
# GPyOpt learn wrapper
####

from class_FITBOMM import Bayes_opt
from class_FITBOMM import Bayes_opt_batch

def generate_initial_points_x(init_type, seed):
    if init_type == "random":
        func = utilities.random_mixed_unnormalized
    elif init_type == "sobol":
        func = utilities.sobol_mixed_unnormalized
    a, b, c = func(num_continuous_dim, num_discrete_dim, num_categorical_dim,
                   continuous_bounds, discrete_bounds, categorical_choice,
                   initial_num, seed)
    
    if type(b) != int: # b and c = 0 if no discrete or categorical domain
        init_points = np.hstack((a,b))
    if type(c) != int:
        init_points = np.hstack((init_points, c))

    return init_points

def generate_initial_points_y(X):
    # X is an array of array
    y = np.zeros((X.shape[0],1))
    for i in range(X.shape[0]):
        error = gpyopt_objective(X[i])
        y[i] = error
    return y
        

def FITBO_wrapper(batch_size = 2, heuristic = "cl-min"):
    
    # Setting default values
    BO_method = 'FITBOMM'
    burnin = 100
    sample_size = 50
    resample_interval = 1

    dir_name = "Exp_Data/boston_dnn/fitbo/batch_" + str(batch_size) + "/"
    
    try:
        os.mkdir(dir_name)
    except FileExistsError:
        pass
    
    if batch_size == 1: # Sequential
        heuristic = "sequential"        
        for j in range(seed_size):

            seed = j
            np.random.seed(seed)
            print("Currently on seed: ", j)
            x_ob = generate_initial_points_x(init_type, seed)
            y_ob = generate_initial_points_y(x_ob)
            bayes_opt = Bayes_opt(fitbo_objective, fitbo_lb, fitbo_ub, var_noise = 0)
            bayes_opt.initialise(x_ob, y_ob)
            X_optimum, Y_optimum = bayes_opt.iteration_step(iterations=total_evals, mc_burn=burnin, \
                                                            mc_samples=sample_size, bo_method=BO_method, \
                                                            seed=seed, resample_interval= resample_interval, \
                                                            dir_name = dir_name)
            
            X_file_name = dir_name + "batch_" + str(batch_size) + ",seed_" + str(seed_size) + "," + str(heuristic) + ",X_optimum" 
            Y_file_name = dir_name + "batch_" + str(batch_size) + ",seed_" + str(seed_size) + "," + str(heuristic) + ",Y_optimum"
            X_hist_file_name = dir_name + "batch_" + str(batch_size) + ",seed_" + str(seed_size) + "," + str(heuristic) + ",X_hist" 
            Y_hist_file_name = dir_name + "batch_" + str(batch_size) + ",seed_" + str(seed_size) + "," + str(heuristic) + ",Y_hist" 
            np.save(X_file_name, X_optimum) # results_IR/L2 is np array of shape (num_iterations + 1, seed_size)
            np.save(Y_file_name, Y_optimum)
            np.save(X_hist_file_name, bayes_opt.X) 
            np.save(Y_hist_file_name, bayes_opt.Y)

    else: # Batch
        num_batches = int(total_evals / batch_size)
        results_X_hist = np.zeros(shape=(seed_size, total_evals + initial_num, input_dim)) 
        results_X_optimum = np.zeros(shape=(seed_size, num_batches + 1, input_dim)) 
        results_Y_hist = np.zeros(shape=(seed_size, total_evals + initial_num))  
        results_Y_optimum = np.zeros(shape=(seed_size, num_batches + 1))

        for j in range(seed_size):
            print("Currently on seed: ", j)
            seed = j
            np.random.seed(seed)
            x_ob = generate_initial_points_x(init_type, seed)
            y_ob = generate_initial_points_y(x_ob)
            bayes_opt = Bayes_opt_batch(fitbo_objective, fitbo_lb, fitbo_ub, var_noise = 0, input_type = input_type)
            bayes_opt.initialise(x_ob, y_ob)
            X_optimum, Y_optimum = bayes_opt.iteration_step_batch(num_batches=num_batches, mc_burn=burnin, mc_samples=sample_size, \
                                                                              bo_method=BO_method, seed=seed, resample_interval= resample_interval, \
                                                                              batch_size = batch_size, heuristic = heuristic, 
                                                                              dir_name = dir_name)
            
            results_X_hist[j, :] = bayes_opt.X
            results_X_optimum[j, :] = X_optimum
            results_Y_hist[j, :] = bayes_opt.Y.flatten()
            results_Y_optimum[j, :] = Y_optimum.flatten()
        
        X_file_name = dir_name + "batch_" + str(batch_size) + ",seed_" + str(seed_size) + "," + str(heuristic) + ",X_optimum"  
        Y_file_name = dir_name + "batch_" + str(batch_size) + ",seed_" + str(seed_size) + "," + str(heuristic) + ",Y_optimum" 
        X_hist_file_name = dir_name + "batch_" + str(batch_size) + ",seed_" + str(seed_size) + "," + str(heuristic) + ",X_hist" 
        Y_hist_file_name = dir_name + "batch_" + str(batch_size) + ",seed_" + str(seed_size) + "," + str(heuristic) + ",Y_hist"
        
        np.save(X_file_name, results_X_optimum) 
        np.save(Y_file_name, results_Y_optimum)
        np.save(X_hist_file_name, results_X_hist) 
        np.save(Y_hist_file_name, results_Y_hist)

####
# Running experiments
####    

batch_list = [2,4,8]
heuristic_list = ['cl-min', 'cl-max', 'kb']
# heuristic_list = ['cl-min']
error_list = []

# FITBO_wrapper(batch_size = 1)

for batch in batch_list:
    gpyopt_wrapper(batch_size = batch)  # EI, Local Penalization by default  
    """
    for heur in heuristic_list:
        try:
            FITBO_wrapper(batch_size = batch, heuristic = heur)
        except:
            error_run = heur + str(batch) + "_batch" 
            error_list.append(error_run)
    """
   














