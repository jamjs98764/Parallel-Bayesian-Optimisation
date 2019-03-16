#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 14:49:14 2019

@author: jian
"""


"""
Created on Fri Nov 10 13:45:16 2017

@author: robin
"""
###################
###################
# FITBO
###################
###################

import numpy as np
import scipy as sp
import os
from Test_Funcs import egg,hartmann,branin,func1D
from class_FITBOMM import Bayes_opt
from class_FITBOMM import Bayes_opt_batch

test_func = "branin"

BO_method = 'FITBOMM'
burnin = 100
sample_size = 50
resample_interval = 1
seed_size = 5
num_iterations = 8
batch = False
batch_size = 2
heuristic = "kb"

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
    
else:
    print("Function does not exist in repository")
    

sigma0 = np.sqrt(var_noise)

results_IR = np.zeros(shape=(seed_size, num_iterations + 1)) # Immediate regret
results_L2 = np.zeros(shape=(seed_size, num_iterations + 1)) # L2 norm of x
   

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

    for j in range(seed_size):
        seed = j
        np.random.seed(seed)
        x_ob = np.random.uniform(0., 1., (initialsamplesize, d)) # QUESTION: why not initialized with Latin hypercube or Cobol seq
        y_ob = obj_func(x_ob) + sigma0 * np.random.randn(initialsamplesize, 1)

        bayes_opt = Bayes_opt(obj_func, np.zeros(d), np.ones(d), var_noise)
        bayes_opt.initialise(x_ob, y_ob)
        X_optimum, Y_optimum = bayes_opt.iteration_step(iterations=num_iterations, mc_burn=burnin, \
                                                        mc_samples=sample_size, bo_method=BO_method, \
                                                        seed=seed, resample_interval= resample_interval, \
                                                        dir_name = dir_name)
        results_IR[j, :] = np.abs(Y_optimum - true_min).ravel()

        if test_func == 'branin': # Because branin has 3 global minima
            results_L2_candiate_1 = np.linalg.norm(X_optimum - true_location[0, :], axis=1)
            results_L2_candiate_2 = np.linalg.norm(X_optimum - true_location[1, :], axis=1)
            results_L2_candiate_3 = np.linalg.norm(X_optimum - true_location[2, :], axis=1)
            results_L2_all_candidates = np.array([results_L2_candiate_1, results_L2_candiate_2, results_L2_candiate_3])
            results_L2[j, :] = np.min(results_L2_all_candidates, axis=0).ravel()
        else:
            results_L2[j, :] = np.linalg.norm(X_optimum - true_location[0, :], axis=1).ravel()
            print(np.linalg.norm(X_optimum - true_location[0, :], axis=1).ravel())

        X_opt_file_name = dir_name + 'A_results_L2,sequential' 
        Y_opt_file_name = dir_name + 'A_results_IR,sequential'
        
        np.save(X_opt_file_name, results_L2) # results_IR/L2 is np array of shape (num_iterations + 1, seed_size)
        np.save(Y_opt_file_name, results_IR)
    
###################
###################
# GPyOpt
###################
###################

import GPyOpt
import numpy as np
import Test_Funcs 
from numpy.random import seed
import os

var_noise = 1.0e-3 # y_next = self.func(x_next) + np.random.normal(0, self.var_noise, len(x_next)) 
seed_size = 1

test_func = "branin"
acq_func = "EI"
X_eval_type = "random"
iterations = 8
batch_size = 1

# Noise
sigma0 = np.sqrt(var_noise)

# Values for marginalisation of GP hyperparameters
n_samples = 150
n_burning = 100


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
   

X_record = {}
min_y_record = {}

for seed_i in range(seed_size):
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
        y_ob[i] = obj_func_noise(x_ob[i,:]) 

    y_ob = y_ob + sigma0 * np.random.randn(initialsamplesize, 1) # initial sample points have noise too

    if batch == True:
        # batch
        BO = GPyOpt.methods.BayesianOptimization(f = obj_func_noise,  
                                                domain = domain,                  
                                                acquisition_type = acq_func,
                                                evaluator_type = "random",
                                                model_type=gp_model,              
                                                normalize_Y = False,
                                                X = x_ob,
                                                Y = y_ob,
                                                # NEW CHANGES HERE
                                                #initial_design_numdata = initialsamplesize,
                                                #initial_design_type = 'random',
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
                                                X = x_ob,
                                                Y = y_ob,
                                                
                                                #initial_design_numdata = initialsamplesize,
                                                #initial_design_type = 'random',
                                                
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
    min_y = np.zeros((num_iter, 1)) # cols = output dimension
    for i in range(num_iter):
        min_y[i] = obj_func(X_opt[i])

    X_record[seed_i] = X_opt[initialsamplesize:] # Initial samples dont count
    min_y_record[seed_i] = min_y[initialsamplesize:]
 
