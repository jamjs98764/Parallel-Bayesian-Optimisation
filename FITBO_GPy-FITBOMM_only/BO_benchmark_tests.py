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

var_noise = 1.0e-3 # y_next = self.func(x_next) + np.random.normal(0, self.var_noise, len(x_next)) 

def wrapper_GPyOpt(test_func, acq_func = "EI", eval_type = "random", \
    seed_size = 50, iterations = 40, batch_size = 2):
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
    min_y_record = {}

    for seed_i in range(seed_size):
        np.random.seed(seed_i)

        # Noisy function
        def obj_func_noise(x):
            noisy_eval = obj_func(x) + sigma0 * np.random.randn()
            return noisy_eval

        # Generating initialising points
        x_ob = np.random.uniform(0., 1., (initialsamplesize, d)) 
        y_ob = obj_func(x_ob) + sigma0 * np.random.randn(initialsamplesize, 1)

        if batch == True:
            # batch
            BO = GPyOpt.methods.BayesianOptimization(f = obj_func_noise,  
                                                    domain = domain,                  
                                                    acquisition_type = acq_func,
                                                    evaluator_type = eval_type,
                                                    model_type='GP',              
                                                    normalize_Y = False,
                                                    initial_design_numdata = 0,
                                                    X = x_ob,
                                                    Y = y_ob,
                                                    batch_size = batch_size,
                                                    num_cores = num_cores,
                                                    acquisition_jitter = 0)
            BO.run_optimization(max_iter = int(iterations / batch_size))
        else:
            # sequential
            BO = GPyOpt.methods.BayesianOptimization(f = obj_func_noise,  
                                                    domain = domain,                  
                                                    acquisition_type = acq_func,
                                                    model_type='GP',              
                                                    normalize_Y = False,
                                                    initial_design_numdata = 0,
                                                    X = x_ob,
                                                    Y = y_ob,
                                                    batch_size = batch_size,
                                                    num_cores = num_cores,
                                                    acquisition_jitter = 0)
            BO.run_optimization(max_iter = int(iterations))           
            
        eval_record = BO.get_evaluations()[0]
        x, min_y = BO.return_convergence()
        X_record[seed_i] = x[initialsamplesize:] # Initial samples dont count
        min_y_record[seed_i] = min_y[initialsamplesize:]
        
    return X_record, min_y_record   


def saving_data(X_record, min_y_record):
    """
    For saving data
    """
    dir_name = 'Exp_Data/gpyopt/' + test_func + ',' + str(2) + '_seed,' + str(batch_size) + '_batch/'
    file_name = dir_name + str(acq_func) + ',' + str(eval_type) + ',results_vars.pickle'
    
    try: # creates new folder
        os.mkdir(dir_name)
    except FileExistsError:
        pass
    
    pickle_dict = {
        "X": X_record, 
        "min_y": min_y_record
        }
    
    with open(file_name, 'wb') as f:
        pickle.dump(pickle_dict, f)
        
    return 0    

#acq_funcs =  ["EI", "EI_MCMC", "MPI_MCMC",  "LCB", "LCB_MCMC"]
#evaluator_types = ["sequential", "random", "local_penalization", "thompson_sampling"]  

batch_sizes = [4]
test_funcs = ["egg"]
acq_funcs =  ["EI", "LCB"]
evaluator_types = ["random", "local_penalization"] # does not matter for batch size = 1  

for test_func in test_funcs:
    for batch_size in batch_sizes:
        for acq_func in acq_funcs:
            for eval_type in evaluator_types:
                print(test_func, batch_size, acq_func, eval_type)
                X_record, min_y_record = wrapper_GPyOpt(test_func, acq_func = acq_func, eval_type = eval_type, \
                                                  batch_size = batch_size)
                saving_data(X_record, min_y_record)
                    
batch_sizes = [1]
test_funcs = ["egg"]
acq_funcs =  ["EI", "LCB"]
evaluator_types = ["sequential"] # does not matter for batch size = 1  

for test_func in test_funcs:
    for batch_size in batch_sizes:
        for acq_func in acq_funcs:
            for eval_type in evaluator_types:
                print(test_func, batch_size, acq_func, eval_type)
                X_record, min_y_record = wrapper_GPyOpt(test_func, acq_func = acq_func, eval_type = eval_type, \
                                                  batch_size = batch_size)
                saving_data(X_record, min_y_record)


#BO.plot_convergence()
#BO.plot_acquisition()

