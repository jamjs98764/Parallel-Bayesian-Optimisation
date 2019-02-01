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

var_noise = 1.0e-3 # y_next = self.func(x_next) + np.random.normal(0, self.var_noise, len(x_next)) 

def wrapper_GPyOpt(test_func, acq_func = "EI", eval_type = "random", \
    seed_size = 30, iterations = 40, batch_size = 1,):
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
    
    for seed in seed_size:
        np.random.seed(seed)

        # Noisy function
        def obj_func_noise(x):
            noisy_eval = obj_func(x) + sigma0 * np.random.randn()

        # Generating initialising points
        x_ob = np.random.uniform(0., 1., (initialsamplesize, d)) 
        y_ob = obj_func(x_ob) + sigma0 * np.random.randn(initialsamplesize, 1)

        if batch == False:
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
                                                    evaluator_type = eval_type,
                                                    model_type='GP',              
                                                    normalize_Y = False,
                                                    initial_design_numdata = 0,
                                                    X = x_ob,
                                                    Y = y_ob,
                                                    batch_size = batch_size,
                                                    num_cores = num_cores,
                                                    acquisition_jitter = 0)
            BO.run_optimization(max_iter = iterations)           
            
        query_record = BO.get_evaluations()[0]
        
    return BO, query_record
    

#BO.plot_convergence()
#BO.plot_acquisition()

test_func, acq_func = "EI", eval_type = "random", \
    seed_size = 30, iterations = 40, batch = False, batch_size = 1,)

batch_sizes = [1, 4]
test_funcs = ["hartmann"]

for batch_size in batch_sizes:
    for test_func in test_funcs:
        BO, query_record = wrapper_GPyOpt("branin")

x, y = BO.return_convergence()



"""
Batch BO using GPyOpt 

Documentation:
https://gpyopt.readthedocs.io/en/latest/GPyOpt.core.html#GPyOpt.core.bo.BO
https://gpyopt.readthedocs.io/en/latest/GPyOpt.methods.html#

Sample:

objective_true  = GPyOpt.objective_examples.experiments2d.branin()
objective_noisy = GPyOpt.objective_examples.experiments2d.branin(sd = var_noise)
bounds = objective_noisy.bounds

objective_true.plot()

domain = [{'name': 'var_1', 'type': 'continuous', 'domain': bounds[0]}, ## use default bounds
          {'name': 'var_2', 'type': 'continuous', 'domain': bounds[1]}]

batch_size = 1
num_cores = 1

BO_demo_parallel = GPyOpt.methods.BayesianOptimization(f=objective_noisy.f,  
                                            domain = domain,                  
                                            acquisition_type = 'EI',              
                                            normalize_Y = True,
                                            initial_design_numdata = 3,
                                            evaluator_type = 'random',
                                            batch_size = batch_size,
                                            num_cores = num_cores,
                                            acquisition_jitter = 0)

# --- Run the optimization for 10 iterations
max_iter = 20                                       
BO_demo_parallel.run_optimization(max_iter)    

BO_demo_parallel.plot_acquisition()

BO_demo_parallel.plot_convergence()

# Get past evaluatiosn
past_query = BO_demo_parallel.get_evaluations()[0]
"""

