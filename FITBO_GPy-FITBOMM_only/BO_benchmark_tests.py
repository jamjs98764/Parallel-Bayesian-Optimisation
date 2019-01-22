# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 15:18:49 2019

@author: jianhong
"""
import GPyOpt
import numpy as np
import scipy as sp
import Test_Funcs 
from functools import partial
from numpy.random import seed
seed(123)

var_noise = 1.0e-3 # y_next = self.func(x_next) + np.random.normal(0, self.var_noise, len(x_next)) 

def wrapper_GPyOpt(test_func, acq_func = "EI", iterations = 20, batch = False, batch_size = 1,):
    """
    Wrapper function which implements GPyOpt BO
    Returns all query points
    
    acq_func: "EI" / "EI_MCMC" / "MPI_MCMC" /  "LCB" / "LCB_MCMC"    
    """
    num_cores = 1 
    
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
    
    if batch == False:
        BO = GPyOpt.methods.BayesianOptimization(f = obj_func,  
                                                domain = domain,                  
                                                acquisition_type = acq_func,              
                                                normalize_Y = True,
                                                initial_design_numdata = initialsamplesize,
                                                evaluator_type = 'random',
                                                batch_size = batch_size,
                                                num_cores = num_cores,
                                                acquisition_jitter = 0)
        BO.run_optimization(max_iter = int(iterations / batch_size))
        
    query_record = BO.get_evaluations()[0]
    
    return BO, query_record
    
BO, query_record = wrapper_GPyOpt("branin")
BO.plot_convergence()
        
"""
Batch BO using GPyOpt 

Documentation:
https://gpyopt.readthedocs.io/en/latest/GPyOpt.core.html#GPyOpt.core.bo.BO
https://gpyopt.readthedocs.io/en/latest/GPyOpt.methods.html#

"""

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


