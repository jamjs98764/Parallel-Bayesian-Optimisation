
"""
Created on Fri Nov 10 13:45:16 2017

@author: robin
"""

import numpy as np
import scipy as sp
import os
from Test_Funcs import egg,hartmann,branin,func1D
from class_FITBOMM import Bayes_opt
from class_FITBOMM import Bayes_opt_batch


def BO_test(test_func, BO_method, burnin = 100, sample_size = 50, resample_interval = 1, \
            seed_size = 30, num_iterations = 40, batch = False, batch_size = 2, heuristic = "kb"):

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
        
    else:
        print("Function does not exist in repository")
        return 0
    
    sigma0 = np.sqrt(var_noise)

    results_IR = np.zeros(shape=(seed_size, num_iterations + 1)) # Immediate regret
    results_L2 = np.zeros(shape=(seed_size, num_iterations + 1)) # L2 norm of x
    
    # Creating directory to save
    dir_name = 'Exp_Data/' + test_func + ',' + str(seed_size) + '_seed,' + str(batch_size) + '_batch_size/' 
   
    try:
        os.mkdir(dir_name)
    except FileExistsError:
        pass
    
    if batch == False:
        results_IR = np.zeros(shape=(seed_size, num_iterations + 1)) # Immediate regret
        results_L2 = np.zeros(shape=(seed_size, num_iterations + 1)) # L2 norm of x

        for j in range(seed_size):
            seed = j
            np.random.seed(seed)
            x_ob = np.random.uniform(0., 1., (initialsamplesize, d)) # QUESTION: why not initialized with Latin hypercube or Cobol seq
            y_ob = obj_func(x_ob) + sigma0 * np.random.randn(initialsamplesize, 1)
    
            bayes_opt = Bayes_opt(obj_func, np.zeros(d), np.ones(d), var_noise)
            bayes_opt.initialise(x_ob, y_ob)
            print("Initialised")
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

    if batch == True:
        num_batches = int(num_iterations / batch_size)
        results_IR = np.zeros(shape=(seed_size, num_batches + 1)) # Immediate regret
        results_L2 = np.zeros(shape=(seed_size, num_batches + 1)) # L2 norm of x
        
        for j in range(seed_size):
            seed = j
            np.random.seed(seed)
            x_ob = np.random.uniform(0., 1., (initialsamplesize, d)) # QUESTION: why not initialized with Latin hypercube or Cobol seq
            y_ob = obj_func(x_ob) + sigma0 * np.random.randn(initialsamplesize, 1)
    
            bayes_opt = Bayes_opt_batch(obj_func, np.zeros(d), np.ones(d), var_noise)
            bayes_opt.initialise(x_ob, y_ob)
            X_optimum, Y_optimum = bayes_opt.iteration_step_batch(num_batches=num_batches, mc_burn=burnin, mc_samples=sample_size, \
                                                                              bo_method=BO_method, seed=seed, resample_interval= resample_interval, \
                                                                              batch_size = batch_size, heuristic = heuristic, 
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
                print(results_L2)
            
            X_opt_file_name = dir_name + 'A_results_L2,' + heuristic + '_heuristic'
            Y_opt_file_name = dir_name + 'A_results_IR,' + heuristic + '_heuristic' 
            np.save(X_opt_file_name, results_L2)
            np.save(Y_opt_file_name, results_IR)
        
        
#####
# Running tests
#####

def test_all(test_func, current_batch_size):    
    ## Single test sequential

    BO_test(test_func = test_func, BO_method = 'FITBOMM', batch_size = current_batch_size)
    
    ## Single test batch
    
    BO_test(test_func = test_func, BO_method = 'FITBOMM', batch = True, batch_size = current_batch_size, heuristic = 'kb')
    BO_test(test_func = test_func, BO_method = 'FITBOMM', batch = True, batch_size = current_batch_size, heuristic = 'cl-mean')
    BO_test(test_func = test_func, BO_method = 'FITBOMM', batch = True, batch_size = current_batch_size, heuristic = 'cl-min')
    BO_test(test_func = test_func, BO_method = 'FITBOMM', batch = True, batch_size = current_batch_size, heuristic = 'cl-max')
    
    BO_test(test_func = test_func, BO_method = 'FITBOMM', batch = True, batch_size = current_batch_size, heuristic = 'random')  
    
    BO_test(test_func = test_func, BO_method = 'FITBOMM', batch = True, batch_size = current_batch_size, heuristic = 'random_except_1st')  
    
    return None


batch_sizes = [4]
test_funcs = ["egg", "branin", "hartmann"]

for batch_size in batch_sizes:
    for test_func in test_funcs:
        test_all(test_func, batch_size)
        
print("Finished 4-batch tests")

"""
batch_sizes = [8]
test_funcs = ["egg", "branin", "hartmann"]

for batch_size in batch_sizes:
    for test_func in test_funcs:
        test_all(test_func, batch_size)

print("Finished 8-batch tests")
"""

