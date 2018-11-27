
"""
Created on Fri Nov 10 13:45:16 2017

@author: robin
"""

import numpy as np
import scipy as sp
from Test_Funcs import egg,hartmann,branin,func1D
from class_FITBOMM import Bayes_opt

def BO_test(test_func, BO_method, burnin=100, sample_size=50, interval=1):
    # BO_method is either FITBOMM (moment matching) or FITBO (quadrature) 
    seed_size = 20 # Run BO experiment for x times 
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
    else:
        obj_func = hartmann
        d = 6
        initialsamplesize = 9
        true_min = np.array([-18.22368011])
        true_location = np.array([[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]])

    var_noise = 1.0e-3
    num_iterations = 50
    sigma0 = np.sqrt(var_noise)

    results_IR = np.zeros(shape=(seed_size, num_iterations + 1)) # Immediate regret
    results_L2 = np.zeros(shape=(seed_size, num_iterations + 1)) # L2 norm of x

    for j in range(seed_size):
        seed = j
        np.random.seed(seed)
        x_ob = np.random.uniform(0., 1., (initialsamplesize, d)) # QUESTION: why not initialized with Latin hypercube or Cobol seq
        y_ob = obj_func(x_ob) + sigma0 * np.random.randn(initialsamplesize, 1)

        bayes_opt = Bayes_opt(obj_func, np.zeros(d), np.ones(d), var_noise)
        bayes_opt.initialise(x_ob, y_ob)
        X_optimum, Y_optimum = bayes_opt.iteration_step(iterations=num_iterations, mc_burn=burnin, mc_samples=sample_size, \
                                                                          bo_method=BO_method, seed=seed, resample_interval=interval)

        results_IR[j, :] = np.abs(Y_optimum - true_min).ravel()

        if test_func == 'branin': # Because branin has 3 global minima
            results_L2_candiate_1 = np.linalg.norm(X_optimum - true_location[0, :], axis=1)
            results_L2_candiate_2 = np.linalg.norm(X_optimum - true_location[1, :], axis=1)
            results_L2_candiate_3 = np.linalg.norm(X_optimum - true_location[2, :], axis=1)
            results_L2_all_candidates = np.array([results_L2_candiate_1, results_L2_candiate_2, results_L2_candiate_3])
            results_L2[j, :] = np.min(results_L2_all_candidates, axis=0).ravel()
        else:
            results_L2[j, :] = np.linalg.norm(X_optimum - true_location[0, :], axis=1).ravel()

        X_opt_file_name = 'Exp_Data/' + test_func + BO_method + '_results_L2'
        Y_opt_file_name = 'Exp_Data/' + test_func + BO_method + '_results_IR'

        np.save(X_opt_file_name, results_IR)
        np.save(Y_opt_file_name, results_L2)

if __name__ == '__main__':
    BO_test(test_func='egg', BO_method='FITBOMM')
