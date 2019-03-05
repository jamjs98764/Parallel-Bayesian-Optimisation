# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 16:08:52 2018

@author: jianhong

Performs GradientBoosting Regressor on Boston dataset
Tries 3 different packages: scikit, gpyopt, fitbo

Records cross-validation error at each iteration
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.datasets import load_boston
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from skopt.space import Real, Integer
from skopt.utils import use_named_args

boston = load_boston()
X, y = boston.data, boston.target
n_features = X.shape[1]

# Input domain range
space  = [Integer(1, 5, name='max_depth'),
          Real(10**-5, 10**0, "log-uniform", name='learning_rate'),
          Integer(1, n_features, name='max_features'),
          Integer(2, 100, name='min_samples_split'),
          Integer(1, 100, name='min_samples_leaf')]

space_gpyopt = [{"name": "max_depth", "type": "discrete", "domain": (1,2,3,4,5)},
				{"name": "learning_rate", "type": "continuous", "domain": (10**-5,10**0)},
				{"name": "max_features", "type": "discrete", "domain": (1,n_features)},
				{"name": "min_samples_split", "type": "discrete", "domain": tuple(np.arange(2,101))},
				{"name": "min_samples_leaf", "type": "discrete", "domain": tuple(np.arange(1,101))},
				]


"""
total_evals = 80
initial_num = 10

seed_size = 30
"""
total_evals = 5
initial_num = 2

seed_size = 3

n_folds = 5

# gradient boosted trees tend to do well on problems like this
reg = GradientBoostingRegressor(n_estimators=50, random_state=0)

# this decorator allows your objective function to receive a the parameters as
# keyword arguments. This is particularly convenient when you want to set scikit-learn
# estimator parameters
@use_named_args(space)
def objective(**params):
    reg.set_params(**params)
    return -np.mean(cross_val_score(reg, X, y, cv= n_folds, n_jobs=1,
                                    scoring="neg_mean_absolute_error"))

def gpyopt_objective(x):
	x = x[0]
	print(x)
	# Wrapper around "objective" to suit gpyopt notation
	params = {
		"max_depth": x[0],
		"learning_rate": x[1],
		"max_features": int(x[2]),
		"min_samples_split": int(x[3]),
		"min_samples_leaf": int(x[4]),
	}
	print(params)
	reg.set_params(**params)
	return -np.mean(cross_val_score(reg, X, y, cv= n_folds, n_jobs=1,
		scoring="neg_mean_absolute_error"))

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
	dir_name = "Exp_Data/boston_gbr/sklearn/"
	
	for seed in range(seed_size):
		res_gp = gp_minimize(objective, space_gpyopt, n_calls = total_evals, 
			random_state = seed, n_random_starts = initial_num, acq_func = acq_func)
		file_name = dir_name + "batch_" + str(batch) + "," + acq_func + ",seed_" + str(seed)

		with open(file_name, 'wb') as f:
			pickle.dump(res_gp, f)

		print("""Best parameters:
		- max_depth=%d
		- learning_rate=%.6f
		- max_features=%d
		- min_samples_split=%d
		- min_samples_leaf=%d""" % (res_gp.x[0], res_gp.x[1], 
		                            res_gp.x[2], res_gp.x[3], 
		                            res_gp.x[4]))

# sklearn_wrapper()

####
# GPyOpt learn wrapper
####

def gpyopt_wrapper(acq_func = 'EI', batch_size = 1, eval_type = 'local_penalization'):
	import GPyOpt
	dir_name = "Exp_Data/boston_gbr/gpyopt/"
	
	for seed in range(seed_size):
		BO = GPyOpt.methods.BayesianOptimization(f = gpyopt_objective,
												domain = space_gpyopt,
												acquisition_type = acq_func,
												evaluator_type = eval_type,
												model_type="GP",
												initial_design_numdata = initial_num,
                                                initial_design_type = 'random',
												batch_size = batch_size,
												n_burning = 100,
												n_samples = 150)
		
		BO.run_optimization(max_iter = int(total_evals / batch_size))

		file_name = dir_name + "batch_" + str(batch_size) + "," + acq_func + ",seed_" + str(seed_size)

		with open(file_name, 'wb') as f:
			pickle.dump(BO, f)

# gpyopt_wrapper()


####
# GPyOpt learn wrapper
####

from class_FITBOMM import Bayes_opt
from class_FITBOMM import Bayes_opt_batch

def FITBO_wrapper(batch_size = 2, heuristic = "kb"):
	
	# Setting default values
    BO_method = 'FITBOMM'
    burnin = 100
    sample_size = 50
    resample_interval = 1

    dir_name = "Exp_Data/boston_gbr/fitbo/"
    
    if batch == False: # Sequential
        results_error = np.zeros(shape=(seed_size, num_iterations + 1)) 

        for j in range(seed_size):
            seed = j
            np.random.seed(seed)
            x_ob = np.random.uniform(0., 1., (initialsamplesize, d)) 
            y_ob = obj_func(x_ob) + sigma0 * np.random.randn(initialsamplesize, 1)
    
            bayes_opt = Bayes_opt(obj_func, np.zeros(d), np.ones(d), var_noise)
            bayes_opt.initialise(x_ob, y_ob)
            X_optimum, Y_optimum = bayes_opt.iteration_step(iterations=num_iterations, mc_burn=burnin, \
                                                            mc_samples=sample_size, bo_method=BO_method, \
                                                            seed=seed, resample_interval= resample_interval, \
                                                            dir_name = dir_name)
            results_IR[j, :] = np.abs(Y_optimum - true_min).ravel()
    
            np.save(X_opt_file_name, results_L2) # results_IR/L2 is np array of shape (num_iterations + 1, seed_size)
            np.save(Y_opt_file_name, results_IR)

    if batch == True:
        num_batches = int(num_iterations / batch_size)
        results_error = np.zeros(shape=(seed_size, num_batches + 1)) 
        
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
            
            X_opt_file_name = dir_name + 'A_results_L2,' + heuristic + '_heuristic'
            Y_opt_file_name = dir_name + 'A_results_IR,' + heuristic + '_heuristic' 
            np.save(X_opt_file_name, results_L2)
            np.save(Y_opt_file_name, results_IR)
