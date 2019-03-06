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
import utilities


"""
total_evals = 80
initial_num = 10

seed_size = 30
"""
total_evals = 32
initial_num = 4

seed_size = 2

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

space  = [Real(10**-5, 10**0, name='learning_rate'),
          Integer(1, 5, name='max_depth'),
          Integer(1, n_features, name='max_features'),
          Integer(2, 100, name='min_samples_split'),
          Integer(1, 100, name='min_samples_leaf')]

space_gpyopt = [{"name": "learning_rate", "type": "continuous", "domain": (10**-5,10**0)},
                {"name": "max_depth", "type": "discrete", "domain": (1,2,3,4,5)},
                {"name": "max_features", "type": "discrete", "domain": tuple(np.arange(1,n_features))},
                {"name": "min_samples_split", "type": "discrete", "domain": tuple(np.arange(2,101))},
                {"name": "min_samples_leaf", "type": "discrete", "domain": tuple(np.arange(1,101))},]

# For FITBO
"""
num_continuous_dim, num_discrete_dim, num_categorical_dim, 
                            continuous_bounds, discrete_bounds, categorical_choice, 
"""

num_continuous_dim = 1
num_discrete_dim = 4 
num_categorical_dim = 0

continuous_bounds = [(10**-5,10**0)]

discrete_bounds = [(1,5),
                   (1,n_features),
                   (2,201),
                   (1,101)]

fitbo_lb = [continuous_bounds[0][0]]
fitbo_ub = [continuous_bounds[0][1]]

for i in discrete_bounds:
    fitbo_lb.append(i[0])
    fitbo_ub.append(i[1])
    
categorical_choice = []

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
    x = np.ravel(x)

    # Wrapper around "objective" to suit gpyopt notation
    params = {
        "learning_rate": x[0],
        "max_depth": x[1],
        "max_features": int(x[2]),
        "min_samples_split": int(x[3]),
        "min_samples_leaf": int(x[4]),
    }
    reg.set_params(**params)
    return -np.mean(cross_val_score(reg, X, y, cv= n_folds, n_jobs=1,
        scoring="neg_mean_absolute_error"))

def fitbo_objective(X):
    # same as gpyopt_objective, except X is size (num_iter, input_dim)
    y = np.zeros((X.shape[0], 1))
    print("X")
    print(X)
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
                                            initial_design_type = init_type,
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

    dir_name = "Exp_Data/boston_gbr/fitbo/"
    
    if batch_size == 1: # Sequential

        for j in range(seed_size):
            seed = j
            np.random.seed(seed)
            
            x_ob = generate_initial_points_x(init_type, seed)
            y_ob = generate_initial_points_y(x_ob)
            bayes_opt = Bayes_opt(fitbo_objective, fitbo_lb, fitbo_ub, var_noise = 0)
            bayes_opt.initialise(x_ob, y_ob)
            X_optimum, Y_optimum = bayes_opt.iteration_step(iterations=total_evals, mc_burn=burnin, \
                                                            mc_samples=sample_size, bo_method=BO_method, \
                                                            seed=seed, resample_interval= resample_interval, \
                                                            dir_name = dir_name)
            
            X_file_name = dir_name + "batch_" + str(batch_size) + ",seed_" + str(seed_size)
            Y_file_name = dir_name + "batch_" + str(batch_size) + ",seed_" + str(seed_size)
            np.save(X_file_name, X_optimum) # results_IR/L2 is np array of shape (num_iterations + 1, seed_size)
            np.save(Y_file_name, Y_optimum)

    else: # Batch
        num_batches = int(total_evals / batch_size)
        
        for j in range(seed_size):
            seed = j
            np.random.seed(seed)
            x_ob = generate_initial_points_x(init_type, seed)
            y_ob = generate_initial_points_y(x_ob)
            
            bayes_opt = Bayes_opt_batch(fitbo_objective, fitbo_lb, fitbo_ub, var_noise = 0)
            bayes_opt.initialise(x_ob, y_ob)
            X_optimum, Y_optimum = bayes_opt.iteration_step_batch(num_batches=num_batches, mc_burn=burnin, mc_samples=sample_size, \
                                                                              bo_method=BO_method, seed=seed, resample_interval= resample_interval, \
                                                                              batch_size = batch_size, heuristic = heuristic, 
                                                                              dir_name = dir_name)
            X_file_name = dir_name + "batch_" + str(batch_size) + ",seed_" + str(seed_size)
            Y_file_name = dir_name + "batch_" + str(batch_size) + ",seed_" + str(seed_size)
            np.save(X_file_name, X_optimum) # results_IR/L2 is np array of shape (num_iterations + 1, seed_size)
            np.save(Y_file_name, Y_optimum)
    
            
FITBO_wrapper(batch_size = 2)