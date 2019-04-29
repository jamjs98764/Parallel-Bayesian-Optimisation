#!/usr/bin/env python3
# -*- coding: utf-8 -*-



# Import modules
import numpy as np
from pyswarms.single.global_best import GlobalBestPSO
from Test_Funcs import branin_pso, egg_pso, hartmann_pso
import pickle
import os


var_noise = 1.0e-3  # Same noise setting as FITBO tests
sigma0 = np.sqrt(var_noise)
num_iters = 80
seed_size = 50

options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

def test_pso(test_func, seed_size, num_iters, batch_size):
    if test_func == 'branin':
        obj_func = branin_pso
        d = 2
        x_max = 1 * np.ones(2)
        x_min = 0 * x_max
        bounds = (x_min, x_max)
        initialsamplesize = 3

    elif test_func == 'egg':
        obj_func = egg_pso
        d = 2
        x_max = 1 * np.ones(2)
        x_min = 0 * x_max
        bounds = (x_min, x_max)
        initialsamplesize = 3

    elif test_func == 'hartmann':
        obj_func = hartmann_pso
        d = 6
        x_max = 1 * np.ones(6)
        x_min = 0 * x_max
        bounds = (x_min, x_max)
        initialsamplesize = 9

    else:
        print("Function does not exist in repository")
        return 0

    def obj_func_noise(x):
        noiseless = obj_func(x)
        iters = noiseless.shape[0]
        noisy_eval = noiseless + sigma0 * np.random.random((1,iters))
        return noisy_eval[0]

    # Creating dicts to store
    pos_dict = {}
    cost_dict = {}

    for seed_i in range(seed_size):
        np.random.seed(seed_i)
        """
        # Generating initial samples
        init_pos = np.random.random((initialsamplesize, d))
        print(init_pos)
        """
        # Running optimisation
        optimizer = GlobalBestPSO(n_particles = batch_size, dimensions=d, options=options, bounds=bounds, init_pos = None)
        best_cost, best_pos = optimizer.optimize(obj_func_noise, num_iters + 1, seed_i)

        # Recording results
        pos_hist = optimizer.pos_history
        cost_hist = optimizer.cost_history

        pos_dict[seed_i] = pos_hist
        cost_dict[seed_i] = np.transpose(np.array([cost_hist]))

    return pos_dict, cost_dict

def saving_data(pos_dict, cost_dict):
    """
    For saving data

    pos_dict =
    cost_dict = y_opt
    """
    dir_name = 'Exp_Data/pso/' + test_func + ',' + str(seed_size) + '_seed,' + str(batch_size) + '_batch/'
    file_name = dir_name + 'results_vars.pickle'

    try: # creates new folder
        os.mkdir(dir_name)
    except FileExistsError:
        pass

    pickle_dict = {
        "X": pos_dict,
        "min_y": cost_dict,
        }

    with open(file_name, 'wb') as f:
        pickle.dump(pickle_dict, f)

    return 0

############
# Running experiments
############

batch_sizes = [8]
test_funcs = ["egg", "branin", "hartmann"]

for test_func in test_funcs:
    for batch_size in batch_sizes:
        print(test_func, batch_size)
        effective_iters = int(num_iters/batch_size)
        pos_dict, cost_dict = test_pso(test_func, seed_size, effective_iters, batch_size)
        saving_data(pos_dict, cost_dict)

from plotting_utilities import *

test = load_pso_error("egg", "IR", 2, 50)

a, b = load_pickle_pso("egg", 50, 8)