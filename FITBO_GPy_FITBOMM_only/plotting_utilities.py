#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 17 March 2019

@author: Jian

Helper functions used by ipynb notebooks to plot diagrams
"""
import numpy as np
import seaborn as sns
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os


def load_pickle_gpyopt(func, seed_size, batch_size, acq_func, eval_type, ard):
    if ard:
        dir_name = "Exp_Data/gpyopt_ard/" + func + ',' + str(seed_size) + '_seed,' + str(batch_size) + '_batch/'
    else:
        dir_name = "Exp_Data/gpyopt/" + func + ',' + str(seed_size) + '_seed,' + str(batch_size) + '_batch/'
    file_name = dir_name + acq_func + ',' + eval_type + ',results_vars.pickle'

    with open(file_name, 'rb') as f:  # Python 3: open(..., 'rb')
        pickle_dict = pickle.load(f)
        X = pickle_dict["X"]
        min_y = pickle_dict["min_y"]

    return X, min_y


def load_pickle_pso(func, seed_size, batch_size):
    dir_name = "Exp_Data/pso/" + func + ',' + str(seed_size) + '_seed,' + str(batch_size) + '_batch/'
    file_name = dir_name + 'results_vars.pickle'

    with open(file_name, 'rb') as f:  # Python 3: open(..., 'rb')
        pickle_dict = pickle.load(f)
        X = pickle_dict["X"]
        min_y = pickle_dict["min_y"]

    return X, min_y

# acq_func: "EI" / "EI_MCMC" / "MPI_MCMC" /  "LCB" / "LCB_MCMC"
# evaluator_type: sequential / random  (1st random in Jian's deifnition) / local_penalization / thompson_sampling

def load_gpyopt_error(func, metric, batch_size, seed_size, acq_func, eval_type, ard = False):
    x, y = load_pickle_gpyopt(func, seed_size, batch_size, acq_func, eval_type, ard)
    if metric == "L2":
        result = unpack_l2(func, x)
    elif metric == "IR":
        result = unpack_IR(func, y)

    if batch_size > 1:
        result = np.repeat(result, repeats = batch_size * np.ones(result.shape[1], dtype = int), axis = 1)
        result = result[:,(batch_size-1):] # Do not duplicate initial error

    df = np_to_df(result)
    return df

def load_pso_error(func, metric, batch_size, seed_size):
    x, y = load_pickle_pso(func, seed_size, batch_size)
    if metric == "L2":
        result = unpack_l2(func, x)
    elif metric == "IR":
        result = unpack_IR(func, y)

    if batch_size > 1:
        result = np.repeat(result, repeats = batch_size * np.ones(result.shape[1], dtype = int), axis = 1)
        result = result[:,(batch_size-1):] # Do not duplicate initial error

    df = np_to_df(result)
    return df

def unpack_IR(func, dic):
    # Used to unpack min_y values from gpyopt pickle dictionary to get Immediate Regret
    max_seed = max(dic.keys()) + 1 # how many seeds
    col_size = dic[0].shape[0] # how many iterations

    IR = np.zeros((max_seed, col_size))
    # Minimum y value
    min_y_dict = {
        "hartmann": np.array([-18.22368011]),
        "egg": np.array([-9.596407]),
        "branin": np.array([-14.96021125])
    }

    true_min_y = min_y_dict[func]

    for seed_i in range(max_seed):
        current_min_y = dic[seed_i]

        if current_min_y.shape[0] == IR.shape[1]: # check whether matches, as gpyopt has errors
            IR[seed_i,:] = abs(dic[seed_i] - true_min_y).flatten()
        else:
            print("Seed " + str(seed_i) + " has errors in data.")
            IR[seed_i,:] = np.zeros(IR.shape[1])

    IR = IR[~np.all(IR == 0, axis=1)] # remove rows with full zero (errors)
    IR = min_y_hist(IR)

    return IR

def unpack_l2(func, dic):
    # Used to unpack x values from gpyopt pickle dictionary to get L2 error
    max_seed = max(dic.keys()) + 1 # how many seeds
    col_size = dic[0].shape[0] # how many iterations
    dim = dic[0].shape[1] # input dimensions
    l2_norm = np.zeros((max_seed, col_size))

    # Minimiser x value
    min_x_dict = {
        "hartmann": np.array([[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]]),
        "egg": np.array([[1.0, 0.7895]]),
        "branin": np.array([[0.1239, 0.8183],[0.5428, 0.1517],[0.9617, 0.1650]])
    }

    true_x_min = min_x_dict[func]

    for seed_i in range(max_seed):
        if func == "branin":
            d1 = np.linalg.norm(true_x_min[0] - dic[seed_i], axis = 1)
            d2 = np.linalg.norm(true_x_min[1] - dic[seed_i], axis = 1)
            d3 = np.linalg.norm(true_x_min[2] - dic[seed_i], axis = 1)
            all_d = np.array([d1, d2, d3])
            l2_distance = np.min(all_d, axis = 0).ravel()

        else:
            l2_distance = np.linalg.norm(true_x_min - dic[seed_i], axis = 1).ravel()

        if l2_distance.shape[0] == l2_norm.shape[1]: # check whether matches, as gpyopt has errors
            l2_norm[seed_i,:] = l2_distance
        else:
            print("Seed " + str(seed_i) + " has errors in data.")
            l2_norm[seed_i,:] = np.zeros(l2_norm.shape[1])

    l2_norm = l2_norm[~np.all(l2_norm == 0, axis=1)] # remove rows with full zero (errors)

    return l2_norm

def unpack_l2_pso(func, dic):
    # Used to unpack x values from pso dictionary to get L2 error
    max_seed = max(dic.keys()) + 1 # how many seeds
    col_size = dic[0].shape[0] # how many iterations
    dim = dic[0].shape[1] # input dimensions
    l2_norm = np.zeros((max_seed, col_size))

    # Minimiser x value
    min_x_dict = {
        "hartmann": np.array([[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]]),
        "egg": np.array([[1.0, 0.7895]]),
        "branin": np.array([[0.1239, 0.8183],[0.5428, 0.1517],[0.9617, 0.1650]])
    }

    true_x_min = min_x_dict[func]

    for seed_i in range(max_seed):
        if func == "branin":
            d1 = np.linalg.norm(true_x_min[0] - dic[seed_i], axis = 1)
            d2 = np.linalg.norm(true_x_min[1] - dic[seed_i], axis = 1)
            d3 = np.linalg.norm(true_x_min[2] - dic[seed_i], axis = 1)
            all_d = np.array([d1, d2, d3])
            l2_distance = np.min(all_d, axis = 0).ravel()

        else:
            l2_distance = np.linalg.norm(true_x_min - dic[seed_i], axis = 1).ravel()

        if l2_distance.shape[0] == l2_norm.shape[1]: # check whether matches, as gpyopt has errors
            l2_norm[seed_i,:] = l2_distance
        else:
            print("Seed " + str(seed_i) + " has errors in data.")
            l2_norm[seed_i,:] = np.zeros(l2_norm.shape[1])

    l2_norm = l2_norm[~np.all(l2_norm == 0, axis=1)] # remove rows with full zero (errors)

    return l2_norm


# Helper functions

def np_to_df(array):
    df = pd.DataFrame(array)
    df = df.stack()
    df = df.to_frame()
    df.index.names = (['seed', 'iters'])
    df.columns = ['values']
    df.reset_index(inplace = True)
    return df

def load_pickle(seed_size, current_seed, func, batch, heuristic, batch_size, target_val):
    if batch == False:
        dir_name = "Exp_Data/" + func + "," + str(seed_size) + "_seed,sequential/" \
        + str(current_seed) + "_seed/"
        file_name = dir_name + "sequential,intermediate_vars.pickle"

    else:
        dir_name = "Exp_Data/" + func + ',' + str(seed_size) + '_seed,' + str(batch_size) + '_batch_size/' \
        + str(current_seed) + "_seed/"
        file_name = dir_name + heuristic + ',intermediate_vars.pickle'
    with open(file_name, 'rb') as f:  # Python 3: open(..., 'rb')
        pickle_dict = pickle.load(f)
        value = pickle_dict[target_val]
    return value

def error_vs_iterations_v2(func = "egg", seed_size = 2, metrics = "IR", batch = False, batch_size = 2, heuristic = "kb"):
    if batch == False:
        dir_name = "Exp_Data/" + func + "," + str(seed_size) + "_seed,sequential/"
        filename = "A_results_" + metrics + ",sequential.npy"
        results = np.load(dir_name + filename)
    else:
        dir_name = "Exp_Data/" + func + "," + str(seed_size) + "_seed," + str(batch_size) + "_batch_size/"
        filename = "A_results_" + metrics + "," + heuristic + "_heuristic.npy"
        results = np.load(dir_name + filename)
        results = np.repeat(results, repeats = batch_size * np.ones(results.shape[1], dtype = int), axis = 1)

    df = np_to_df(results)
    return df

def min_y_hist(y_hist):
    """
    Takes y_hist and returns min_y_hist (same size), which is the minimum y_hist at each iteration
    """
    import copy

    y_copy = copy.deepcopy(y_hist)
    seed_size, iters = y_copy.shape

    for i in range(seed_size):
        current_min = y_copy[i,0]

        for j in range(iters):
            if y_copy[i,j] < current_min:
                current_min = y_copy[i,j]
            y_copy[i,j] = current_min
    return y_copy