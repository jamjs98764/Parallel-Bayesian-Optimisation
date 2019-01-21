# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 14:56:44 2018

@author: jianhong
"""
import numpy as np
import seaborn as sns
import pandas as pd

#####
# 1. Error against number of iterations
#####

def error_vs_iterations(func = "egg", seed_size = 2, metrics = "IR", batch = False, batch_size = 2, heuristic = "kb"):
    if batch == False:
        dir_name = func + "," + str(seed_size) + "_seed," + str(batch_size) + "_batch_size/"
        filename = "A_results_" + metrics + ",sequential.npy"
        results = np.load(dir_name + filename)
    else:
        dir_name = func + "," + str(seed_size) + "_seed," + str(batch_size) + "_batch_size/"
        filename = "A_results_" + metrics + "," + heuristic + "_heuristic.npy"
        results = np.load(dir_name + filename)
        results = np.repeat(results, repeats = batch_size * np.ones(results.shape[1], dtype = int), axis = 1)
    
    df = pd.DataFrame(results)
    df = df.stack()
    df = df.to_frame()
    df.index.names = (['seed', 'iters'])
    df.columns = ['values']
    df.reset_index(inplace = True)
    return df
    
    #return results, seed_size, num_iterations

seed_size = 30
batch_sizes = [2, 8]
test_funcs = ["egg", "branin", "hartmann"]
metrics = ["IR", "L2"]

plot_choice = {
        "seq_results": 1, 
        "random_results": 1,
        "random1_results": 0,
        "kb_results": 1,
        "cl-mean_results": 1, 
        "cl-min_results": 1,
        "cl-max_results": 0
        }

label_lookup = {
        "seq_results": "Sequential", 
        "random_results": "Fully Random",
        "random1_results": "Random excl. 1st",
        "kb_results": "Kriging Believer",
        "cl-mean_results": "Constant Liar (Mean)", 
        "cl-min_results": "Constant Liar (Min)",
        "cl-max_results": "Constant Liar (Max)"
        }

metric_lookup = {
        "L2": "L2 norm between actual and guessed x*",
        "IR": "Absolute distance between actual and guessed y*",
        }

def plot_error_vs_iterations(seed_size, batch_sizes, test_funcs, metrics, plot_choice):   
    # Loads, plots and saves graphs     
    for metric in metrics:
        for batch_size in batch_sizes:
            for func in test_funcs:
                seq_results = error_vs_iterations(batch = False, func = func, batch_size = batch_size, seed_size = seed_size)    
                random_results = error_vs_iterations(func = func, batch = True, batch_size = batch_size, seed_size = seed_size, heuristic = "random")
                random1_results = error_vs_iterations(func = func, batch = True, batch_size = batch_size, seed_size = seed_size, heuristic = "random_except_1st")
                kb_results = error_vs_iterations(func = func, batch = True, batch_size = batch_size, seed_size = seed_size, heuristic = "kb")
                cl_mean_results = error_vs_iterations(func = func, batch = True, batch_size = batch_size, seed_size = seed_size, heuristic = "cl-mean")
                cl_min_results = error_vs_iterations(func = func, batch = True, batch_size = batch_size, seed_size = seed_size, heuristic = "cl-min")
                cl_max_results = error_vs_iterations(func = func, batch = True, batch_size = batch_size, seed_size = seed_size, heuristic = "cl-max")
                        
                for key, value in plot_choice.items():
                    if value == 1:
                        fig = sns.lineplot(x = 'iters', y = 'values', data = key, err_style = "band", label = label_lookup[key])
                
                graph_title = str(batch_size) + "-Batch on "+ str(func) + " Function"
                fig.set(xlabel = "No. of Iterations", ylabel = metric_lookup[metric], title = graph_title)


#####
# 1. Error against number of iterations
#####
