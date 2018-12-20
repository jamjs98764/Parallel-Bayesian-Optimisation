# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 14:56:44 2018

@author: jianhong
"""
import numpy as np
import seaborn as sns
#####
# Error against number of iterations
#####

def error_vs_iterations(func = "egg", metrics = "L2", batch = False, batch_size = 2, heuristic = "kb", aggr_seed = "mean", 
                        color = 'g'):
    if batch == False:
        filename = 'Exp_Data/' + func + "FITBOMM_results_" + metrics + ".npy"
        results = np.load(filename)
    else:
        filename = 'Exp_Data/' + func + "FITBOMM_results_" + metrics + "," + str(batch_size) + "_batch_size," + heuristic + "_heuristic.npy"
        results = np.load(filename)
        results = np.repeat(results, repeats = batch_size * np.ones(results.shape[1], dtype = int), axis = 1)
    seed_size = results.shape[1]
    num_iterations = results.shape[0]
    
    # fig = sns.lineplot(x = np.arange(num_iterations), y = results)

    fig = sns.tsplot(data = results)
    fig.set(xlabel = "No of Iterations", ylabel = "Error")
        
#error_vs_iterations()

error_vs_iterations(batch = True)
    
error_vs_iterations(batch = True, heuristic = "random", color = 'b')
    
    
    
    