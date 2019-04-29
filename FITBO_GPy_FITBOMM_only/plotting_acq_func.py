#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 16:16:51 2019

@author: jian
"""

from plotting_utilities import *
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def return_acq_func(func, batch_size, batch_num, iter_num, heuristic = "cl-min"):
    dir_name = "Exp_Data/" + str(func) + ",1_seed," + str(batch_size) + "_batch_size/"
    file_name = dir_name + heuristic + ",batch_" + str(batch_num) + ",iter_" + str(iter_num) + ",acq_func.npy"
    acq_func = np.load(file_name)
    return acq_func

def load_pickle_x(seed_size, seed, func, batch, heuristic, batch_size):
    """
    Loads past queries, excluding initial points
    Returns array of x1 and x2
    """
    loaded_x = load_pickle(seed_size, seed, func, batch, heuristic, batch_size, "X")
    if func == "hartmann":
        init_x = 6
    else:
        init_x = 3

    x = loaded_x[init_x:,:]

    return x

# Minimiser x value
min_x_dict = {
    "hartmann": np.array([[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]]),
    "egg": np.array([[1.0, 0.7895]]),
    "branin": np.array([[0.1239, 0.8183],[0.5428, 0.1517],[0.9617, 0.1650]])
}

#########
# Branin
#########
cur_func = "branin"
batch_size = 2
heuristic = "cl-min"
cur_batch = 1

a = return_acq_func(cur_func, batch_size, cur_batch, 0, heuristic = heuristic)
b = return_acq_func(cur_func, batch_size, cur_batch, 1, heuristic = heuristic)
c = return_acq_func(cur_func, batch_size, cur_batch + 1, 0, heuristic = heuristic)
d = return_acq_func(cur_func, batch_size, cur_batch + 1, 1, heuristic = heuristic)

x_hist = load_pickle_x(1, 0, cur_func, True, heuristic, batch_size)

diff_ba = b - a
diff_cb = c - b
diff_dc = d - c

grid_size = 100
x = np.linspace(0, 1.0, grid_size)
y = np.linspace(0, 1.0, grid_size)

###
# Plot 1 (Between batches)
###
lons = x_hist[2:4,0]
lats = x_hist[2:4,1]

fig = plt.figure()
plt.contourf(x, y, diff_cb, cmap = "viridis")
plt.colorbar()
plt.title("Change in FITBO Acq Func between \n Iterations 4 & 5 for 2-Batch Branin")
plt.scatter(lons, lats, marker = '^', label='Queries')
plt.gca().set_aspect('equal', adjustable='box')

# Adding global minimum
x_min = min_x_dict[cur_func]
if (len(x_min)>1):
        plt.plot(np.array(x_min)[:,0], np.array(x_min)[:,1], 'm.', markersize=15, label='Global Minimums')
else:
    plt.plot(x_min[0][0], x_min[0][1], 'm.', markersize=10, label='Global Minimum')

plt.legend(loc='center left', bbox_to_anchor=(1.35, 0.5))
plt.tight_layout()
fig.savefig('Exp_Data/Graphs/branin_acq_betweenbatch.png', transparent=True)
plt.show()

###
# Plot 2 (Within batch)
###
lons = x_hist[2,0]
lats = x_hist[2,1]

fig = plt.figure()
plt.contourf(x, y, diff_ba, cmap = "viridis")
plt.colorbar()
plt.title("Change in FITBO Acq Func between \n Iterations 3 & 4 for 2-Batch Branin")
plt.scatter(lons, lats, marker = '^', label='Queries')
plt.gca().set_aspect('equal', adjustable='box')

# Adding global minimum
x_min = min_x_dict[cur_func]
if (len(x_min)>1):
        plt.plot(np.array(x_min)[:,0], np.array(x_min)[:,1], 'm.', markersize=15, label='Global Minimums')
else:
    plt.plot(x_min[0][0], x_min[0][1], 'm.', markersize=10, label='Global Minimum')

plt.legend(loc='center left', bbox_to_anchor=(1.35, 0.5))
plt.tight_layout()
fig.savefig('Exp_Data/Graphs/branin_acq_withinbatch.png', transparent=True)
plt.figure()
plt.show()


#########
# GPyOpt - Branin
#########
cur_func = "branin"
batch_size = 2

a = np.load("1acq_func.npy")
b = np.load("2acq_func.npy")
c = np.load("3acq_func.npy")

diff_ba = b - a
diff_cb = c - b

grid_size = 100
x = np.linspace(0, 1.0, grid_size)
y = np.linspace(0, 1.0, grid_size)

###
# Plot 1 (Between batches)
###
lons = [1.0]
lats = [0.0]

fig = plt.figure()
plt.contourf(x, y, diff_cb, cmap = "viridis")
plt.colorbar()
plt.title("Change in EI-LP Acq Func between \n Iterations 3 & 4 for 2-Batch Branin")
plt.scatter(lons, lats, marker = '^', label='Queries')
plt.gca().set_aspect('equal', adjustable='box')

# Adding global minimum
x_min = min_x_dict[cur_func]
if (len(x_min)>1):
        plt.plot(np.array(x_min)[:,0], np.array(x_min)[:,1], 'm.', markersize=15, label='Global Minimums')
else:
    plt.plot(x_min[0][0], x_min[0][1], 'm.', markersize=10, label='Global Minimum')

plt.legend(loc='center left', bbox_to_anchor=(1.35, 0.5))
plt.tight_layout()
fig.savefig('Exp_Data/Graphs/gpy-ei_branin_acq_betweenbatch.png', transparent=True)
plt.show()




