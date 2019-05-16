
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import numpy as np
from tensorflow import set_random_seed
import cifar_utils
import pickle

np.set_printoptions(suppress=True)

initial_num = 3
a = np.load("allseed_normalised_initx.npy")
b = np.load("allseed_normalised_inity.npy")
x_init_dict = {}
y_init_dict = {}
x_init_dict[0] = a[0:3]
x_init_dict[1] = a[3:6]
x_init_dict[2] = a[6:9]
y_init_dict[0] = b[0:3]
y_init_dict[1] = b[3:6]
y_init_dict[2] = b[6:9]

from pyswarms.single.global_best import GlobalBestPSO
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

seed_size = 3
total_evals = 4

def pso_wrapper(seed_size, num_iters, batch_size):
    obj_func = cifar_utils.cifar_cnn_gpyopt
    d = 6
    x_min = np.zeros((1,6))
    x_max = np.ones((1,6))
    bounds = (x_min, x_max)

    # Creating dicts to store
    pos_dict = {}
    cost_dict = {}

    for seed_i in range(seed_size):
        np.random.seed(seed_i)
        set_random_seed(seed_i)
        
        # Generating initial samples
        init_pos = np.random.random((batch_size, d))
        init_pos[0] = x_init_dict[seed_i][0]
        init_pos[1] = x_init_dict[seed_i][1]

        if batch_size > 2:
        	init_pos[2] = x_init_dict[seed_i][2]
        
        # Running optimisation
        optimizer = GlobalBestPSO(n_particles = batch_size, dimensions=d, options=options, bounds=bounds, init_pos = None)
        best_cost, best_pos = optimizer.optimize(obj_func, num_iters + 1, seed_i)

        # Recording results
        pos_hist = optimizer.pos_history
        cost_hist = optimizer.cost_history

        pos_dict[seed_i] = pos_hist
        cost_dict[seed_i] = np.transpose(np.array([cost_hist]))

    return pos_dict, cost_dict


def saving_data_pso(pos_dict, cost_dict, batch_size):
    """
    For saving data

    pos_dict =
    cost_dict = y_opt
    """
    dir_name = 'Exp_Data/boston_gbr/pso/' + str(seed_size) + '_seed,' + str(batch_size) + '_batch/'
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

for batch in [2]:
    a, b = pso_wrapper(seed_size, total_evals, batch)
    saving_data_pso(a, b, batch)
