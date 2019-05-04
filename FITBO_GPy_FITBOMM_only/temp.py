# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle

save_name = "Exp_Data/cifar10/FITBO/"
file_name = "batch_2,gpyopt_EI_LP,results_vars.pickle"

"""
a = np.load("Exp_Data/temp/2_ei_a.npy")
b = np.load("Exp_Data/temp/2_ei_b.npy")
c = np.load("Exp_Data/temp/2_ei_c.npy")
d = np.load("Exp_Data/temp/2_ei_d.npy")
e = np.load("Exp_Data/temp/2_ei_e.npy")
f = np.load("Exp_Data/temp/2_ei_f.npy")
g = np.load("Exp_Data/temp/2_ei_g.npy")
h = np.load("Exp_Data/temp/2_ei_h.npy")
i = np.load("Exp_Data/temp/2_ei_i.npy")

X = {0:a, 1:d, 2:g}
min_y = {0:b, 1:e, 2:h}
eval_record = {0:c, 1:f, 2:i}

temp_dict ={"X": X,
            "min_y": min_y,
            "eval_record": eval_record}

with open(save_name+file_name, 'wb') as f:
    pickle.dump(temp_dict, f)


a = np.load(save_name + "2_batchnorm_batch_2,seed_3,cl-min,Y_optimum.npy")

b = np.load("Exp_Data/temp/80_iter,seed_12,2_batch,clmin,Y_opt.npy")
c = np.load("Exp_Data/temp/80_iter,seed_3rd,2_batch,clmin,Y_opt.npy")

d = np.vstack((b[0], b[1], c[2]))

np.save(save_name + "80_iter,batch_1,seed_3,cl-min,Y_optimum.npy",d)

b = np.load("Exp_Data/temp/80_iter,seed_3,4_batch,clmin,Y_opt.npy")

np.save(save_name + "80_iter,batch_4,seed_3,cl-min,Y_optimum.npy",b)
"""

with open("Exp_Data/temp/random_y_hist.pickle", 'rb') as f:  # Python 3: open(..., 'rb')
    pickle_dict = pickle.load(f)
    print(pickle_dict.keys())