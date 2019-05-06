"""
Box plots for epoch time
"""

import numpy as np
import seaborn as sns
import pandas as pd

df = pd.read_csv("training_time.csv")

df2 = df.drop([0,3,6,9], axis = 0)

fig = sns.boxplot(data = df2, palette = "Blues_d")

"""
graph_title = "Training Times Across Various Batch Sizes in Stochastic Gradient Descent"
fig.set(ylabel = "Training Time per Epoch (seconds)", title = graph_title)
save_path = "Exp_Data/Graphs/run_time.png"
fig2 = fig.get_figure()
fig2.savefig(save_path)
"""


# Creating linear model to extrapolate time
x_axis = np.array([1,2,3,4,5,6,7,8]) # log scale

time_mean = df.mean(axis = 0)
time_var = df.var(axis = 0)/2

from sklearn.linear_model import LinearRegression

lg_mean = LinearRegression()
lg_mean.fit(x_axis.reshape(-1, 1), list(time_mean))

lg_var = LinearRegression()
lg_var.fit(x_axis.reshape(-1, 1), list(time_var))

"""
Calculate Idle Time
"""

def idle_time(row, batch_size):
    subarrays = np.split(row, 80/batch_size) # list of subarrays
    subarray_b = []
    for item in subarrays:
        subarray_b.append(abs(item - max(item)))

    subarray_c = np.array(subarray_b)
    ttl_idle_time = subarray_c.sum()
    return ttl_idle_time

# For FITBO CL-min 2

cl_min_2 = np.load("Exp_Data/cifar10/FITBO/80_iter,batch_2,seed_3,clmin,X_hist.npy")[:,:,0]
cl_min_2 = cl_min_2*256 + 2

ttl_time_cl2 = np.zeros((3,83))
for i in range(3):
    ttl_time_cl2[i,:] = np.atleast_2d(lg_mean.predict(np.log2(cl_min_2[i]).reshape(-1, 1)))

ttl_time_cl2 = ttl_time_cl2[:,3:] # remove initial samples

sum_time_2 = np.sum(ttl_time_cl2, axis = 1)

idle_time_2 = []
for row in ttl_time_cl2:
    idle_time_2.append(idle_time(row, 2))
idle_time_2 = np.array(idle_time_2)


# For FITBO CL-min 4

cl_min_4 = np.load("Exp_Data/cifar10/FITBO/80_iter,seed_3,4_batch,clmin,X_hist.npy")[:,:,0]
cl_min_4 = cl_min_4*256 + 2

ttl_time_cl4 = np.zeros((3,83))
for i in range(3):
    ttl_time_cl4[i,:] = np.atleast_2d(lg_mean.predict(np.log2(cl_min_4[i]).reshape(-1, 1)))

ttl_time_cl4 = ttl_time_cl4[:,3:] # remove initial samples

sum_time_4 = np.sum(ttl_time_cl4, axis = 1)

idle_time_4 = []
for row in ttl_time_cl4:
    idle_time_4.append(idle_time(row, 4))
idle_time_4 = np.array(idle_time_4)


# For EI-LP 2
import pickle

with open("Exp_Data/cifar10/gpyopt/80iter,batch_2,gpyopt_EI_LP,results_vars.pickle", 'rb') as f:
    pickle_dict = pickle.load(f)
    x_ei2 = pickle_dict["eval_record"]

x_ei2_array = np.zeros((3,80,6))
for i in range(3):
    x_ei2_array[i] = x_ei2[i]

x_ei2_array = x_ei2_array[:,:,0]
x_ei2_array = x_ei2_array*256 + 2

ttl_time_ei2 = np.zeros((3,80))

for i in range(3):
    ttl_time_ei2[i,:] = np.atleast_2d(lg_mean.predict(np.log2(x_ei2_array[i]).reshape(-1, 1)))

sum_time_ei2 = np.sum(ttl_time_ei2, axis = 1)

idle_time_ei2 = []
for row in ttl_time_ei2:
    idle_time_ei2.append(idle_time(row, 2))
idle_time_ei2 = np.array(idle_time_ei2)


# For EI-LP 4

with open("Exp_Data/cifar10/gpyopt/80iter,batch_4,gpyopt_EI_LP,results_vars.pickle", 'rb') as f:
    pickle_dict = pickle.load(f)
    x_ei4 = pickle_dict["eval_record"]

x_ei4_array = np.zeros((3,80,6))
for i in range(3):
    x_ei4_array[i] = x_ei4[i]

x_ei4_array = x_ei4_array[:,:,0]
x_ei4_array = x_ei4_array*256 + 2

ttl_time_ei4 = np.zeros((3,80))

for i in range(3):
    ttl_time_ei4[i,:] = np.atleast_2d(lg_mean.predict(np.log2(x_ei4_array[i]).reshape(-1, 1)))

sum_time_ei4 = np.sum(ttl_time_ei4, axis = 1)

idle_time_ei4 = []
for row in ttl_time_ei4:
    idle_time_ei4.append(idle_time(row, 4))
idle_time_ei4 = np.array(idle_time_ei4)


import numpy as np
import matplotlib.pyplot as plt


"""
Stacked Bar Plot
"""
N = 4 # ORDER: fitbo 2, ei2, fitbo4, ei4
active_time = (sum_time_2.mean(), sum_time_ei4.mean(), sum_time_4.mean(), sum_time_ei4.mean())
idle_time = (idle_time_2.mean(), idle_time_ei2.mean(), idle_time_4.mean(), idle_time_ei4.mean())
active_var = (np.ptp(sum_time_2)/2, np.ptp(sum_time_ei4)/2, np.ptp(sum_time_4)/2, np.ptp(sum_time_ei4)/2)
idle_var = (np.ptp(idle_time_2)/2, np.ptp(idle_time_ei2)/2, np.ptp(idle_time_4)/2, np.ptp(idle_time_ei4)/2)

ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, active_time, width, yerr=active_var, color = '#4169E1')
p2 = plt.bar(ind, idle_time, width,
             bottom=active_time, yerr=idle_var, color = "#AFEEEE")

plt.ylabel('Total Training Time for 80 Iterations (seconds)', fontsize=12)
plt.title('Active and Idle Training due to Synchronous Setting', fontsize=12)
plt.xticks(ind, ('FITBO 2-Batch', 'EI-LP 2-Batch', 'FITBO 4-Batch', 'EI-LP 4-Batch'), fontsize=12.5)
plt.legend((p1[0], p2[0]), ('Active', 'Idle'))

plt.show()