"""
Box plots for epoch time
"""

import numpy as np
import seaborn as sns
import pandas as pd

df = pd.read_csv("training_time.csv")

df2 = df.drop([0,3,6,9], axis = 0)

fig = sns.boxplot(data = df2, palette = "Blues_d")


graph_title = "Training Times Across Various Batch Sizes in Stochastic Gradient Descent"
fig.set(ylabel = "Training Time per Epoch (seconds)", title = graph_title)
save_path = "Exp_Data/Graphs/run_time.png"
fig2 = fig.get_figure()
fig2.savefig(save_path)


"""
Calculate Idle Time
"""
cl_min_2 = np.load("Exp_Data/cifar10/FITBO/80_iter,batch_2,seed_3,cl-min,X_hist.npy")[:,:,0]
cl_min_2 = cl_min_2*256 + 2

time_mean = df.mean(axis = 0)
time_var = df.var(axis = 0)/2

x_axis = np.array([1,2,3,4,5,6,7,8]) # log scale

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

lg_mean = LinearRegression()
lg_mean.fit(x_axis.reshape(-1, 1), list(time_mean))

lg_var = LinearRegression()
lg_var.fit(x_axis.reshape(-1, 1), list(time_var))

# For FITBO CL-min 2
ttl_time_cl2 = np.zeros((3,43))
for i in range(3):
    ttl_time_cl2[i,:] = np.atleast_2d(lg_mean.predict(np.log2(cl_min_2[i]).reshape(-1, 1)))

sum_time_2 = np.sum(ttl_time_cl2, axis = 1)

def idle_time(row, batch_size):



# For FITBO CL-min 4
