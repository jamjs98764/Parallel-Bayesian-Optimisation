#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 15:22:03 2019

@author: jian
"""
from plotting_utilities import *
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set()


#########
# Branin
#########
num_init = 3

X_hist_4 = load_pickle(50, 0, "branin", True, "kb", 4, "X")
X_hist_4 = X_hist_4[num_init:]

X_hist_1 = load_pickle(50, 0, "branin", False, "kb", 1, "X")
X_hist_1 = X_hist_1[num_init:]

diff_4 = np.diff(X_hist_4, axis = 0)
diff_1 = np.diff(X_hist_1, axis = 0)

norm_4 = np.sum(np.abs(diff_4)**2,axis=-1)**(1./2)
norm_1 = np.sum(np.abs(diff_1)**2,axis=-1)**(1./2)

x = np.arange(39)
plt.plot(x, norm_1)
plt.plot(x, norm_4)