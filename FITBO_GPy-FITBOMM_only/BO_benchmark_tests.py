# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 15:18:49 2019

@author: jianhong
"""

#####
# Running tests
#####

import GPyOpt
import numpy as np
import scipy as sp
from Test_Funcs import egg,hartmann,branin,func1D

var_noise = 1.0e-3 # y_next = self.func(x_next) + np.random.normal(0, self.var_noise, len(x_next)) 