# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 16:13:03 2018

@author: jianhong
"""
import numpy as np

def kriging_believer(self, x):
    ''' Returns KB guess which is expectation of current GP normal for point x'''
    if self.GP_normal == []:
        kb_guess = np.random.rand() * abs(self.ub - self.lb) # Return random point there is no past Y
    else:
        kb_guess = self._marginalised_posterior_mean(x)
    return kb_guess

def constant_liar(self, x, cl_setting = "mean"):
    ''' Returns CL guess which is solely based on past Y's. Has 3 choices for settings: min, max, mean '''
    
    if cl_setting == "min":
        return min(self.Y)
    elif cl_setting == "max":
        return max(self.Y)
    else: 
        return np.mean(self.Y)