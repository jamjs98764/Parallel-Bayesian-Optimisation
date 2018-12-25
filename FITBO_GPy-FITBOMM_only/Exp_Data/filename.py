# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 16:32:09 2018

@author: jianhong
"""

import os

func = 'branin'
batch = '2'
dir_name = func + ',5_seed,' + batch + '_batch_size' 
os.chdir(dir_name)

# Changing L2 to IR
for file in os.listdir():
    src=file
    
    if src[0:13] == 'A_results_L2,':
           change = "B_results_IR," + src[13:]
           os.rename(src,change)

    if src[0:13] == 'A_results_IR,':
           change = "B_results_L2," + src[13:]
           os.rename(src,change)

# Changing B back to A
for file in os.listdir():
    src=file
    
    if src[0:2] == 'A,':
        change = "A" + src[2:]
        os.rename(src,change)
        
    
    