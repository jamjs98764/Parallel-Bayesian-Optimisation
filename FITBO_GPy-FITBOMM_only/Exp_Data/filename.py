# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 16:32:09 2018

@author: jianhong

Used to change filename and directories post-running

"""

import os

func = 'branin'
batch = '2'
dir_name = func + ',5_seed,' + batch + '_batch_size' 
os.chdir(dir_name)

# egg,30_seed,sequentialA_results_IR,cl-mean_heuristic.npy

# Creating new subfolders
dir_name = "Exp_Data/branin,30_seed,4_batch_size"
os.mkdir(dir_name)
for i in range(30):
	try:
	    os.mkdir(dir_name + "/" + str(i) + "_seed")
	except FileExistsError:
	    pass

# Moving intermediate vars
import shutil
import os
for i  in range(30):
    source = "hartmann,30_seed,sequential" + str(i) + "_seed/"
    dest = "hartmann,30_seed,4_batch_size/" + str(i) + "_seed/"    
    files = os.listdir(source)
    for f in files:
            shutil.move(source+f, dest)

"""
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
"""
    
    