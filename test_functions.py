###########
# Benchmark functions for optimization tasks: Branin, Eggholder, Hartmann
###########
import numpy as np

def branin(x1, x2, a = 1, b = 5.1/((4*np.pi)**2), c = 5/np.pi, r = 6, s = 10, t = 1/(8*np.pi)):
	# Input domain [-5, 10] for x1, [0, 15] for x2
	# https://www.sfu.ca/~ssurjano/branin.html
	return a * (x2 - b * x1**2 + c*x1 - r) + s * (1 - t) * np.cos(x1) + s

def eggholder(x1, x2):
	# Input domain [-512, 512] for both dimensions
	# https://www.sfu.ca/~ssurjano/egg.html
	return -(x2 + 47) * np.sin(np.sqrt(abs(x2 + x1/2 + 47))) - x1 * np.sin(np.sqrt(abs(x1 - (x2 + 47))))

branin_minimiser = [(-np.pi, 12.275), (np.pi, 2.275), (9.42478, 2.475)]
branin_minimum = 0.397887

eggholder_minimiser = (512, 404.2319)
eggholder_minimum = -959.6407

