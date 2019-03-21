import GPy
from Test_Funcs import egg,hartmann,branin,func1D
import numpy as np

X = np.random.uniform(0.,1.,(500,2))
sigma0= 1.0e-3

branin_y = branin(X) + sigma0 * np.random.randn()

kernel_branin = GPy.kern.RBF(input_dim=2, ARD = True)
m_branin = GPy.models.GPRegression(X,branin_y,kernel_branin)
m_branin.optimize(messages=True)
print(m_branin.kern.lengthscale)
"""
  index  |  GP_regression.rbf.lengthscale  |  constraints  |  priors
  [0]    |                     0.36986751  |      +ve      |
  [1]    |                     6.24718483  |      +ve      |
"""

egg_y = egg(X)
kernel_egg = GPy.kern.RBF(input_dim=2, ARD = True)
m_egg = GPy.models.GPRegression(X, egg_y, kernel_egg)
m_egg.optimize(messages=True)
print(m_egg.kern.lengthscale)

"""
  index  |  GP_regression.rbf.lengthscale  |  constraints  |  priors
  [0]    |                     0.05293395  |      +ve      |
  [1]    |                     0.05648268  |      +ve      |
"""
