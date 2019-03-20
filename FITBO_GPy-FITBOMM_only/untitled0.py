
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np

def gaussian(x,x0,sigma):
  return np.exp(-np.power((x - x0)/sigma, 2.)/2.)*(1/(np.sqrt(2*np.pi)*sigma))



mean_1 = 3
var_1 = 0.1

mean_2 = 4
var_2 = 0.2
mean_3 = 2
var_3 = 0.4

xgrid = np.linspace(0,5,5000)

f1 = gaussian(xgrid, mean_1, var_1)

f23 = (gaussian(xgrid, mean_2, var_2) + gaussian(xgrid, mean_3, var_3) ) /2

plt.plot(xgrid, f1)
plt.plot(xgrid, f23)


sns.lineplot(x = xgrid, y = f1)
sns.lineplot(x = xgrid, y = f23)