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