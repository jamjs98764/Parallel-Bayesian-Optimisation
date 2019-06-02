# 4YP
- University of Oxford Engineering Science 4th year project on Bayesian Optimization
- Report titled "Parallel Bayesian Optimisation for Hyperparameter Tuning of Machine Learning Models"
- Winner of "EIBF Prize for best Engineers in Business Project" from the Department of Engineering Science, Oxford

# Code Base
- Dir "Exp_Data" stores saved results. Notebooks "Error vs Iterations" plots the results found in the report.
- Scripts "class_FITBOMM..." contains code for various versions of FITBO
- Scripts "FITBO_BO_Exp..." contains code to run various experiments to generate results stored in Exp_Data
- Script "batch_proposals" contains code for Kriging Believer and Constant Liar heuristics for parallel BO

# Acknowledgements
- Original FITBO code forked from Binxin Ru's Github: https://github.com/rubinxin/FITBO
- Original GPyOpt code forked from SheffieldML's Github: https://github.com/SheffieldML/GPyOpt
