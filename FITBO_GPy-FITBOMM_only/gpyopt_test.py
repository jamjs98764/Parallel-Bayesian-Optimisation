import GPyOpt

# --- Objective function
objective_true  = GPyOpt.objective_examples.experiments2d.eggholder()                 # true function
objective_noisy = GPyOpt.objective_examples.experiments2d.branin(sd = 0.1)         # noisy version
bounds = objective_noisy.bounds 

# --- Objective function
objective_true  = GPyOpt.objective_examples.experiments2d.eggholder()                 # true function
objective_noisy = GPyOpt.objective_examples.experiments2d.branin(sd = 0.1)         # noisy version
bounds = objective_noisy.bounds 

batch_size = 4
num_cores = 4

domain = [{'name': 'var_1', 'type': 'continuous', 'domain': bounds[0]}, ## use default bounds
          {'name': 'var_2', 'type': 'continuous', 'domain': bounds[1]}]

from numpy.random import seed
seed(123)
BO_demo_parallel = GPyOpt.methods.BayesianOptimization(f=objective_noisy.f,  
                                            domain = domain,                  
                                            acquisition_type = 'EI_MCMC',
                                            model_type = "GP_MCMC",              
                                            normalize_Y = True,
                                            initial_design_numdata = 10,
                                            evaluator_type = 'local_penalization',
                                            batch_size = batch_size,
                                            num_cores = num_cores,
                                            acquisition_jitter = 0)    


max_iter = 10                                        
BO_demo_parallel.run_optimization(max_iter)

BO_demo_parallel.plot_acquisition()

BO_demo_parallel.plot_convergence()